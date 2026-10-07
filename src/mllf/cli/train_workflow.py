"""Top-level orchestration for online CB-policy training.

The public entry point for the multi-system UniMol CB policy training driver
described in :doc:`workflow` (docs/workflow.rst): given a workflow YAML
config, builds/loads per-system combo manifests, initializes the policy
(optionally warm-started or resumed from checkpoint), and runs the
(optionally curriculum-staged) training loop via
``mllf.cb.reinforce_trainer.train_epoch``.

Usage::

    python -m mllf.cli.train_workflow my_workflow_config.yaml

See ``training/workflow.py`` for a thin repo-local wrapper around this same
entry point, and :doc:`workflow` for the full configuration reference.
"""
from pathlib import Path
from typing import Dict, List
import collections
import copy
import concurrent.futures
import json
import re
import sys
import threading
import warnings

import numpy as np
import torch
import yaml

from mllf.cb.curriculum import (
    apply_stage_combo_filter,
    build_system_index,
    sample_combos_stratified,
)
from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.hindsight import combo_node_labels, pair_attention
from mllf.cb.policy import ContextBaseline, UnimolPolicy
from mllf.cb.reinforce_trainer import append_summary, format_pair_success, train_epoch
from mllf.cb.workflow_utils import (
    action_std_score,
    bayesian_variance_score,
    load_bandit_state,
    save_bandit_state,
    select_combos_by_uncertainty,
)
from mllf.file_handling.generate_combinations import find_site_sub_files
from mllf.file_handling.manifests import (
    build_combo_dir_lookup,
    build_system_manifest,
    count_combinations_fast,
    create_and_load_system_manifest,
    ensure_combo_dir_exists,
)


def run_training(config_path: str) -> None:
    """Run the full online CB-policy training workflow from a YAML config.

    This is everything the training driver does end to end: build/load
    per-system combo manifests, initialize the ``UnimolPolicy`` (optionally
    warm-started from a pretrain checkpoint or resumed from the latest
    ``checkpoint_*.pt``), run the (optionally curriculum-staged) epoch loop
    via ``mllf.cb.reinforce_trainer.train_epoch``, and checkpoint along the
    way.

    Args:
        config_path: Path to a workflow YAML config (see docs/workflow.rst).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")

    verbose = config.get('verbose', False)

    # Suppress per-occurrence warnings; print deduplicated summary at end.
    _warning_counts = collections.Counter()
    _warning_example = {}
    _warning_lock = threading.Lock()
    _path_re_w = re.compile(r'/[^\s:*?"<>|]+')
    _orig_showwarning = warnings.showwarning

    def _collect_warning(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        template = _path_re_w.sub('<path>', text)
        key = (category.__name__, template)
        with _warning_lock:
            _warning_counts[key] += 1
            if key not in _warning_example:
                _warning_example[key] = (text, filename, lineno)

    warnings.showwarning = _collect_warning

    # ------------------------------------------------------------------
    # Step 1: Generate combination directories for every system
    # ------------------------------------------------------------------
    systems_cfg = config.get('systems', [])
    if not systems_cfg:
        raise ValueError("Config must contain a 'systems:' list with at least one entry.")

    # Map: combo_path_str -> system_config dict (for build_graph_and_data)
    combo_system_map: Dict[str, Dict] = {}
    all_combo_paths: List[str] = []

    seed = config.get('seed', 42)

    print(f"\n=== Building system manifests ({len(systems_cfg)} systems) ===")
    large_threshold = config.get('large_combo_threshold', 10_000)
    creation_workers = config.get('max_manifest_workers', 4)
    manifest_max_subs = config.get('max_subs_per_site', 10)
    large_site_threshold = config.get('large_site_sub_threshold', 12)
    large_site_max_subs = config.get('large_site_max_subs', 6)
    max_combos_per_system = config.get('max_combos_per_system', None)
    print(f"  max_subs_per_site : {manifest_max_subs} (default); "
          f"auto-reduced to {large_site_max_subs} for any site with > {large_site_threshold} subs")
    if max_combos_per_system is not None:
        print(f"  max_combos_per_system : {max_combos_per_system:,} (paths loaded per system)")

    # Separate systems into three buckets:
    #   existing  – manifest already on disk (instant load)
    #   small     – needs creation, estimated combos < threshold (safe to parallelise)
    #   large     – needs creation, estimated combos >= threshold (run alone, after all smalls)
    results_by_idx: Dict[int, Dict] = {}
    eff_max_by_idx: Dict[int, int] = {}   # resolved max_subs_per_site per system
    small_creation: List[tuple] = []   # (orig_idx, sys_entry)
    large_creation: List[tuple] = []   # (orig_idx, sys_entry, est_count)

    for i, sys_entry in enumerate(systems_cfg):
        manifest_path = Path(sys_entry['input_dir']) / 'manifest.txt'
        if manifest_path.exists():
            results_by_idx[i] = build_system_manifest(
                sys_entry, max_combos=max_combos_per_system, seed=seed)
        else:
            # Resolve effective max_subs_per_site for this system.
            # Per-system YAML key takes precedence; otherwise check site sub counts.
            if 'max_subs_per_site' in sys_entry:
                eff_max = sys_entry['max_subs_per_site']
            else:
                found = find_site_sub_files(Path(sys_entry['input_dir']))
                max_site_subs = max((len(subs) for subs in found.values()), default=0)
                if max_site_subs > large_site_threshold:
                    eff_max = large_site_max_subs
                    sys_name = sys_entry.get('name', 'unknown')
                    print(f"  [{sys_name}] site has {max_site_subs} subs > {large_site_threshold}; "
                          f"auto-reducing max_subs_per_site to {eff_max}")
                else:
                    eff_max = manifest_max_subs
            eff_max_by_idx[i] = eff_max

            est = count_combinations_fast(sys_entry, max_subs_per_site=eff_max)
            if est >= large_threshold:
                large_creation.append((i, sys_entry, est))
            else:
                small_creation.append((i, sys_entry))

    if small_creation:
        w = min(creation_workers, len(small_creation))
        print(f"  {len(small_creation)} small system(s) (< {large_threshold:,} combos): "
              f"creating in parallel ({w} workers)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=w) as executor:
            future_to_idx = {
                executor.submit(
                    create_and_load_system_manifest, s,
                    eff_max_by_idx[i], max_combos_per_system, seed
                ): i
                for i, s in small_creation
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                results_by_idx[future_to_idx[future]] = future.result()

    if large_creation:
        if small_creation:
            print(f"  All {len(small_creation)} small system(s) complete. "
                  f"Now creating {len(large_creation)} large system(s) one at a time.")
        for i, sys_entry, est in large_creation:
            sys_name = sys_entry.get('name', sys_entry.get('input_dir', 'unknown'))
            print(f"  Large system (~{est:,} combos): {sys_name} — creating alone")
            results_by_idx[i] = create_and_load_system_manifest(
                sys_entry, max_subs_per_site=eff_max_by_idx[i],
                max_combos=max_combos_per_system, seed=seed)

    # Print logs and build combo pool in original system order.
    for i in range(len(systems_cfg)):
        result = results_by_idx[i]
        for line in result['log']:
            print(line)
        for cp in result['combos']:
            combo_system_map[cp] = result['sys_entry']
        all_combo_paths.extend(result['combos'])

    print(f"\n=== Total combos across all systems: {len(all_combo_paths)} ===")

    # Validation is now handled per-epoch via held-out validation_fraction
    # (deterministic sampling from active_train_combos at epoch 0).
    # See: validation_fraction and eval_freq config parameters.
    train_combos = all_combo_paths

    _curriculum_cfg = config.get('curriculum', {})
    if _curriculum_cfg.get('enabled', False):
        _stages = _curriculum_cfg.get('stages', [])
        print(f"\n=== Curriculum Learning Enabled: {len(_stages)} stages ===")
        for _i, _s in enumerate(_stages, 1):
            print(f"  Stage {_i}: {_s['name']} — "
                  f"{_s.get('min_subs_per_site', 1)}-{_s.get('max_subs_per_site', '?')} subs/site, "
                  f"{_s.get('min_sites', 1)}-{_s.get('max_sites', '?')} sites, "
                  f"{_s['epochs']} epochs")

    # ------------------------------------------------------------------
    # Step 3: Initialize Uni-Mol + UnimolPolicy
    # ------------------------------------------------------------------
    print("\n=== Initializing Uni-Mol CB Policy ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Using pretrained Uni-Mol representation...")

    # Uni-Mol representation is loaded on-demand by compute functions
    # No explicit encoder model needed here

    train_config = config.get('training', {})
    policy_config = train_config.get('policy', {})
    unimol_dim = policy_config.get('unimol_dim', 512)  # Dimension of each Uni-Mol embedding (512D)
    mlp_hidden = policy_config.get('mlp_hidden', 64)
    num_bias_types = 4  # Linear, quadratic, skew, end

    # NeuralLinear + Thompson Sampling: opt-in bandit config, sibling to
    # curriculum:/reward:/unimol:. Default 'reinforce' reproduces today's
    # behavior exactly when the `bandit:` block is absent from the config.
    bandit_cfg = config.get('bandit', {})
    algorithm = bandit_cfg.get('algorithm', 'reinforce')
    use_bayesian_heads = policy_config.get('use_bayesian_heads', False)
    if use_bayesian_heads and algorithm != 'neurallinear_ts':
        print("  Warning: training.policy.use_bayesian_heads=True but "
              "bandit.algorithm != 'neurallinear_ts' — the policy will have "
              "Bayesian heads but nothing will train them online.")

    # Initialize policy with dual embeddings mode enabled
    # Dual mode uses [ligand_only (512D) + full (512D)] = 1024D edge inputs
    # This matches pretraining configuration where:
    #   - edge input = [diff_ligand, mean_full]
    #   - diff_ligand captures substituent-specific info (antisymmetric)
    #   - mean_full captures environmental context (symmetric)
    policy = UnimolPolicy(
        unimol_dim=unimol_dim,
        mlp_hidden=mlp_hidden,
        mlp_out_dim=num_bias_types,
        use_dual_embeddings=True,  # Enable dual embedding mode for online training
        use_bayesian_heads=use_bayesian_heads,
        bayesian_prior_precision=bandit_cfg.get('prior_precision', 1.0),
    ).to(device)

    # Optionally freeze bias heads (policy.freeze_heads). Not needed for skew /
    # end any more: compute_pair_reward() now gives each its own observable
    # from the λ trajectory (the pair's skew and end asymmetries of the biased
    # profile; see parse_pair_profile). Only when an output has no λ trajectory
    # do those dims fall back to proxies built from the linear/quadratic
    # signals, where freezing can still be worth considering.
    _bias_dim_names = ['linear', 'quadratic', 'skew', 'end']
    freeze_heads = policy_config.get('freeze_heads', [])
    if freeze_heads:
        for _name in freeze_heads:
            if _name not in _bias_dim_names:
                print(f"  Warning: unknown freeze_heads entry '{_name}' — skipping")
                continue
            _idx = _bias_dim_names.index(_name)
            for p in policy.edge_mlp.mlps[_idx].parameters():
                p.requires_grad = False
        print(f"Frozen bias heads (no online gradient): {freeze_heads}")

    optimizer_config = train_config.get('optimizer', {})
    base_lr = optimizer_config.get('lr', 0.001)
    
    # Create optimizer for EdgeValueMLP parameters (trainable ones only —
    # frozen heads, if any, are excluded so Adam doesn't keep dead moment
    # buffers around for them).
    rl_optimizer = torch.optim.Adam(
        [p for p in policy.edge_mlp.parameters() if p.requires_grad], lr=base_lr
    )
    rl_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        rl_optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7, verbose=False,
    )

    print(f"Uni-Mol policy: {sum(p.numel() for p in policy.parameters())} params")
    print(f"Uni-Mol dimension: {unimol_dim}")
    print(f"MLP hidden dimension: {mlp_hidden}")

    # Context-dependent REINFORCE baseline (opt-in, REINFORCE-mode only --
    # see ContextBaseline docstring and train_epoch's history_replay_cfg/
    # baseline_net params). Replaces the pooled scalar-per-bias-type EMA
    # baseline with one conditioned on edge features + system identity, so
    # "was this action good" is judged relative to what's typical for THAT
    # system/context, not relative to every system pooled together.
    context_baseline_cfg = train_config.get('context_baseline', {})
    system_to_idx = build_system_index(systems_cfg)
    baseline_net = None
    baseline_optimizer = None
    if context_baseline_cfg.get('enabled', False) and algorithm != 'neurallinear_ts':
        baseline_net = ContextBaseline(
            # NOTE: 64, not mlp_hidden -- BiasHeadMLP.feat_dim (the trunk
            # output ContextBaseline consumes via forward_features) is
            # hardcoded to 64 regardless of mlp_hidden, which is actually an
            # unused, API-compatibility-only constructor param (see
            # EdgeValueMLP.__init__'s docstring).
            trunk_dim=64, num_bias_types=num_bias_types,
            num_systems=len(system_to_idx),
            system_embed_dim=context_baseline_cfg.get('system_embed_dim', 8),
            hidden=context_baseline_cfg.get('hidden_dim', 32),
        ).to(device)
        baseline_optimizer = torch.optim.Adam(
            baseline_net.parameters(), lr=context_baseline_cfg.get('lr', 1e-3),
        )
        print(f"Context baseline enabled: {sum(p.numel() for p in baseline_net.parameters())} "
              f"params, {len(system_to_idx)} systems (incl. unknown fallback)")
    elif context_baseline_cfg.get('enabled', False):
        print("  Warning: training.context_baseline.enabled=True but "
              "bandit.algorithm == 'neurallinear_ts' — context baseline is a "
              "REINFORCE-only concept (NL+TS's closed-form update has no "
              "advantage/baseline step); ignoring.")

    # Replay-style AWR (opt-in, REINFORCE-mode only) -- see
    # _sample_replay_items / _compute_replay_awr_loss and train_epoch's
    # history_replay_cfg param.
    history_replay_cfg = config.get('replay', {})
    if history_replay_cfg.get('enabled', False) and algorithm == 'neurallinear_ts':
        print("  Warning: replay.enabled=True but bandit.algorithm == "
              "'neurallinear_ts' — history replay folds an extra REINFORCE-"
              "style loss into the gradient step NL+TS doesn't take; ignoring.")
        history_replay_cfg = {}

    # ------------------------------------------------------------------
    # Step 4: Load pretrained model / resume from checkpoint
    # ------------------------------------------------------------------
    output_config = config.get('output', {})
    checkpoint_dir = Path(output_config.get('base_dir', 'checkpoints'))

    # NeuralLinear + TS per-combo failure/defer bookkeeping (see
    # record_combo_failure / select_combos_by_uncertainty). Harmless to load
    # even in reinforce mode — it's simply never read or written then.
    bandit_state_path = checkpoint_dir / 'bandit_state.json'
    bandit_state = load_bandit_state(bandit_state_path)

    start_epoch = 0
    start_stage_idx = 0

    pretrain_path = config.get('pretrain', {}).get('model_path', None)
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            try:
                policy.load_state_dict(checkpoint['policy_state_dict'])
                rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
                # Optional: only present in checkpoints saved with a context
                # baseline enabled. Missing key is expected (and harmless) for
                # older checkpoints or when the feature is off this run.
                if baseline_net is not None and 'baseline_state_dict' in checkpoint:
                    baseline_net.load_state_dict(checkpoint['baseline_state_dict'])
                    if 'baseline_optimizer_state_dict' in checkpoint:
                        baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])
                    print("  Resumed context baseline state")
                elif baseline_net is not None:
                    print("  Note: context baseline enabled but no saved baseline state found "
                          "in this checkpoint (starting baseline from scratch)")
                # Handle missing 'epoch' key in old checkpoints
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                else:
                    # Infer epoch from checkpoint filename (e.g., checkpoint_009.pt -> epoch 9)
                    match = re.search(r'checkpoint_(\d+)', latest_checkpoint.name)
                    start_epoch = (int(match.group(1)) + 1) if match else 0
                    print(f"  Warning: 'epoch' key missing from checkpoint — inferred start_epoch={start_epoch} from filename")
                start_stage_idx = checkpoint.get('current_stage_idx', None)
            except (RuntimeError, KeyError) as _ckpt_err:
                print(f"  Warning: checkpoint loading failed — {_ckpt_err}")
                print(f"  Falling back to pretrained model (training restarts from epoch 0).")
                start_epoch = 0
                if pretrain_path and Path(pretrain_path).exists():
                    ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
                    if 'policy_state' in ckpt:
                        policy.load_state_dict(ckpt['policy_state'])
                        print("  Loaded policy weights from pretrain checkpoint")
        # No checkpoints found in checkpoint_dir — load pretrained model instead
        elif pretrain_path and Path(pretrain_path).exists():
            print(f"Loading pretrained policy from {pretrain_path}")
            checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
            if 'policy_state' in checkpoint:
                missing, unexpected = policy.load_state_dict(checkpoint['policy_state'], strict=False)
                if missing or unexpected:
                    print(f"  Warning: partial weight load — "
                          f"{len(missing)} missing, {len(unexpected)} unexpected keys")
                else:
                    print("  Loaded policy weights from pretrain checkpoint")
            # Verify weights were loaded
            weight_norm = sum(p.norm().item() ** 2 for p in policy.parameters()) ** 0.5
            print(f"  Policy weight norm: {weight_norm:.4f}")
            print(f"  Scale factors: {policy.scale_factors.tolist()}")
    elif pretrain_path and Path(pretrain_path).exists():
        print(f"Loading pretrained policy from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
        if 'policy_state' in checkpoint:
            missing, unexpected = policy.load_state_dict(checkpoint['policy_state'], strict=False)
            if missing or unexpected:
                print(f"  Warning: partial weight load — "
                      f"{len(missing)} missing, {len(unexpected)} unexpected keys")
            else:
                print("  Loaded policy weights from pretrain checkpoint")
        # Verify weights were loaded
        weight_norm = sum(p.norm().item() ** 2 for p in policy.parameters()) ** 0.5
        print(f"  Policy weight norm: {weight_norm:.4f}")
        print(f"  Scale factors: {policy.scale_factors.tolist()}")

    # ------------------------------------------------------------------
    # Step 5: Training loop
    # ------------------------------------------------------------------
    print("\n=== Training ===")
    curriculum_config = config.get('curriculum', {})
    curriculum_enabled = curriculum_config.get('enabled', False)

    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        num_epochs = sum(stage['epochs'] for stage in stages)
        print(f"Curriculum: {len(stages)} stages, {num_epochs} total epochs")
    else:
        num_epochs = train_config.get('num_epochs', 100)
        print(f"Training for {num_epochs} epochs")

    all_stats = []
    ema_baseline: dict = {}
    current_stage_idx = 0
    current_stage_epoch = 0
    active_archive_jobs = []

    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Lightweight per-epoch summary log (batch reward stats + epoch summary
    # only) — the main training_status_*.out log captures every print
    # statement and can grow to hundreds of MB over a long run, making it
    # hard to skim training progress. This file stays small and easy to tail.
    summary_log_path = checkpoint_dir / 'epoch_summary.log'
    print(f"Epoch summary log: {summary_log_path}")

    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        if start_epoch > 0:
            if start_stage_idx is not None:
                current_stage_idx = start_stage_idx
            else:
                epoch_counter = 0
                current_stage_idx = len(stages) - 1
                for _si, _st in enumerate(stages):
                    if epoch_counter + _st['epochs'] > start_epoch:
                        current_stage_idx = _si
                        break
                    epoch_counter += _st['epochs']
            epoch_offset = sum(stages[_si]['epochs'] for _si in range(current_stage_idx))
            current_stage_epoch = start_epoch - epoch_offset

        current_stage = stages[current_stage_idx]
        resumed = start_epoch > 0
        print(f"\n=== Stage {current_stage_idx+1}/{len(stages)}: {current_stage['name']}"
              f"{' (resumed)' if resumed else ''} ===")
        active_train_combos = apply_stage_combo_filter(
            train_combos, current_stage, curriculum_config,
            stage_idx=current_stage_idx, seed=seed,
            apply_transition_filter=(resumed and current_stage_idx > 0),
            combo_system_map=combo_system_map,
        )
        if resumed:
            _resume_lr = current_stage.get('policy_lr')
            if _resume_lr is not None:
                for _pg in rl_optimizer.param_groups:
                    _pg['lr'] = _resume_lr
                print(f"  Policy LR restored to {_resume_lr}")
        epoch_config = config.copy()
        if current_stage.get('reward_override'):
            epoch_reward = {**config.get('reward', {}), **current_stage['reward_override']}
            epoch_config = {**config, 'reward': epoch_reward}
    else:
        active_train_combos = train_combos
        epoch_config = config

    # Set up held-out validation set (before epochs start)
    # Provides ground truth for policy improvement, avoiding the misleading
    # batch-stat inference that allowed baseline chasing to hide degradation.
    # Persisted to disk so a resumed run (start_epoch > 0) restores the SAME
    # validation combos instead of silently losing validation forever (the
    # set was previously only ever built when start_epoch == 0).
    validation_combos_path = checkpoint_dir / 'validation_combos.json'
    validation_combos = []
    # Optional fixed held-out set from another run (e.g. to compare two
    # training settings on the same pairs). The seeded draw below picks by
    # index, so it changes whenever the combo pool does.
    _fixed_val_file = config.get('validation_combos_file')
    _val_per_system = config.get('validation_per_system')
    if start_epoch == 0 and num_epochs > 0 and _val_per_system and not _fixed_val_file:
        # A fixed number of held-out pairs per system, drawn from every system
        # in the run (all curriculum stages, so a system that only joins in a
        # later stage is held out from the start), with the first stage's
        # site/substituent limits.
        _stage0 = dict((curriculum_config.get('stages') or [{}])[0]) if curriculum_enabled else {}
        _stage0.pop('systems', None)
        _pool = (apply_stage_combo_filter(train_combos, _stage0, curriculum_config, stage_idx=0,
                                          seed=seed, apply_transition_filter=False)
                 if _stage0 else list(train_combos))
        _by_sys: Dict[str, List[str]] = {}
        for c in _pool:
            _by_sys.setdefault(combo_system_map.get(c, {}).get('name'), []).append(c)
        for _name in sorted(_by_sys, key=str):
            _cands = sorted(_by_sys[_name])
            _k = min(int(_val_per_system), max(len(_cands) - 1, 0))
            if _k > 0:
                _idx = np.random.RandomState(seed=42).choice(len(_cands), size=_k, replace=False)
                validation_combos.extend(_cands[i] for i in sorted(_idx))
        _val_set = set(validation_combos)
        active_train_combos = [c for c in active_train_combos if c not in _val_set]
        with open(validation_combos_path, 'w') as _vf:
            json.dump(validation_combos, _vf)
        print(f"Held-out validation set: {_val_per_system} per system, {len(validation_combos)} combos "
              f"over {len(_by_sys)} system(s) (excluded from training pool)")
    elif start_epoch == 0 and num_epochs > 0 and _fixed_val_file:
        with open(_fixed_val_file, 'r') as _vf:
            _fixed = json.load(_vf)
        _pool = set(active_train_combos)
        validation_combos = [c for c in _fixed if c in _pool]
        if len(validation_combos) < len(_fixed):
            print(f"  Warning: {len(_fixed) - len(validation_combos)} of {len(_fixed)} combos in "
                  f"{_fixed_val_file} are not in this run's pool; validating on the rest")
        n_from_file = len(validation_combos)
        # Systems the file doesn't cover get their own seeded held-out draw
        # (validation_per_new_system combos each), so a run that adds systems
        # keeps the file's set for comparison and still validates the new ones.
        _covered = {combo_system_map.get(c, {}).get('name') for c in validation_combos}
        _n_new = int(config.get('validation_per_new_system', 6))
        _by_sys: Dict[str, List[str]] = {}
        for c in active_train_combos:
            _name = combo_system_map.get(c, {}).get('name')
            if _name not in _covered and c not in set(validation_combos):
                _by_sys.setdefault(_name, []).append(c)
        for _name in sorted(_by_sys, key=str):
            _cands = sorted(_by_sys[_name])
            _k = min(_n_new, max(len(_cands) - 1, 0))
            if _k > 0:
                _idx = np.random.RandomState(seed=42).choice(len(_cands), size=_k, replace=False)
                validation_combos.extend(_cands[i] for i in sorted(_idx))
        _val_set = set(validation_combos)
        active_train_combos = [c for c in active_train_combos if c not in _val_set]
        with open(validation_combos_path, 'w') as _vf:
            json.dump(validation_combos, _vf)
        print(f"Held-out validation set from {_fixed_val_file}: {n_from_file} combos"
              + (f" + {len(validation_combos) - n_from_file} drawn for {len(_by_sys)} system(s) "
                 f"not in it" if len(validation_combos) > n_from_file else "")
              + " (excluded from training pool)")
    elif start_epoch == 0 and num_epochs > 0 and len(active_train_combos) > 20:
        val_fraction = config.get('validation_fraction', 0.1)
        # Cap validation size based on max_train_combos_per_stage (curriculum training budget)
        # rather than arbitrarily. This keeps validation at ~10% of per-epoch training batch.
        _progression = config.get('progression', {})
        _max_stage_combos = 1000  # default fallback
        if _progression and 'stages' in _progression and len(_progression['stages']) > 0:
            _max_stage_combos = _progression['stages'][0].get('max_train_combos_per_stage', 1000)
        _val_size_max = max(100, int(_max_stage_combos * val_fraction))  # Min 100, else 10% of stage budget
        val_size = min(_val_size_max, max(10, int(len(active_train_combos) * val_fraction)))
        rng_val = np.random.RandomState(seed=42)
        val_indices = rng_val.choice(
            len(active_train_combos),
            size=min(val_size, len(active_train_combos)),
            replace=False
        )
        validation_combos = [active_train_combos[i] for i in sorted(val_indices)]
        train_combos_filtered = [
            c for i, c in enumerate(active_train_combos) if i not in val_indices
        ]
        active_train_combos = train_combos_filtered
        try:
            with open(validation_combos_path, 'w') as _vf:
                json.dump(validation_combos, _vf)
        except Exception as _vf_err:
            print(f"  Warning: could not save validation combo set: {_vf_err}")
        print(f"\n{'='*70}")
        print(f"Held-Out Validation Set (Ground Truth for Policy Quality)")
        print(f"{'='*70}")
        print(f"Validation combos: {len(validation_combos)} (fixed, evaluated deterministically each epoch)")
        print(f"Training combos: {len(active_train_combos)} (updated per epoch)")
        print(f"Purpose: Detect true improvement vs. baseline chasing (training batch composition shifts each epoch)")
        print(f"{'='*70}\n")
    elif validation_combos_path.exists():
        try:
            with open(validation_combos_path, 'r') as _vf:
                validation_combos = json.load(_vf)
            _val_set = set(validation_combos)
            active_train_combos = [c for c in active_train_combos if c not in _val_set]
            print(f"Restored held-out validation set from {validation_combos_path}: "
                  f"{len(validation_combos)} combos (excluded from training pool)")
        except Exception as _vf_err:
            print(f"  Warning: could not restore validation combo set: {_vf_err}")

    _gate_state, _gate_score = None, None   # validation gate (see below)
    _gate_key = None                        # held-out set the gate's scores refer to
    # Best validated model (shared_update: regression with the gate): saved as
    # best_model.pt, and what the run ends on if the last validation is worse.
    _best_state, _best_score, _best_epoch, _last_val_score = None, None, None, None
    for epoch in range(start_epoch, num_epochs):
        if curriculum_enabled:
            stages = curriculum_config.get('stages', [])
            stage_epochs = stages[current_stage_idx]['epochs']
            if current_stage_epoch >= stage_epochs:
                progression_type = curriculum_config.get('progression', {}).get('type', 'epoch')
                can_advance = (progression_type == 'epoch')
                if not can_advance:
                    reward_threshold = curriculum_config.get('progression', {}).get('reward_threshold', 0.0)
                    recent_rewards = [s['avg_reward'] for s in all_stats[-5:] if 'avg_reward' in s]
                    avg_recent = np.mean(recent_rewards) if recent_rewards else -999.0
                    can_advance = avg_recent >= reward_threshold
                    if not can_advance:
                        print(f"\nStage {current_stage_idx+1} kept: avg reward "
                              f"{avg_recent:.4f} < threshold {reward_threshold}")

                if can_advance and current_stage_idx + 1 < len(stages):
                    current_stage_idx += 1
                    current_stage_epoch = 0
                    current_stage = stages[current_stage_idx]
                    print(f"\n{'='*60}")
                    print(f"=== Stage {current_stage_idx+1}/{len(stages)}: {current_stage['name']} ===")
                    print(f"{'='*60}")
                    active_train_combos = apply_stage_combo_filter(
                        train_combos, current_stage, curriculum_config,
                        stage_idx=current_stage_idx, seed=seed,
                        apply_transition_filter=True,
                        combo_system_map=combo_system_map,
                    )
                    # Keep the held-out set out of every stage's pool.
                    _val_set = set(validation_combos)
                    active_train_combos = [c for c in active_train_combos if c not in _val_set]
                    _new_lr = current_stage.get('policy_lr')
                    if _new_lr is not None:
                        for _pg in rl_optimizer.param_groups:
                            _pg['lr'] = _new_lr
                        rl_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            rl_optimizer, mode='max', factor=0.5, patience=3,
                            min_lr=1e-7, verbose=False,
                        )
                        print(f"  Policy LR updated to {_new_lr}, scheduler reset")
                    epoch_config = config.copy()
                    if current_stage.get('reward_override'):
                        epoch_reward = {**config.get('reward', {}), **current_stage['reward_override']}
                        epoch_config = {**config, 'reward': epoch_reward}

        stage_label = (
            f"Stage {current_stage_idx+1}/{len(stages)} "
            f"({stages[current_stage_idx]['name']}) "
            f"epoch {current_stage_epoch+1}/{stages[current_stage_idx]['epochs']}"
        ) if curriculum_enabled else f"epoch {epoch}"
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{num_epochs-1}  [{stage_label}]")
        print(f"{'='*80}")

        if curriculum_enabled:
            _cur_stage = stages[current_stage_idx]
        else:
            _cur_stage = {}
        _update_policy = _cur_stage.get('update_policy', True)
        _bc_mode = _cur_stage.get('bc_mode', False)
        _bc_source_epoch = _cur_stage.get('bc_source_epoch', None)
        _replay_from_epoch = _cur_stage.get('replay_from_epoch', None)
        _deterministic_actions = _cur_stage.get('deterministic_actions', False)
        _bc_auxiliary = _cur_stage.get('bc_auxiliary_weight', 0.0)

        # Re-sample combos each epoch so max_train_combos_per_stage draws a fresh
        # random subset rather than reusing the same one fixed at stage start.
        _max_epoch_combos = _cur_stage.get('max_train_combos',
                            curriculum_config.get('max_train_combos_per_stage'))

        # Uncertainty-driven combo selection. active_train_combos has already
        # been through the curriculum/transition filters above — that is the
        # "simulator-tractability gate" from the design note, applied
        # identically regardless of selection mode; this only changes how the
        # epoch's subset is picked from what survives that gate.
        #
        # Two scoring backends, chosen by `algorithm` (config knob is
        # `bandit.selection`, not tied to the training algorithm itself):
        #   - neurallinear_ts: Bayesian posterior predictive variance
        #     (bayesian_variance_score) — true epistemic uncertainty, but
        #     requires the Bayesian heads that mode maintains. Defaults on.
        #   - reinforce: the policy's own action std, exp(log_std)
        #     (action_std_score) — reuses the Gaussian the policy already
        #     samples from, so it works without any extra model state, but
        #     reflects learned exploration noise rather than epistemic
        #     uncertainty. Both selection strategies have merit (uncertainty
        #     favors combos still poorly understood; random avoids biasing
        #     the curriculum toward noisy/high-variance heads), so this
        #     defaults off for REINFORCE and is opt-in via
        #     `bandit.selection: uncertainty`.
        _default_selection = 'uncertainty' if algorithm == 'neurallinear_ts' else 'random'
        use_uncertainty_selection = (
            bandit_cfg.get('selection', _default_selection) == 'uncertainty'
        )
        _uncertainty_score_fn = (
            bayesian_variance_score if algorithm == 'neurallinear_ts' else action_std_score
        )
        # shared_update: regression re-seeds each head's mean but never adds
        # posterior observations, so n_obs stays at the pretrained value (0)
        # and this gate would hold selection at random for the whole run. The
        # pretraining-calibrated precision it keeps is the posterior that
        # selection scores against, so skip the gate there.
        _shared_regression = (bandit_cfg.get('update_target', 'awr') == 'hindsight'
                              and bandit_cfg.get('shared_update', 'posterior') == 'regression')
        if use_uncertainty_selection and algorithm == 'neurallinear_ts' and not _shared_regression:
            _min_obs = bandit_cfg.get('min_observations_before_confident', 3)
            _n_obs_per_head = [mlp.bayesian_head.n_obs.item() for mlp in policy.edge_mlp.mlps]
            if min(_n_obs_per_head) < _min_obs:
                print(f"  NeuralLinear+TS: posterior not yet confident "
                      f"(min n_obs={min(_n_obs_per_head):.0f} < {_min_obs}) — "
                      f"falling back to random combo selection for this epoch")
                use_uncertainty_selection = False

        if _max_epoch_combos is not None and len(active_train_combos) > _max_epoch_combos and use_uncertainty_selection:
            # Scoring a candidate costs the same as training on it (a full
            # Uni-Mol embedding build), so ranking the entire curriculum pool
            # (can be 100K+ combos) every epoch is intractable. Cap how many
            # candidates are actually built+scored, reseeded each epoch so a
            # fresh slice of the pool is considered over time.
            _candidate_pool_size = bandit_cfg.get('candidate_pool_size', 5 * _max_epoch_combos)
            _min_per_system = curriculum_config.get('min_combos_per_system', 5)
            _uncertainty_candidates = active_train_combos
            if len(_uncertainty_candidates) > _candidate_pool_size:
                # Stratified (not uniform) subsample: a uniform draw over the
                # whole pool is dominated by whichever systems happen to have
                # the largest manifest, starving small/underrepresented
                # systems out of the candidate pool before scoring even runs.
                _cand_system_pools: Dict[str, List[str]] = {}
                for _c in _uncertainty_candidates:
                    _cand_system_pools.setdefault(
                        combo_system_map.get(_c, {}).get('name', 'unknown'), []
                    ).append(_c)
                _uncertainty_candidates = sample_combos_stratified(
                    _cand_system_pools, n_total=_candidate_pool_size,
                    min_per_system=_min_per_system, seed=seed + epoch + 20_000,
                )
            # Most of a large curriculum pool has never been materialized on
            # disk (combo dirs are created lazily) — pre-scan metadata once
            # for this epoch's candidate slice so ensure_dir_fn below doesn't
            # fall back to a full linear JSON scan per missing combo.
            _uncertainty_dir_lookup = build_combo_dir_lookup(
                _uncertainty_candidates, combo_system_map)

            def _ensure_dir_for_scoring(combo_path, sys_cfg, cfg, verbose):
                return ensure_combo_dir_exists(
                    combo_path, sys_cfg, cfg, verbose=verbose,
                    _lookup=_uncertainty_dir_lookup)

            # Pairs' own online state (last run failed, exploration widened)
            # adds to the ranking so stuck pairs keep being revisited.
            def _pair_attention_fn(combo_dir, data):
                return pair_attention(
                    bandit_state, combo_system_map.get(combo_dir, {}).get('name', 'unknown'),
                    combo_node_labels(combo_dir), data.edge_index)
            epoch_combos = select_combos_by_uncertainty(
                _uncertainty_candidates, policy, combo_system_map, config,
                n_select=_max_epoch_combos, bandit_state=bandit_state,
                device=device, build_graph_fn=build_graph_and_data,
                ensure_dir_fn=_ensure_dir_for_scoring,
                min_per_system=_min_per_system,
                score_fn=_uncertainty_score_fn,
                attention_fn=_pair_attention_fn,
                attention_weight=float(bandit_cfg.get('selection_attention_weight', 0.0)),
            )
            _score_label = 'posterior variance' if algorithm == 'neurallinear_ts' else 'action std'
            print(f"  Epoch combo sample (uncertainty-driven, {_score_label}): "
                  f"{len(epoch_combos)}/{len(active_train_combos)} combos selected "
                  f"({len(_uncertainty_candidates)} candidates scored, "
                  f"min_per_system={_min_per_system})")
        elif _max_epoch_combos is not None and len(active_train_combos) > _max_epoch_combos:
            # Check if stratified sampling is enabled
            use_stratified = curriculum_config.get('stratified_sampling', True)

            if use_stratified:
                # Build system -> combos mapping for stratified sampling
                system_combo_pools = {}
                for combo_str in active_train_combos:
                    sys_cfg = combo_system_map.get(combo_str, {})
                    sys_name = sys_cfg.get('name', 'unknown')
                    if sys_name not in system_combo_pools:
                        system_combo_pools[sys_name] = []
                    system_combo_pools[sys_name].append(combo_str)
                
                # Use stratified sampling with minimum allocation per system
                min_per_sys = curriculum_config.get('min_combos_per_system', 5)
                epoch_combos = sample_combos_stratified(
                    system_combo_pools,
                    n_total=_max_epoch_combos,
                    min_per_system=min_per_sys,
                    seed=seed + epoch
                )
                print(f"  Epoch combo sample (stratified): {len(epoch_combos)}/{len(active_train_combos)} "
                      f"combos selected (min_per_system={min_per_sys}, seed={seed + epoch})")
                # Log allocation per system
                alloc_str = ', '.join([f"{sys}:{sum(1 for c in epoch_combos if combo_system_map.get(c, {}).get('name') == sys)}"
                                      for sys in sorted(system_combo_pools.keys())])
                print(f"  Allocation per system: {alloc_str}")
            else:
                # Original uniform random sampling
                _epoch_rng = np.random.RandomState(seed + epoch)
                epoch_combos = list(_epoch_rng.choice(
                    active_train_combos, size=_max_epoch_combos, replace=False))
                print(f"  Epoch combo sample (uniform): {_max_epoch_combos}/{len(active_train_combos)} "
                      f"combos selected (seed={seed + epoch})")
        else:
            epoch_combos = active_train_combos

        # Sample a reserve pool for on-the-fly replacement of permanently failed combos.
        _reserve_size = config.get('max_failed_combos_per_epoch', 50)
        _epoch_pool_set = set(epoch_combos)
        _reserve_candidates = [c for c in active_train_combos if c not in _epoch_pool_set]
        if _reserve_candidates and _reserve_size > 0:
            _res_rng = np.random.RandomState(seed + epoch + 10_000)
            _epoch_reserve = list(_res_rng.choice(
                _reserve_candidates,
                size=min(_reserve_size, len(_reserve_candidates)),
                replace=False))
        else:
            _epoch_reserve = []

        stats = train_epoch(
            policy, rl_optimizer,
            epoch_combos, combo_system_map,
            epoch, epoch_config, device,
            update_policy=_update_policy,
            bc_mode=_bc_mode,
            bc_source_epoch=_bc_source_epoch,
            replay_from_epoch=_replay_from_epoch,
            deterministic_actions=_deterministic_actions,
            bc_auxiliary_weight=_bc_auxiliary,
            ema_baseline=ema_baseline,
            reserve_pool=_epoch_reserve,
            verbose=verbose,
            summary_log_path=summary_log_path,
            bandit_state=bandit_state,
            baseline_net=baseline_net,
            baseline_optimizer=baseline_optimizer,
            system_to_idx=system_to_idx,
            history_replay_cfg=history_replay_cfg,
        )
        all_stats.append(stats)
        if algorithm == 'neurallinear_ts':
            save_bandit_state(bandit_state_path, bandit_state)

        # Held-out validation: deterministic evaluation on fixed set
        # This provides unbiased ground truth for whether policy is actually improving.
        # Buffered (not written immediately) for the same reason as
        # train_epoch's detail_lines -- see the epoch-summary-first write below.
        _val_detail_lines: List[str] = []
        # Validate every eval_freq epochs and always after the last epoch, so the
        # run's end point is measured (and the best model can be chosen).
        # Validate only the systems the current stage trains on (a stage with
        # `systems:` leaves the others for later).
        _stage_systems = (set(current_stage['systems'])
                          if curriculum_enabled and current_stage.get('systems') else None)
        _epoch_val = ([c for c in validation_combos
                       if combo_system_map.get(c, {}).get('name') in _stage_systems]
                      if _stage_systems else validation_combos)
        if _epoch_val and (epoch % config.get('eval_freq', 1) == 0
                           or epoch == num_epochs - 1):
            print(f"\n  [VALIDATION] Evaluating on {len(_epoch_val)} held-out combos (deterministic)...")
            val_stats = train_epoch(
                policy, rl_optimizer,
                _epoch_val, combo_system_map,
                epoch, epoch_config, device,
                update_policy=False,  # No gradient updates on validation
                bc_mode=False,
                deterministic_actions=True,  # Deterministic: no exploration noise
                ema_baseline=None,  # No EMA baseline tracking (batch stats only)
                reserve_pool=None,
                verbose=False,
                summary_log_path=None,
            )
            val_reward = val_stats.get('avg_reward', float('nan'))
            val_std = val_stats.get('reward_std', float('nan'))
            train_reward = stats.get('avg_reward', float('nan'))
            train_std = stats.get('reward_std', float('nan'))
            val_loss = val_stats.get('loss', float('nan'))
            train_loss = stats.get('loss', float('nan'))
            _val_line_1 = (
                f"  [VALIDATION] Train Reward: {train_reward:7.4f}±{train_std:6.4f}  |  "
                f"Val Reward: {val_reward:7.4f}±{val_std:6.4f}  (delta: {val_reward - train_reward:+.4f})"
            )
            _val_line_2 = f"  [VALIDATION] Train Loss: {train_loss:7.4f}  |  Val Loss: {val_loss:7.4f}"
            print(_val_line_1)
            print(_val_line_2)
            _val_detail_lines.append(_val_line_1)
            _val_detail_lines.append(_val_line_2)
            _val_pair_line = format_pair_success(val_stats.get('pair_success', {}), 'validation')
            if _val_pair_line:
                print(_val_pair_line)
                _val_detail_lines.append(_val_pair_line)
            # Store validation stats for analysis
            stats['val_reward'] = val_reward
            stats['val_reward_std'] = val_std
            stats['val_loss'] = val_loss
            stats['val_n_combos'] = len(_epoch_val)
            stats['val_pair_success'] = val_stats.get('pair_success', {})

            # Validation gate (bandit.shared_update: regression): the full
            # network moves each epoch, so keep it only while held-out pairs
            # hold up. A pass that scores worse than the last accepted one by
            # more than the margin restores that accepted model; per-pair
            # offsets (bandit_state) are left as they are.
            _reg_gate = (bandit_cfg.get('shared_update', 'posterior') == 'regression'
                         and (bandit_cfg.get('regression') or {}).get('validation_gate', True))
            if _reg_gate:
                _val_key = frozenset(_epoch_val)
                if _gate_key is not None and _val_key != _gate_key:
                    # e.g. a new stage adds a system: scores aren't comparable
                    print("  [VALIDATION GATE] held-out set changed (new stage): gate and best "
                          "model reset")
                    _gate_state = _gate_score = None
                    _best_state = _best_score = _best_epoch = None
                _gate_key = _val_key
                _score = int(val_stats.get('pair_success', {}).get('both', 0))
                _margin = int((bandit_cfg.get('regression') or {}).get('gate_margin', 2))
                if _gate_score is not None and _score < _gate_score - _margin:
                    policy.load_state_dict(_gate_state)
                    _gate_msg = (f"  [VALIDATION GATE] held-out success {_score} < accepted "
                                 f"{_gate_score} - {_margin}: restored the last accepted model")
                else:
                    _gate_state = copy.deepcopy(policy.state_dict())
                    _gate_score = _score if _gate_score is None else max(_gate_score, _score)
                    _gate_msg = (f"  [VALIDATION GATE] held-out success {_score}: model accepted "
                                 f"(reference {_gate_score})")
                _last_val_score = _score
                if _best_score is None or _score > _best_score:
                    _best_state = copy.deepcopy(policy.state_dict())
                    _best_score, _best_epoch = _score, epoch
                    torch.save({'policy_state_dict': _best_state, 'epoch': epoch,
                                'val_success': _score}, checkpoint_dir / 'best_model.pt')
                    _gate_msg += f"; new best (epoch {epoch}) saved to best_model.pt"
                print(_gate_msg)
                _val_detail_lines.append(_gate_msg)

        try:
            _summary_lines = [f"\nEpoch {epoch} Summary:"]
            print(_summary_lines[-1])
            _summary_lines.append(f"  Policy Loss: {stats['loss']:.4f}")
            print(_summary_lines[-1])
            _summary_lines.append(f"  Avg Reward: {stats['avg_reward']:.4f}±{stats.get('reward_std', 0.0):.4f}")
            print(_summary_lines[-1])
            # Add validation metrics if available
            if 'val_reward' in stats and not np.isnan(stats['val_reward']):
                _summary_lines.append(
                    f"  Val Reward: {stats['val_reward']:.4f}±{stats.get('val_reward_std', 0.0):.4f} "
                    f"({stats.get('val_n_combos', len(validation_combos))} combos)"
                )
                print(_summary_lines[-1])
            n_success_ep = stats.get('n_success', 0)
            _summary_lines.append(f"  Success batch: {n_success_ep}")
            print(_summary_lines[-1])
            _train_pair_line = format_pair_success(stats.get('pair_success', {}), 'train batch')
            if _train_pair_line:
                _summary_lines.append(_train_pair_line)
                print(_summary_lines[-1])
            if 'val_pair_success' in stats:
                _val_pair_line_2 = format_pair_success(stats['val_pair_success'], 'validation')
                if _val_pair_line_2:
                    _summary_lines.append(_val_pair_line_2)
                    print(_summary_lines[-1])
            _pf = stats.get('permanently_failed', [])
            if _pf:
                _summary_lines.append(f"  Permanently failed ({len(_pf)} combo(s)):")
                print(_summary_lines[-1])
                for _fc in _pf:
                    _summary_lines.append(f"    {_fc}")
                    print(_summary_lines[-1])
            _summary_lines.append(
                f"  Policy updated: "
                f"{'Yes (BC)' if stats.get('bc_mode') else ('Yes' if stats.get('update_policy', True) else 'No')}"
            )
            print(_summary_lines[-1])
            sys.stdout.flush()  # Ensure all summary lines are flushed before appending to file

            try:
                # Summary header first, then every detail line generated while
                # producing this epoch (batch reward stats, failure gradient,
                # history replay, validation) -- so epoch_summary.log reads
                # top-down per epoch (header, then supporting detail) instead
                # of details trickling in ahead of the summary that explains
                # them. Console output above is unaffected -- it stays in the
                # original chronological order these were computed in.
                append_summary(summary_log_path, '\n'.join(_summary_lines))
                _epoch_detail_lines = stats.get('detail_lines', []) + _val_detail_lines
                if _epoch_detail_lines:
                    append_summary(summary_log_path, '\n'.join(_epoch_detail_lines))
            except Exception as _sum_err:
                print(f"  Warning: Failed to write epoch summary to file: {_sum_err}")
                sys.stdout.flush()
        except Exception as _summary_err:
            print(f"\nERROR: Failed to print epoch summary: {_summary_err}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

        # rl_optimizer only steps in REINFORCE and BC; NL+TS updates in closed
        # form or, with regression, with its own optimizer at regression.lr.
        if (not stats.get('bc_mode') and stats.get('update_policy', True)
                and algorithm != 'neurallinear_ts'):
            prev_lrs = [pg['lr'] for pg in rl_optimizer.param_groups]
            rl_scheduler.step(n_success_ep)
            new_lrs = [pg['lr'] for pg in rl_optimizer.param_groups]
            if new_lrs != prev_lrs:
                print(f"  ReduceLROnPlateau fired: LR {prev_lrs[0]:.2e} → {new_lrs[0]:.2e}")

        if curriculum_enabled:
            current_stage_epoch += 1

        # Only present when a context baseline is enabled; loaded back
        # defensively above (missing key is fine for older/non-baseline runs).
        _baseline_ckpt_fields = {}
        if baseline_net is not None:
            _baseline_ckpt_fields = {
                'baseline_state_dict': baseline_net.state_dict(),
                'baseline_optimizer_state_dict': baseline_optimizer.state_dict(),
            }

        if output_config.get('save_checkpoints', True):
            if epoch % output_config.get('checkpoint_freq', 5) == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_{epoch:03d}.pt'
                torch.save({
                    'epoch': epoch,
                    'policy_state_dict': policy.state_dict(),
                    'rl_optimizer_state_dict': rl_optimizer.state_dict(),
                    'current_stage_idx': current_stage_idx,
                    'stats': stats,
                    **_baseline_ckpt_fields,
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n=== Training Complete ===")

    # End on the best validated model when the last validation scored lower.
    _final_from = 'last epoch'
    if _best_state is not None and _last_val_score is not None and _last_val_score < _best_score:
        policy.load_state_dict(_best_state)
        _final_from = f'best validated model (epoch {_best_epoch}, held-out success {_best_score})'
        print(f"Restored the best validated model: epoch {_best_epoch}, held-out success "
              f"{_best_score} (last validation {_last_val_score})")

    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'final_from': _final_from,
        'policy_state_dict': policy.state_dict(),
        'rl_optimizer_state_dict': rl_optimizer.state_dict(),
        'all_stats': all_stats,
        **_baseline_ckpt_fields,
    }, final_path)
    print(f"Saved final model: {final_path}")

    # Restore warning handler and print summary
    warnings.showwarning = _orig_showwarning
    if _warning_counts:
        print(f"\n{'='*60}")
        print(f"Warning Summary ({sum(_warning_counts.values())} total occurrences, "
              f"{len(_warning_counts)} unique types)")
        print(f"{'='*60}")
        for (cat, template), count in sorted(_warning_counts.items(), key=lambda kv: -kv[1]):
            example_text, ex_file, ex_line = _warning_example[(cat, template)]
            short_file = Path(ex_file).name if ex_file else '?'
            print(f"  [{count:5d}x] {cat} ({short_file}:{ex_line})")
            print(f"           {example_text[:200]}")
        print(f"{'='*60}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description='Run online CB-policy training from a YAML config',
        epilog='Example: python -m mllf.cli.train_workflow examples/workflow_14benz.yaml',
    )
    p.add_argument('config', help='Path to workflow YAML config file')
    args = p.parse_args()
    run_training(args.config)


if __name__ == '__main__':
    main()
