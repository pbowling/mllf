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
from typing import Dict, List, Optional, Tuple
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

from mllf.cb import epoch_log, system_stages
from mllf.cb.cost_ledger import LEDGER_FILE, load_ledger, record_runs, summarize_ledger
from mllf.cb.curriculum import (
    apply_stage_combo_filter,
    build_system_index,
    filter_combos_by_curriculum,
    sample_combos_stratified,
)
from mllf.cb.export_biases import export_final_biases, label_final_biases
from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.hindsight import combo_node_labels, merge_revisits, pair_attention, select_revisits
from mllf.cb.policy import ContextBaseline, UnimolPolicy
from mllf.cb.reinforce_trainer import append_summary, format_pair_success, train_epoch
from mllf.cb.workflow_utils import (
    action_std_score,
    bayesian_variance_score,
    load_bandit_state,
    save_bandit_state,
    select_combos_by_uncertainty,
)
from mllf.file_handling.combo_sets import build_combo_sets, reversed_pair
from mllf.file_handling.generate_combinations import find_site_sub_files
from mllf.file_handling.manifests import (
    build_combo_dir_lookup,
    build_system_manifest,
    count_combinations_fast,
    create_and_load_system_manifest,
    ensure_combo_dir_exists,
)


def _policy_weights(checkpoint: Dict):
    """Policy weights from a pretraining checkpoint ('policy_state') or an
    online-training one ('policy_state_dict': checkpoint_*.pt, final_model.pt,
    best_model.pt); None if neither."""
    return checkpoint.get('policy_state') or checkpoint.get('policy_state_dict')


def run_training(config_path: str) -> None:
    """Run the full online CB-policy training workflow from a YAML config.

    This is everything the training driver does end to end: build/load
    per-system combo manifests, initialize the ``UnimolPolicy`` (optionally
    warm-started from a pretrain checkpoint or resumed from the latest
    ``checkpoint_*.pt``), run the (optionally curriculum-staged) epoch loop
    via ``mllf.cb.reinforce_trainer.train_epoch``, and checkpoint along the
    way.

    ``mode: deploy`` tests a trained model on systems it has not seen: it
    loads ``deploy.model_path``, runs each system through the curriculum
    stages on its own (``mllf.cb.system_stages``) with per-pair offsets
    learning, keeps the shared model fixed unless
    ``deploy.update_shared_model`` is true, holds out no validation set, and
    writes only to its own ``output.base_dir`` (never the model's directory).

    Args:
        config_path: Path to a workflow YAML config (see docs/workflow.rst).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")

    mode = config.get('mode', 'train')
    if mode not in ('train', 'deploy'):
        raise ValueError(f"mode must be 'train' or 'deploy', got {mode!r}")
    deploy = mode == 'deploy'
    deploy_cfg = config.get('deploy') or {}
    if deploy:
        _bc = config.get('bandit') or {}
        if not (config.get('curriculum') or {}).get('enabled'):
            raise ValueError("mode: deploy needs curriculum stages (curriculum.enabled: true)")
        if (_bc.get('algorithm') != 'neurallinear_ts' or _bc.get('update_target') != 'hindsight'
                or not _bc.get('pair_offsets', True)):
            raise ValueError("mode: deploy needs bandit.algorithm: neurallinear_ts, "
                             "update_target: hindsight and pair_offsets (per-pair learning)")
        _model = deploy_cfg.get('model_path')
        if not _model or not Path(_model).exists():
            raise FileNotFoundError(f"deploy.model_path not found: {_model}")
        _out = Path((config.get('output') or {}).get('base_dir', 'checkpoints')).resolve()
        if Path(_model).resolve().parent == _out:
            raise ValueError("deploy output.base_dir is the trained model's directory; "
                             "use a separate directory so the trained model is never overwritten")
        # The trained model is the starting point (loaded like a pretrain checkpoint).
        config.setdefault('pretrain', {})['model_path'] = _model
        print(f"Mode: deploy (model {_model}; shared model "
              f"{'updated' if deploy_cfg.get('update_shared_model') else 'fixed'}; "
              f"per-pair offsets learn)")

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

    # Explicit combo sets (systems' `combo_sets:`, e.g. ALF's substituent
    # groups and their pairs; mllf.file_handling.combo_sets), selected by a
    # stage's `combo_set:`. Not in the manifests; created on demand.
    combo_sets_by_system: Dict[str, Dict[str, List[str]]] = {}
    for sys_entry in systems_cfg:
        if not sys_entry.get('combo_sets'):
            continue
        _sets, _warns = build_combo_sets(sys_entry, config.get('alf_root'))
        combo_sets_by_system[sys_entry.get('name')] = _sets
        for _paths in _sets.values():
            for _p in _paths:
                combo_system_map[_p] = sys_entry
        print(f"  [{sys_entry.get('name')}] combo sets: "
              + ', '.join(f"{k} {len(v)}" for k, v in _sets.items()))
        for _w in _warns:
            print(f"    warning: {_w}")

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
                    if _policy_weights(ckpt) is not None:
                        policy.load_state_dict(_policy_weights(ckpt))
                        print("  Loaded policy weights from pretrain checkpoint")
        # No checkpoints found in checkpoint_dir — load pretrained model instead
        elif pretrain_path and Path(pretrain_path).exists():
            print(f"Loading pretrained policy from {pretrain_path}")
            checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
            if _policy_weights(checkpoint) is not None:
                missing, unexpected = policy.load_state_dict(_policy_weights(checkpoint), strict=False)
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
        if _policy_weights(checkpoint) is not None:
            missing, unexpected = policy.load_state_dict(_policy_weights(checkpoint), strict=False)
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
        # A combo_set stage (per-system mode only) has no manifest filter; the
        # per-system pools are rebuilt every epoch.
        active_train_combos = list(train_combos) if current_stage.get('combo_set') else apply_stage_combo_filter(
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

    # Per-system stages (curriculum.per_system; always in mode: deploy): each
    # system moves through the stages on its own when its stage's `advance`
    # criterion holds (mllf.cb.system_stages). Stage keys that change the
    # whole epoch's training settings cannot differ between systems in one
    # batch, so they are ignored here.
    per_system = curriculum_enabled and (deploy or bool(curriculum_config.get('per_system', False)))
    if curriculum_enabled and not per_system and any(st.get('combo_set') for st in stages):
        raise ValueError("a stage's combo_set needs curriculum.per_system: true")
    stage_state: Dict[str, Dict] = {}
    stage_state_path = checkpoint_dir / system_stages.STATE_FILE
    combos_by_system: Dict[str, List[str]] = {}
    _ps_pool_cache: Dict[tuple, List[str]] = {}
    if per_system:
        epoch_config = config
        for c in train_combos:
            combos_by_system.setdefault(combo_system_map.get(c, {}).get('name'), []).append(c)
        _ignored = sorted({k for st in stages for k in st if k in (
            'update_policy', 'reward_override', 'policy_lr', 'bc_mode', 'bc_source_epoch',
            'replay_from_epoch', 'deterministic_actions', 'bc_auxiliary_weight', 'max_train_combos')})
        if _ignored:
            print(f"  Warning: per-system stages ignore stage keys {_ignored} "
                  f"(they would apply to every system in the epoch)")
        stage_state = (system_stages.load_state(stage_state_path) if start_epoch > 0 else None) or {}
        _fresh = system_stages.init_state([s.get('name') for s in systems_cfg], stages)
        for _name, _st in _fresh.items():
            stage_state.setdefault(_name, _st)
        system_stages.save_state(stage_state_path, stage_state)
        print("Per-system stages:")
        for _name, _st in stage_state.items():
            print(f"  {_name}: " + ('done' if _st['done'] else
                                   f"{stages[_st['stage']]['name']} (epoch {_st['stage_epochs']} "
                                   f"of {stages[_st['stage']]['epochs']})"))

    def _system_stage_pool(sys_name: str) -> List[str]:
        """The system's combos for its current stage (stage filters, minus `systems`)."""
        st = stage_state.get(sys_name)
        if not st or st['done'] or st['stage'] is None:
            return []
        key = (sys_name, st['stage'])
        if key not in _ps_pool_cache:
            _stage = dict(stages[st['stage']])
            _stage.pop('systems', None)
            print(f"  [{sys_name}] stage {_stage['name']}:")
            if _stage.get('combo_set'):
                # An explicit set (e.g. ALF's groups); the stage's site/sub
                # filters, if given, still apply to it.
                _cands = combo_sets_by_system.get(sys_name, {}).get(_stage['combo_set'], [])
                _ps_pool_cache[key] = (apply_stage_combo_filter(
                    _cands, _stage, curriculum_config, stage_idx=st['stage'], seed=seed,
                    apply_transition_filter=False) if 'min_sites' in _stage else list(_cands))
                print(f"  combo set {_stage['combo_set']}: {len(_ps_pool_cache[key])} combos")
            else:
                _ps_pool_cache[key] = apply_stage_combo_filter(
                    combos_by_system.get(sys_name, []), _stage, curriculum_config,
                    stage_idx=st['stage'], seed=seed, apply_transition_filter=st['stage'] > 0)
        return _ps_pool_cache[key]

    def _system_group_pool(sys_name: str) -> List[str]:
        """Groups (the current stage's ``group_set``) that join the pool once ready."""
        st = stage_state.get(sys_name)
        if not st or st['done'] or st['stage'] is None:
            return []
        gset = stages[st['stage']].get('group_set')
        return list(combo_sets_by_system.get(sys_name, {}).get(gset, [])) if gset else []

    def _stage_names() -> Dict[str, str]:
        """System -> the stage it is training in (for per-pair stage counts and the ledger)."""
        if per_system:
            return {s: stages[st['stage']]['name'] for s, st in stage_state.items()
                    if not st['done'] and st['stage'] is not None}
        if curriculum_enabled:
            return {s.get('name'): stages[current_stage_idx]['name'] for s in systems_cfg}
        return {}

    def _epoch_log_block(epoch, epoch_combos, stats, val_ps, gate, passed, reopened, rows,
                         changes, cost) -> str:
        """The epoch's epoch_summary.log block (mllf.cb.epoch_log)."""
        runs = {'groups': 0, 'reference': 0, 'reversed': 0, 'other': 0}
        for c in epoch_combos:
            sn = combo_system_map.get(c, {}).get('name') or Path(c).parent.name
            if Path(c).name in {Path(g).name for g in _system_group_pool(sn)}:
                runs['groups'] += 1
                continue
            want = set(_ref_keys.get(sn, ()))
            for ks in ((stage_state.get(sn) or {}).get('group_hold') or {}).values():
                want |= set(ks)
            keys = system_stages.combo_pair_keys(sn, c)
            if keys and all(k in want for k in keys):
                runs['reference'] += 1
                runs['reversed'] += bool(re.match(r'pair\d+r_', Path(c).name))
            else:
                runs['other'] += 1
        for r in rows:
            r['ns'] = ((cost or {}).get(r['system']) or {}).get('train', {}).get('ns', 0.0)
        return epoch_log.format_epoch_block(
            epoch, runs, failed=stats.get('permanently_failed', []),
            pair_short=stats.get('pair_success_short'), pair_full=stats.get('pair_success_full'),
            groups_run=runs['groups'], groups_passed=passed, groups_reopened=reopened,
            validation=val_ps, gate=gate, regression=stats.get('regression'),
            hindsight=stats.get('hindsight'), systems=rows, stage_changes=changes,
            ns_train=sum(c['train']['ns'] for c in (cost or {}).values()) if cost else None,
            ns_validation=sum(c['validation']['ns'] for c in (cost or {}).values()) if cost else None)

    def _export_final(sys_name: str) -> None:
        """Production biases for a system that finished its last stage."""
        if not curriculum_config.get('export_final_biases', True):
            return
        sys_cfg = next((s for s in systems_cfg if s.get('name') == sys_name), None)
        if sys_cfg is None:
            return
        # export_combo_set: one export per combo of that set (e.g. each ALF
        # group, as ALF ran production); else the whole system (or
        # export_combo_dir).
        _set = sys_cfg.get('export_combo_set')
        _targets = ([(Path(c).name, c) for c in combo_sets_by_system.get(sys_name, {}).get(_set, [])]
                    if _set else [(None, sys_cfg.get('export_combo_dir'))])
        for _sub, _cdir in _targets:
            try:
                if _sub is not None:
                    _cdir = ensure_combo_dir_exists(Path(_cdir), sys_cfg, config)
                _out = checkpoint_dir / 'final_biases' / sys_name
                path = export_final_biases(
                    policy, sys_cfg, config, bandit_state, _out / _sub if _sub else _out,
                    device=device, combo_dir=_cdir)
                print(f"  [EXPORT] {sys_name}: final biases written to {path}")
            except Exception as _exp_err:
                print(f"  [EXPORT] {sys_name}{'/' + _sub if _sub else ''}: export failed: {_exp_err}")
        # Confidence labels: CONFIDENCE.txt per group; groups that never passed
        # are renamed <group>_unconverged.
        try:
            _n = label_final_biases(checkpoint_dir, config, [sys_name])
            if _n:
                print(f"  [EXPORT] {sys_name}: confidence labels for {_n} group(s)")
        except Exception as _lab_err:
            print(f"  [EXPORT] {sys_name}: labelling failed: {_lab_err}")

    # Shared-model updates: always in training; in deploy only on request.
    update_shared = (not deploy) or bool(deploy_cfg.get('update_shared_model', False))

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
    # mode: deploy holds out nothing: every pair of a test system is part of the test.
    if deploy:
        print("Held-out validation: none (mode: deploy)")
    elif _val_per_system is not None and int(_val_per_system) == 0 and not _fixed_val_file:
        # validation_per_system: 0 -- hold nothing out (e.g. a small test run).
        print("Held-out validation: none (validation_per_system: 0)")
    elif start_epoch == 0 and num_epochs > 0 and _val_per_system and not _fixed_val_file:
        # A fixed number of held-out pairs per system, drawn from every system
        # in the run (all curriculum stages, so a system that only joins in a
        # later stage is held out from the start), with the first stage's
        # site/substituent limits.
        _stage0 = dict((curriculum_config.get('stages') or [{}])[0]) if curriculum_enabled else {}
        _stage0.pop('systems', None)
        if _stage0.get('combo_set'):
            # Training runs explicit sets (e.g. ALF's groups and their pairs):
            # hold out manifest pairs in no combo of any of the system's sets,
            # so validation measures the shared model on pairs training never
            # runs (in any stage).
            _pool = []
            _used: Dict[str, set] = {
                _name: {k for _paths in _sets.values() for p in _paths
                        for k in system_stages.combo_pair_keys(_name, p)}
                for _name, _sets in combo_sets_by_system.items()}
            for c in filter_combos_by_curriculum(train_combos, 1, 1, 2, 2):
                _name = combo_system_map.get(c, {}).get('name')
                if _name in _used and not set(system_stages.combo_pair_keys(_name, c)) & _used[_name]:
                    _pool.append(c)
        else:
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
        _active_systems: List[str] = []
        if per_system:
            # A stage with no combos for a system (filters leave none) is skipped.
            _val_set_ps = set(validation_combos)
            for _sn in sorted(stage_state):
                _st = stage_state[_sn]
                while not _st['done'] and not [c for c in _system_stage_pool(_sn) if c not in _val_set_ps]:
                    _skipped = stages[_st['stage']]['name']
                    _nxt = system_stages.advance(_st, stages, _sn, epoch, 'no_combos')
                    print(f"  [STAGE] {_sn}: no combos for stage {_skipped}; skipped")
                    if _nxt is None:
                        _export_final(_sn)
            system_stages.save_state(stage_state_path, stage_state)
            _active_systems = sorted(s for s, st in stage_state.items() if not st['done'])
            if not _active_systems:
                print(f"\nAll systems finished their last stage (epoch {epoch}); stopping.")
                break
            active_train_combos = [c for s in _active_systems for c in _system_stage_pool(s)
                                   if c not in _val_set_ps]
            # A pairs_resolved stage stops running pairs that are resolved,
            # and, with retire_after_epochs, pairs run in that many of the
            # stage's epochs without resolving (retired).
            _n_before = len(active_train_combos)
            _keep = []
            for c in active_train_combos:
                _sn = combo_system_map.get(c, {}).get('name')
                _stg = stages[stage_state[_sn]['stage']]
                _adv = _stg.get('advance') or {}
                if _adv.get('type') in ('pairs_resolved', 'groups_done') and system_stages.combo_resolved(
                        bandit_state, _sn, c, _stg['name'], _adv, bandit_cfg.get('offset_prior_sd'),
                        retire_after=_stg.get('retire_after_epochs')):
                    continue
                _keep.append(c)
            active_train_combos = _keep
            if len(_keep) < _n_before:
                print(f"  Resolved or retired pairs left the pool: {_n_before - len(_keep)} combo(s)")
            # Reference pairs per system: the stage's reference_set (e.g. each
            # ALF group's reference vs the others) when the pool holds more
            # pairs than those, else the pool itself.
            _ref_keys: Dict[str, set] = {}
            for _sn in _active_systems:
                _stg = stages[stage_state[_sn]['stage']]
                _rset = _stg.get('reference_set')
                _src = (combo_sets_by_system.get(_sn, {}).get(_rset, []) if _rset
                        else _system_stage_pool(_sn))
                _ref_keys[_sn] = {k for c in _src for k in system_stages.combo_pair_keys(_sn, c)}
            # A stage with a group_set runs each group once group_ready_fraction
            # of its reference pairs are resolved (retired ones left out),
            # until the group passes the group test. A group sent back to the
            # pair phase (group_max_failures) waits for its reopened pairs.
            _ready_groups: List[str] = []
            for _sn in _active_systems:
                _stg = stages[stage_state[_sn]['stage']]
                if not _stg.get('group_set'):
                    continue
                _passed = stage_state[_sn].get('groups_done') or {}
                _holds = stage_state[_sn].get('group_hold') or {}
                for g in _system_group_pool(_sn):
                    if Path(g).name not in _passed and system_stages.group_ready(
                            bandit_state, _sn, g, _ref_keys[_sn], _stg['name'], _stg.get('advance') or {},
                            bandit_cfg.get('offset_prior_sd'),
                            fraction=float(_stg.get('group_ready_fraction', 1.0)),
                            retire_after=_stg.get('retire_after_epochs'),
                            held=_holds.get(Path(g).name)):
                        _ready_groups.append(g)
                        _holds.pop(Path(g).name, None)
            if _ready_groups:
                active_train_combos = active_train_combos + _ready_groups
                print(f"  Ready groups in the pool: {len(_ready_groups)}")
        elif curriculum_enabled:
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

        if per_system:
            _counts = collections.Counter(stages[stage_state[s]['stage']]['name'] for s in _active_systems)
            stage_label = ("per-system stages: "
                           + ', '.join(f"{n} {c}" for n, c in _counts.items())
                           + f"; done {len(stage_state) - len(_active_systems)}")
        elif curriculum_enabled:
            stage_label = (
                f"Stage {current_stage_idx+1}/{len(stages)} "
                f"({stages[current_stage_idx]['name']}) "
                f"epoch {current_stage_epoch+1}/{stages[current_stage_idx]['epochs']}"
            )
        else:
            stage_label = f"epoch {epoch}"
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{num_epochs-1}  [{stage_label}]")
        print(f"{'='*80}")

        if curriculum_enabled and not per_system:
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
        # curriculum.combos_per_system: reference_pairs -- each system's share
        # of the epoch is proportional to its number of reference pairs (at
        # least min_combos_per_system), so a system with 50+ substituents can
        # cover its reference pairs as a 4-substituent one does. The epoch
        # budget grows if the floors push the shares past it.
        _system_quota: Optional[Dict[str, int]] = None
        if (per_system and _max_epoch_combos is not None
                and curriculum_config.get('combos_per_system') == 'reference_pairs'):
            _nref = {s: len(_ref_keys.get(s, ())) for s in _active_systems}
            _tot = sum(_nref.values())
            if _tot:
                _qmin = int(curriculum_config.get('min_combos_per_system', 5))
                _system_quota = {s: max(_qmin, int(round(_max_epoch_combos * n / _tot)))
                                 for s, n in _nref.items()}
                _max_epoch_combos = max(_max_epoch_combos, sum(_system_quota.values()))
                print("  Per-system shares (reference pairs): " + ', '.join(
                    f"{s} {q}" for s, q in sorted(_system_quota.items())))

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
                _sn = combo_system_map.get(combo_dir, {}).get('name', 'unknown')
                att = pair_attention(bandit_state, _sn, combo_node_labels(combo_dir), data.edge_index)
                # A stage's reference_pair_bonus ranks reference pairs ahead of
                # other pairs of similar uncertainty.
                if per_system and _sn in stage_state and not stage_state[_sn]['done']:
                    _bonus = float(stages[stage_state[_sn]['stage']].get('reference_pair_bonus', 0.0))
                    _keys = system_stages.combo_pair_keys(_sn, combo_dir)
                    if _bonus and _keys and all(k in _ref_keys.get(_sn, ()) for k in _keys):
                        att += _bonus
                return att
            epoch_combos = select_combos_by_uncertainty(
                _uncertainty_candidates, policy, combo_system_map, config,
                n_select=_max_epoch_combos, bandit_state=bandit_state,
                device=device, build_graph_fn=build_graph_and_data,
                ensure_dir_fn=_ensure_dir_for_scoring,
                min_per_system=_min_per_system,
                score_fn=_uncertainty_score_fn,
                attention_fn=_pair_attention_fn,
                attention_weight=float(bandit_cfg.get('selection_attention_weight', 0.0)),
                system_quota=_system_quota,
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

        # Revisits (bandit.revisit): rerun pairs already in progress until their
        # exploration is satisfactory, so per-pair corrections (b bracket, c
        # steps) get the consecutive visits they need instead of waiting to be
        # drawn again from a large pool. They take up to `fraction` of each
        # system's minimum allocation; the selection above fills the rest.
        _revisit_cfg = bandit_cfg.get('revisit') or {}
        _revisit_detail_lines: List[str] = []
        if (_revisit_cfg.get('enabled') and _update_policy and algorithm == 'neurallinear_ts'
                and _max_epoch_combos is not None):
            _rv_min = int(curriculum_config.get('min_combos_per_system', 5))
            _rv_frac = float(_revisit_cfg.get('fraction', 0.5))
            _rv_sys_cfg = {s.get('name'): s for s in systems_cfg}
            _rv_systems = {combo_system_map.get(c, {}).get('name') for c in active_train_combos}
            _rv_active = {(combo_system_map.get(c, {}).get('name'), Path(c).name)
                          for c in active_train_combos}
            _rv_val = {(combo_system_map.get(c, {}).get('name'), Path(c).name)
                       for c in validation_combos}

            def _rv_system(c):
                return combo_system_map.get(c, {}).get('name') or Path(c).parent.name

            def _rv_allowed(c):
                k = (_rv_system(c), Path(c).name)
                return (k in _rv_active and k not in _rv_val and Path(c).is_dir()
                        and not bandit_state.get(c, {}).get('deferred', False))

            _rv_quota = {s: int(round(_rv_frac * _rv_min)) for s in _rv_systems if s}
            _revisits, _rv_counts = select_revisits(
                bandit_state, _rv_quota, _rv_allowed,
                successes_needed=int(_revisit_cfg.get('successes', 2)),
                max_visits=int(_revisit_cfg.get('max_visits', 8)))
            for _c in _revisits:   # out_dir paths (also after a restart)
                if _c not in combo_system_map:
                    _s = Path(_c).parent.name
                    if _s in _rv_sys_cfg:
                        combo_system_map[_c] = _rv_sys_cfg[_s]
            epoch_combos = merge_revisits(_revisits, epoch_combos, _rv_system,
                                          n_total=_max_epoch_combos, min_per_system=_rv_min)
            _rv_msg = (f"  Revisits: {len(_revisits)} in-progress pair(s) rerun "
                       f"(pool: {_rv_counts['active']} active, {_rv_counts['satisfied']} satisfied, "
                       f"{_rv_counts['exhausted']} exhausted); {len(epoch_combos)} combos this epoch")
            print(_rv_msg)
            _revisit_detail_lines = [_rv_msg]

        # Per-system stages: pairs awaiting a full-length confirming run go
        # first (all of them), then pairs not yet run in this stage (up to
        # min_combos_per_system per system), then the selection above. In
        # the first run A, selection kept rerunning the same pairs while
        # most reference pairs were never run.
        if per_system and _max_epoch_combos is not None:
            _pmin = int(curriculum_config.get('min_combos_per_system', 5))
            _priority, _n_conf, _n_new = list(_ready_groups), 0, 0
            _n_ref, _n_partner = 0, 0
            _rank = {(combo_system_map.get(c, {}).get('name'), Path(c).name): n
                     for n, c in enumerate(epoch_combos)}
            _store_p = (bandit_state or {}).get('_pair_offsets', {})
            for _sn in _active_systems:
                _stg_p = stages[stage_state[_sn]['stage']]
                _stg_name = _stg_p['name']
                _short_stage = bool(_stg_p.get('run_steps'))
                _first_new = bool(_stg_p.get('prioritize_unvisited', True))
                _strict = _stg_p.get('reference_priority') == 'strict'
                # reference_priority: strict -- the system's share goes to its
                # unresolved reference pairs (and pairs a failed group sent
                # back) before any other pair: not yet run first, then in
                # selection rank. A pair whose run got stuck or stranded
                # (run_both) also runs reversed, lambda starting on its
                # other substituent; the partner takes a slot of the share.
                _want = set(_ref_keys.get(_sn, ()))
                for _ks in (stage_state[_sn].get('group_hold') or {}).values():
                    _want |= set(_ks)
                _refs = []
                _new = []
                for c in active_train_combos:
                    if combo_system_map.get(c, {}).get('name') != _sn or c in _ready_groups:
                        continue
                    _await, _unvisited = system_stages.combo_pair_flags(bandit_state, _sn, c, _stg_name)
                    if _strict:
                        _keys = system_stages.combo_pair_keys(_sn, c)
                        if _keys and all(k in _want for k in _keys):
                            _refs.append((0 if _unvisited else 1,
                                          _rank.get((_sn, Path(c).name), len(_rank)), c, _keys))
                            continue
                    if _await and _short_stage:     # confirmations exist only with short runs
                        _priority.append(c)
                        _n_conf += 1
                    elif _unvisited and _first_new:
                        _new.append(c)
                if _strict:
                    _share = (_system_quota or {}).get(_sn, _pmin)
                    _taken = 0
                    for _, _, c, _keys in sorted(_refs, key=lambda t: t[:2]):
                        if _taken >= _share:
                            break
                        _priority.append(c)
                        _taken += 1
                        _n_ref += 1
                        if (_taken < _share and len(_keys) == 1
                                and (_store_p.get(_keys[0]) or {}).get('run_both')):
                            _rev = reversed_pair(c)
                            if _rev:
                                combo_system_map[_rev] = combo_system_map[c]
                                _priority.append(_rev)
                                _taken += 1
                                _n_partner += 1
                _priority += _new[:_pmin]
                _n_new += min(len(_new), _pmin)
            if _priority:
                epoch_combos = merge_revisits(
                    _priority, epoch_combos,
                    lambda c: combo_system_map.get(c, {}).get('name') or Path(c).parent.name,
                    n_total=_max_epoch_combos, min_per_system=_pmin)
                _pr_msg = (f"  Priority: {len(_ready_groups)} ready group(s), "
                           + (f"{_n_ref} reference pair(s) + {_n_partner} reversed partner(s), "
                              if _n_ref else '')
                           + f"{_n_conf} pair(s) awaiting a full-length confirming run, "
                           f"{_n_new} not yet run; {len(epoch_combos)} combos this epoch")
                print(_pr_msg)
                _revisit_detail_lines.append(_pr_msg)

        def _run_steps(sys_name: str, combo_name: str):
            """(equil, prod) MD steps for a combo's run this epoch: the stage's
            run_steps (short exploratory runs), except a pair awaiting its
            full-length confirming run; None = the template's full length."""
            st = stage_state.get(sys_name) if per_system else None
            if not st or st['done'] or st['stage'] is None:
                return None
            if combo_name in {Path(g).name for g in _system_group_pool(sys_name)}:
                return None                      # group runs are full length
            rs = stages[st['stage']].get('run_steps')
            if not rs:
                return None
            _await, _ = system_stages.combo_pair_flags(bandit_state, sys_name, combo_name,
                                                       stages[st['stage']]['name'])
            return None if _await else (int(rs['equil']), int(rs['prod']))

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
            update_shared=update_shared,
            stage_of_system=_stage_names(),
            run_steps_fn=_run_steps if per_system else None,
            stage_cfg_of_system=({s: stages[stage_state[s]['stage']] for s in _active_systems}
                                 if per_system else None),
        )
        all_stats.append(stats)
        if algorithm == 'neurallinear_ts':
            save_bandit_state(bandit_state_path, bandit_state)
        # Cost ledger (mllf.cb.cost_ledger): simulated ns per system and stage.
        _ledger_path = checkpoint_dir / LEDGER_FILE
        _stage_now = _stage_names()
        record_runs(_ledger_path, stats.get('run_dirs', []), 'train', epoch, _stage_now)

        # Per-system stage progress: record each combos_balanced stage's
        # latest runs, then advance systems whose criterion holds (or that
        # used up the stage's epochs). A system past its last stage is done
        # and its final biases are exported.
        _stage_lines: List[str] = []
        _ended: set = set()            # systems that left a stage this epoch
        _to_export: List[str] = []     # systems that finished their last stage
        _ran: Dict[str, tuple] = {}    # system -> (epoch within its stage, stage max)
        # for the epoch_summary.log block (mllf.cb.epoch_log)
        _log_passed: List[str] = []
        _log_reopened: List[Tuple[str, int]] = []
        _log_rows: List[Dict] = []
        _log_changes: List[Dict] = []
        if per_system:
            for _sys, _rd, _ok in stats.get('run_dirs', []):
                _st = stage_state.get(_sys)
                if not _st or _st['done'] or not _ok:
                    continue
                _adv = stages[_st['stage']].get('advance') or {}
                if _adv.get('type') not in ('combos_balanced', 'groups_done'):
                    continue
                if (_adv.get('type') == 'groups_done'
                        and Path(_rd).parent.name not in {Path(g).name for g in _system_group_pool(_sys)}):
                    continue
                try:
                    _sd = torch.load(Path(_rd) / 'epoch_results.pt', weights_only=False)
                except Exception:
                    continue
                _cdir = Path(_rd).parent
                _nsubs = [len(subs) for _, subs in system_stages.combo_sites(_cdir)]
                _passes = system_stages.site_balanced(
                    _sd, _nsubs, float(_adv.get('min_population_ratio', 0.5)),
                    int(_adv.get('min_site_transitions', 10)))
                _st['combos'][_cdir.name] = _passes
                if _passes and _adv.get('type') == 'groups_done':
                    # first passing run: the group is retired (run kept for reference)
                    if _cdir.name not in (_st.get('groups_done') or {}):
                        _log_passed.append(f"{_sys}/{_cdir.name.split('_site')[0]}")
                    _st.setdefault('groups_done', {}).setdefault(_cdir.name, str(_rd))
                elif _adv.get('type') == 'groups_done' and _cdir.name not in (_st.get('groups_done') or {}):
                    # group_max_failures: a group failing that many runs goes
                    # back to the pair phase -- its pairs with an unsampled
                    # substituent are reopened and it waits for them.
                    _gmax = stages[_st['stage']].get('group_max_failures')
                    _fails = _st.setdefault('group_failures', {})
                    _fails[_cdir.name] = int(_fails.get(_cdir.name, 0)) + 1
                    if _gmax is not None and _fails[_cdir.name] >= int(_gmax):
                        _keys = system_stages.group_failed_pairs(_sys, _cdir, _sd)
                        system_stages.reopen_pairs(bandit_state, _keys, stages[_st['stage']]['name'])
                        _st.setdefault('group_hold', {})[_cdir.name] = _keys
                        _fails[_cdir.name] = 0
                        _st.setdefault('groups_reopened', {})[_cdir.name] = \
                            int((_st.get('groups_reopened') or {}).get(_cdir.name, 0)) + 1
                        _gline = (f"  [GROUP] {_sys}/{_cdir.name}: failed {_gmax} runs; back to the "
                                  f"pair phase ({len(_keys)} pair(s) reopened)")
                        print(_gline)
                        _stage_lines.append(_gline)
                        _log_reopened.append((f"{_sys}/{_cdir.name.split('_site')[0]}", len(_keys)))
                        if algorithm == 'neurallinear_ts':
                            save_bandit_state(bandit_state_path, bandit_state)
            _val_set_ps = set(validation_combos)
            for _sys in _active_systems:
                _st = stage_state[_sys]
                _stage = stages[_st['stage']]
                _adv = _stage.get('advance') or {}
                _type = _adv.get('type', 'epochs')
                _st['stage_epochs'] += 1
                _ran[_sys] = (_st['stage_epochs'], int(_stage['epochs']))
                _prog = None
                _nres = _nret = _rk = None
                if _type == 'pairs_resolved':
                    _keys = {k for c in _system_stage_pool(_sys) if c not in _val_set_ps
                             for k in system_stages.combo_pair_keys(_sys, c)}
                    _prog = system_stages.pairs_progress(
                        bandit_state, _keys, _stage['name'], _adv, bandit_cfg.get('offset_prior_sd'))
                elif _type == 'groups_done':
                    _gn = [Path(g).name for g in _system_group_pool(_sys)]
                    _prog = (sum(1 for n in _gn if n in (_st.get('groups_done') or {})), len(_gn))
                    # reference pairs resolved (what group readiness depends on)
                    _rk = _ref_keys.get(_sys, set())
                    _store = (bandit_state or {}).get('_pair_offsets', {})
                    _res = {k for k in _rk if system_stages.pair_resolved(
                        _store.get(k), _stage['name'], int(_adv.get('successes', 2)),
                        _adv.get('max_offset_sd'), bandit_cfg.get('offset_prior_sd'))}
                    _nres = len(_res)
                    _nret = sum(1 for k in _rk if k not in _res and system_stages.pair_retired(
                        _store.get(k), _stage['name'], _stage.get('retire_after_epochs')))
                elif _type == 'combos_balanced':
                    _prog = system_stages.combos_progress(
                        _st, _adv, [Path(c).name for c in _system_stage_pool(_sys)
                                    if c not in _val_set_ps])
                _met = (_prog is not None and _st['stage_epochs'] >= int(_adv.get('min_epochs', 1))
                        and system_stages.criterion_met(_adv, *_prog))
                _what = {'pairs_resolved': 'pairs resolved', 'groups_done': 'groups passed'}.get(
                    _type, 'combos balanced')
                _line = (f"  [STAGE] {_sys}: {_stage['name']} epoch {_st['stage_epochs']}/{_stage['epochs']}"
                         + (f", {_prog[0]}/{_prog[1]} {_what}" if _prog else '')
                         + (f" ({_nres}/{len(_rk)} reference pairs resolved"
                            + (f", {_nret} retired" if _stage.get('retire_after_epochs') is not None else '')
                            + ")" if _type == 'groups_done' else ''))
                if _met or _st['stage_epochs'] >= int(_stage['epochs']):
                    _reason = ('criterion' if _met else
                               'max_epochs' if _prog is not None else 'epochs')
                    _nxt = system_stages.advance(_st, stages, _sys, epoch, _reason, _prog)
                    _line += f" -> {'done' if _nxt is None else stages[_nxt]['name']} ({_reason})"
                    _log_changes.append({'system': _sys, 'stage': _stage['name'],
                                         'next': None if _nxt is None else stages[_nxt]['name'],
                                         'reason': _reason})
                    _ended.add(_sys)
                    if _nxt is None:
                        _to_export.append(_sys)
                print(_line)
                _stage_lines.append(_line)
                _log_rows.append({
                    'system': _sys, 'stage': _stage['name'],
                    'stage_epoch': _ran[_sys][0], 'stage_max': _ran[_sys][1],
                    'progress': ((_prog[0], _prog[1], epoch_log.PROGRESS_UNITS.get(_type, 'combos'))
                                 if _prog else None),
                    'ref': (_nres, len(_rk)) if _rk is not None else None,
                    'retired': _nret if _stage.get('retire_after_epochs') is not None else None})
            system_stages.save_state(stage_state_path, stage_state)
        # Held-out validation: deterministic evaluation on fixed set
        # This provides unbiased ground truth for whether policy is actually improving.
        # Buffered (not written immediately) for the same reason as
        # train_epoch's detail_lines -- see the epoch-summary-first write below.
        _val_detail_lines: List[str] = []
        _log_val, _log_gate = None, None        # for the epoch_summary.log block
        # Validate every eval_freq epochs and always after the last epoch, so the
        # run's end point is measured (and the best model can be chosen).
        # Validate only the systems the current stage trains on (a stage with
        # `systems:` leaves the others for later).
        _stage_schedule = per_system and config.get('eval_schedule') == 'stage'
        if _stage_schedule:
            # eval_schedule: stage -- each system's held-out pairs at the start,
            # middle and end of each of its stages (epochs 1, ceil(n/2), n of
            # an n-epoch stage), and in the epoch it meets a stage's criterion.
            _stage_systems = {s for s, (e, n) in _ran.items()
                              if e in (1, (n + 1) // 2, n) or s in _ended}
        elif per_system:
            _stage_systems = set(_active_systems)
        else:
            _stage_systems = (set(current_stage['systems'])
                              if curriculum_enabled and current_stage.get('systems') else None)
        _epoch_val = ([c for c in validation_combos
                       if combo_system_map.get(c, {}).get('name') in _stage_systems]
                      if _stage_systems else validation_combos)
        if per_system and _stage_schedule and not _stage_systems:
            _epoch_val = []
        if _epoch_val and (_stage_schedule or epoch % config.get('eval_freq', 1) == 0
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
            record_runs(_ledger_path, val_stats.get('run_dirs', []), 'validation', epoch, _stage_now)
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
            _log_val = stats['val_pair_success']

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
                    _log_gate = f"below accepted {_gate_score} - {_margin}: model restored"
                else:
                    _gate_state = copy.deepcopy(policy.state_dict())
                    _gate_score = _score if _gate_score is None else max(_gate_score, _score)
                    _gate_msg = (f"  [VALIDATION GATE] held-out success {_score}: model accepted "
                                 f"(reference {_gate_score})")
                    _log_gate = "model accepted"
                _last_val_score = _score
                if _best_score is None or _score > _best_score:
                    _best_state = copy.deepcopy(policy.state_dict())
                    _best_score, _best_epoch = _score, epoch
                    torch.save({'policy_state_dict': _best_state, 'epoch': epoch,
                                'val_success': _score}, checkpoint_dir / 'best_model.pt')
                    _gate_msg += f"; new best (epoch {epoch}) saved to best_model.pt"
                    _log_gate += ", new best saved"
                print(_gate_msg)
                _val_detail_lines.append(_gate_msg)

        for _sys in _to_export:
            _export_final(_sys)
        _cost = summarize_ledger(load_ledger(_ledger_path))
        if _cost:
            # Training ns only: validation runs are an internal check of the
            # shared model, not part of converging a system (or of the ALF
            # comparison), and are listed separately.
            _stage_lines.append("  [COST] training ns so far: " + ', '.join(
                f"{_s} {_c['train']['ns']:.1f}" for _s, _c in sorted(_cost.items())))
            print(_stage_lines[-1])
            _vns = sum(_c['validation']['ns'] for _c in _cost.values())
            if _vns:
                _stage_lines.append(f"  [COST] validation ns so far (not in the totals above): {_vns:.1f}")
                print(_stage_lines[-1])

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
            # Short exploratory and full-length runs, each judged by its own
            # success test (the line above mixes them under one diagnostic).
            for _key, _lab in (('pair_success_short', 'train short runs'),
                               ('pair_success_full', 'train full-length runs')):
                if (stats.get(_key) or {}).get('total'):
                    _summary_lines.append(format_pair_success(stats[_key], _lab))
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
                if per_system:
                    # Per-system stages: one concise block (mllf.cb.epoch_log);
                    # the full detail stays in the job's stdout.
                    append_summary(summary_log_path, '\n' + _epoch_log_block(
                        epoch, epoch_combos, stats, _log_val, _log_gate, _log_passed, _log_reopened,
                        _log_rows, _log_changes, _cost))
                else:
                    append_summary(summary_log_path, '\n'.join(_summary_lines))
                    _epoch_detail_lines = (_revisit_detail_lines + stats.get('detail_lines', [])
                                           + _val_detail_lines + _stage_lines)
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
