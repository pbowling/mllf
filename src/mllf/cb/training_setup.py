"""One-time setup of an online-training run (``mllf.cb.training_loop``).

Reading and checking the workflow config, building every system's combo
pool, the policy and its optimizers, restoring a checkpoint or pretrained
weights, and choosing the held-out validation combos.
"""
import concurrent.futures
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from mllf.cb import system_stages
from mllf.cb.bandit_config import (
    BANDIT_ALGORITHMS,
    BanditConfig,
    bandit_algorithm,
    with_hindsight_defaults,
)
from mllf.cb.curriculum import apply_stage_combo_filter, filter_combos_by_curriculum
from mllf.cb.hindsight import BIAS_DIMS
from mllf.cb.policy import ContextBaseline, UnimolPolicy, unimol_options
from mllf.file_handling.combo_sets import build_combo_sets
from mllf.file_handling.generate_combinations import find_site_sub_files
from mllf.file_handling.manifests import (
    build_system_manifest,
    count_combinations_fast,
    create_and_load_system_manifest,
)


log = logging.getLogger(__name__)


# config

def load_training_config(config_path: str) -> Dict:
    """The workflow config with hindsight defaults filled in; in deploy mode
    the trained model becomes the starting point. Raises on an invalid
    mode or deploy setup."""
    with open(config_path, 'r') as f:
        config = with_hindsight_defaults(yaml.safe_load(f))
    log.info(f"Loaded config from {config_path}")
    mode = config.get('mode', 'train')
    if mode not in ('train', 'deploy'):
        raise ValueError(f"mode must be 'train' or 'deploy', got {mode!r}")
    if mode == 'deploy':
        _check_deploy(config)
    unknown = BanditConfig.unknown_keys(config.get('bandit'))
    if unknown:
        log.warning(f"  Warning: bandit keys not used by any setting: {unknown}")
    return config


def _check_deploy(config: Dict) -> None:
    deploy_cfg = config.get('deploy') or {}
    bc = config.get('bandit') or {}
    if not (config.get('curriculum') or {}).get('enabled'):
        raise ValueError("mode: deploy needs curriculum stages (curriculum.enabled: true)")
    if (bandit_algorithm(bc) != 'hindsight' or bc.get('update_target') != 'hindsight'
            or not bc.get('pair_offsets', True)):
        raise ValueError("mode: deploy needs bandit.algorithm: hindsight, "
                         "update_target: hindsight and pair_offsets (per-pair learning)")
    model = deploy_cfg.get('model_path')
    if not model or not Path(model).exists():
        raise FileNotFoundError(f"deploy.model_path not found: {model}")
    out = Path((config.get('output') or {}).get('base_dir', 'checkpoints')).resolve()
    if Path(model).resolve().parent == out:
        raise ValueError("deploy output.base_dir is the trained model's directory; "
                         "use a separate directory so the trained model is never overwritten")
    # The trained model is the starting point (loaded like a pretrain checkpoint).
    config.setdefault('pretrain', {})['model_path'] = model
    log.info(f"Mode: deploy (model {model}; shared model "
             f"{'updated' if deploy_cfg.get('update_shared_model') else 'fixed'}; "
             "per-pair offsets learn)")


# combo pool

def _effective_max_subs(sys_entry: Dict, default_max: int, large_site_threshold: int,
                        large_site_max_subs: int) -> int:
    """max_subs_per_site for a system's new manifest: its own key, else
    reduced for a site with more than ``large_site_threshold`` substituents."""
    if 'max_subs_per_site' in sys_entry:
        return sys_entry['max_subs_per_site']
    found = find_site_sub_files(Path(sys_entry['input_dir']))
    max_site_subs = max((len(subs) for subs in found.values()), default=0)
    if max_site_subs > large_site_threshold:
        log.debug(f"  [{sys_entry.get('name', 'unknown')}] site has {max_site_subs} subs > "
                  f"{large_site_threshold}; auto-reducing max_subs_per_site to "
                  f"{large_site_max_subs}")
        return large_site_max_subs
    return default_max


def build_manifests(config: Dict, systems_cfg: List[Dict]) -> Tuple[Dict[str, Dict], List[str]]:
    """Every system's manifest (loaded, or created: small systems in
    parallel, large ones one at a time to bound memory). Returns
    (combo path -> system entry, all combo paths in system order)."""
    seed = config.get('seed', 42)
    log.info(f"\nBuilding system manifests ({len(systems_cfg)} systems)")
    large_threshold = config.get('large_combo_threshold', 10_000)
    creation_workers = config.get('max_manifest_workers', 4)
    manifest_max_subs = config.get('max_subs_per_site', 10)
    large_site_threshold = config.get('large_site_sub_threshold', 12)
    large_site_max_subs = config.get('large_site_max_subs', 6)
    max_combos = config.get('max_combos_per_system', None)
    log.debug(f"  max_subs_per_site : {manifest_max_subs} (default); "
              f"auto-reduced to {large_site_max_subs} for any site with > {large_site_threshold} "
              "subs")
    if max_combos is not None:
        log.debug(f"  max_combos_per_system : {max_combos:,} (paths loaded per system)")

    results: Dict[int, Dict] = {}
    eff_max: Dict[int, int] = {}
    small: List[tuple] = []     # (idx, sys_entry): estimated combos < threshold
    large: List[tuple] = []     # (idx, sys_entry, estimate)
    for i, sys_entry in enumerate(systems_cfg):
        if (Path(sys_entry['input_dir']) / 'manifest.txt').exists():
            results[i] = build_system_manifest(sys_entry, max_combos=max_combos, seed=seed)
            continue
        eff_max[i] = _effective_max_subs(sys_entry, manifest_max_subs, large_site_threshold,
                                         large_site_max_subs)
        est = count_combinations_fast(sys_entry, max_subs_per_site=eff_max[i])
        if est >= large_threshold:
            large.append((i, sys_entry, est))
        else:
            small.append((i, sys_entry))

    if small:
        w = min(creation_workers, len(small))
        log.debug(f"  {len(small)} small system(s) (< {large_threshold:,} combos): "
                  f"creating in parallel ({w} workers)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=w) as ex:
            futures = {ex.submit(create_and_load_system_manifest, s, eff_max[i], max_combos,
                                 seed): i
                       for i, s in small}
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()
    if large:
        if small:
            log.debug(f"  All {len(small)} small system(s) complete. "
                      f"Now creating {len(large)} large system(s) one at a time.")
        for i, sys_entry, est in large:
            log.debug(f"  Large system (~{est:,} combos): "
                      f"{sys_entry.get('name', sys_entry.get('input_dir', 'unknown'))}"
                      "; creating alone")
            results[i] = create_and_load_system_manifest(sys_entry, max_subs_per_site=eff_max[i],
                                                         max_combos=max_combos, seed=seed)

    combo_system_map: Dict[str, Dict] = {}
    all_combos: List[str] = []
    for i in range(len(systems_cfg)):
        for line in results[i]['log']:
            log.debug('%s', line)
        for cp in results[i]['combos']:
            combo_system_map[cp] = results[i]['sys_entry']
        all_combos.extend(results[i]['combos'])
    log.info(f"\nTotal combos across all systems: {len(all_combos)}")
    return combo_system_map, all_combos


def build_all_combo_sets(config: Dict, systems_cfg: List[Dict], combo_system_map: Dict[str, Dict],
                         explicit_combos: Dict[str, Dict]) -> Dict[str, Dict[str, List[str]]]:
    """Systems' ``combo_sets:`` (e.g. ALF's substituent groups and their
    pairs), selected by a stage's ``combo_set:``. They are in no manifest:
    their infos go into ``explicit_combos`` so the dirs can be created."""
    by_system: Dict[str, Dict[str, List[str]]] = {}
    for sys_entry in systems_cfg:
        if not sys_entry.get('combo_sets'):
            continue
        sets, warns = build_combo_sets(sys_entry, config.get('alf_root'), registry=explicit_combos)
        by_system[sys_entry.get('name')] = sets
        for paths in sets.values():
            for p in paths:
                combo_system_map[p] = sys_entry
        log.debug('%s', f"  [{sys_entry.get('name')}] combo sets: "
                  + ', '.join(f"{k} {len(v)}" for k, v in sets.items()))
        for w in warns:
            log.warning(f"    warning: {w}")
    return by_system


def print_curriculum(curriculum_cfg: Dict) -> None:
    if not curriculum_cfg.get('enabled', False):
        return
    stages = curriculum_cfg.get('stages', [])
    log.info(f"\nCurriculum: {len(stages)} stages")
    for i, s in enumerate(stages, 1):
        log.info(f"  Stage {i}: {s['name']}; "
                 f"{s.get('min_subs_per_site', 1)}-{s.get('max_subs_per_site', '?')} subs/site, "
                 f"{s.get('min_sites', 1)}-{s.get('max_sites', '?')} sites, "
                 f"{s['epochs']} epochs")


# policy

NUM_BIAS_TYPES = len(BIAS_DIMS)   # linear, quadratic, skew, end


def build_policy(config: Dict, device: str, bayesian_default: bool = False) -> UnimolPolicy:
    """The UnimolPolicy a workflow config describes (the unimol.* inputs, with
    ``UNIMOL_DEFAULTS``, must match the pretrained model's)."""
    policy_cfg = config.get('training', {}).get('policy', {})
    unimol = unimol_options(config)
    return UnimolPolicy(
        unimol_dim=policy_cfg.get('unimol_dim', 512),
        mlp_hidden=policy_cfg.get('mlp_hidden', 64),
        mlp_out_dim=NUM_BIAS_TYPES,
        use_dual_embeddings=True,
        env_input=unimol['env_input'],
        site_size_input=unimol['site_size_input'],
        use_bayesian_heads=policy_cfg.get('use_bayesian_heads', bayesian_default),
        bayesian_prior_precision=BanditConfig.from_config(config).prior_precision,
    ).to(device)


def freeze_heads(policy: UnimolPolicy, names: List[str]) -> None:
    """No online gradient for these bias heads (training.policy.freeze_heads)."""
    if not names:
        return
    for name in names:
        if name not in BIAS_DIMS:
            log.warning(f"  Warning: unknown freeze_heads entry '{name}'; skipping")
            continue
        for p in policy.edge_mlp.mlps[BIAS_DIMS.index(name)].parameters():
            p.requires_grad = False
    log.info(f"Frozen bias heads (no online gradient): {names}")


def make_rl_optimizer(policy: UnimolPolicy, lr: float):
    """Adam over the trainable edge-MLP parameters, and its plateau scheduler."""
    opt = torch.optim.Adam([p for p in policy.edge_mlp.parameters() if p.requires_grad], lr=lr)
    return opt, make_scheduler(opt)


def make_scheduler(opt):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3,
                                                      min_lr=1e-7, verbose=False)


def make_context_baseline(train_cfg: Dict, algorithm: str, system_to_idx: Dict[str, int],
                          device: str):
    """REINFORCE's context-dependent baseline (training.context_baseline),
    or (None, None). It judges an action against what is typical for that
    system and context rather than for every system pooled together."""
    cfg = train_cfg.get('context_baseline', {})
    if not cfg.get('enabled', False):
        return None, None
    if algorithm == 'hindsight':
        log.warning("  Warning: training.context_baseline.enabled=True but "
                    "bandit.algorithm == 'hindsight'; context baseline is a "
                    "REINFORCE-only concept (the hindsight update has no "
                    "advantage/baseline step); ignoring.")
        return None, None
    net = ContextBaseline(
        # 64, not mlp_hidden: BiasHeadMLP's trunk output (forward_features) is 64 wide.
        trunk_dim=64, num_bias_types=NUM_BIAS_TYPES, num_systems=len(system_to_idx),
        system_embed_dim=cfg.get('system_embed_dim', 8), hidden=cfg.get('hidden_dim', 32),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get('lr', 1e-3))
    log.info(f"Context baseline enabled: {sum(p.numel() for p in net.parameters())} "
             f"params, {len(system_to_idx)} systems (incl. unknown fallback)")
    return net, opt


def policy_weights(checkpoint: Dict):
    """Policy weights from a pretraining checkpoint ('policy_state') or an
    online-training one ('policy_state_dict'); None if neither."""
    return checkpoint.get('policy_state') or checkpoint.get('policy_state_dict')


def check_pretrain_inputs(config: Dict, pretrain_path: Optional[str]) -> None:
    """The model's node inputs must be built the same way online
    (unimol.ligand_other_sites / site_size_input, in pretrain_metadata.json)."""
    if not pretrain_path:
        return
    meta_path = Path(pretrain_path).parent / 'pretrain_metadata.json'
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    for key in ('ligand_other_sites', 'site_size_input'):
        built = meta.get(key, False)
        if bool(built) != bool(config.get('unimol', {}).get(key, False)):
            raise ValueError(f"{pretrain_path} was pretrained with unimol.{key}={built}; "
                             "set the same in this config")


def load_pretrained(policy: UnimolPolicy, pretrain_path: str, device: str) -> None:
    log.info(f"Loading pretrained policy from {pretrain_path}")
    checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
    if policy_weights(checkpoint) is not None:
        missing, unexpected = policy.load_state_dict(policy_weights(checkpoint), strict=False)
        if missing or unexpected:
            log.warning("  Warning: partial weight load; "
                        f"{len(missing)} missing, {len(unexpected)} unexpected keys")
        else:
            log.info("  Loaded policy weights from pretrain checkpoint")
    weight_norm = sum(p.norm().item() ** 2 for p in policy.parameters()) ** 0.5
    log.debug(f"  Policy weight norm: {weight_norm:.4f}")
    log.debug(f"  Scale factors: {policy.scale_factors.tolist()}")


def restore(policy, rl_optimizer, baseline_net, baseline_optimizer, checkpoint_dir: Path,
            pretrain_path: Optional[str], device: str) -> Tuple[int, Optional[int]]:
    """Resume from the latest checkpoint_*.pt in checkpoint_dir, else start
    from the pretrained model. Returns (start epoch, saved stage index)."""
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt')) if checkpoint_dir.exists() else []
    if not checkpoints:
        if pretrain_path and Path(pretrain_path).exists():
            load_pretrained(policy, pretrain_path, device)
        return 0, 0
    latest = checkpoints[-1]
    log.info(f"Resuming from checkpoint: {latest}")
    checkpoint = torch.load(latest, map_location=device, weights_only=False)
    try:
        policy.load_state_dict(checkpoint['policy_state_dict'])
        rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
        # Only checkpoints saved with a context baseline carry its state.
        if baseline_net is not None and 'baseline_state_dict' in checkpoint:
            baseline_net.load_state_dict(checkpoint['baseline_state_dict'])
            if 'baseline_optimizer_state_dict' in checkpoint:
                baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])
            log.info("  Resumed context baseline state")
        elif baseline_net is not None:
            log.info("  Note: context baseline enabled but no saved baseline state found "
                     "in this checkpoint (starting baseline from scratch)")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        else:
            match = re.search(r'checkpoint_(\d+)', latest.name)
            start_epoch = (int(match.group(1)) + 1) if match else 0
            log.warning("  Warning: 'epoch' key missing from checkpoint; inferred start_epoch="
                        f"{start_epoch} from filename")
        return start_epoch, checkpoint.get('current_stage_idx', None)
    except (RuntimeError, KeyError) as err:
        log.warning(f"  Warning: checkpoint loading failed; {err}")
        log.info("  Falling back to pretrained model (training restarts from epoch 0).")
        if pretrain_path and Path(pretrain_path).exists():
            ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
            if policy_weights(ckpt) is not None:
                policy.load_state_dict(policy_weights(ckpt))
                log.info("  Loaded policy weights from pretrain checkpoint")
        return 0, 0


# validation set

def _draw_per_system(by_sys: Dict[str, List[str]], k_per_system: int) -> List[str]:
    """Seeded draw of k combos per system (leaving each at least one)."""
    out = []
    for name in sorted(by_sys, key=str):
        cands = sorted(by_sys[name])
        k = min(k_per_system, max(len(cands) - 1, 0))
        if k > 0:
            idx = np.random.RandomState(seed=42).choice(len(cands), size=k, replace=False)
            out.extend(cands[i] for i in sorted(idx))
    return out


def _by_system(combos, combo_system_map) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c in combos:
        out.setdefault(combo_system_map.get(c, {}).get('name'), []).append(c)
    return out


def choose_validation(config: Dict, *, deploy: bool, start_epoch: int, num_epochs: int,
                      active: List[str], train_combos: List[str],
                      combo_system_map: Dict[str, Dict],
                      combo_sets_by_system: Dict[str, Dict[str, List[str]]], curriculum_cfg: Dict,
                      curriculum_enabled: bool, seed: int,
                      path: Path) -> Tuple[List[str], List[str]]:
    """The held-out validation combos (persisted to ``path`` so a resumed run
    keeps them) and the training pool without them.

    Per system (validation_per_system), from a file (validation_combos_file,
    plus validation_per_new_system for systems it doesn't cover), or a
    fraction of the pool (validation_fraction); none in deploy mode.
    """
    fixed_file = config.get('validation_combos_file')
    per_system = config.get('validation_per_system')
    fresh = start_epoch == 0 and num_epochs > 0
    if deploy:
        log.info("Held-out validation: none (mode: deploy)")
        return [], active
    if per_system is not None and int(per_system) == 0 and not fixed_file:
        log.info("Held-out validation: none (validation_per_system: 0)")
        return [], active

    if fresh and per_system and not fixed_file:
        # From every system in the run (all stages, so a system that joins
        # later is held out from the start), with the first stage's limits.
        stage0 = dict((curriculum_cfg.get('stages') or [{}])[0]) if curriculum_enabled else {}
        stage0.pop('systems', None)
        if stage0.get('combo_set'):
            # Training runs explicit sets: hold out manifest pairs in no
            # combo of any of the system's sets, so validation measures the
            # model on pairs training never runs.
            used = {name: {k for paths in sets.values() for p in paths
                           for k in system_stages.combo_pair_keys(name, p)}
                    for name, sets in combo_sets_by_system.items()}
            pool = [c for c in filter_combos_by_curriculum(train_combos, 1, 1, 2, 2)
                    if (n := combo_system_map.get(c, {}).get('name')) in used
                    and not set(system_stages.combo_pair_keys(n, c)) & used[n]]
        else:
            pool = (apply_stage_combo_filter(train_combos, stage0, curriculum_cfg, stage_idx=0,
                                             seed=seed, apply_transition_filter=False)
                    if stage0 else list(train_combos))
        by_sys = _by_system(pool, combo_system_map)
        validation = _draw_per_system(by_sys, int(per_system))
        log.info(f"Held-out validation set: {per_system} per system, {len(validation)} combos "
                 f"over {len(by_sys)} system(s) (excluded from training pool)")
    elif fresh and fixed_file:
        with open(fixed_file, 'r') as f:
            fixed = json.load(f)
        pool_set = set(active)
        validation = [c for c in fixed if c in pool_set]
        if len(validation) < len(fixed):
            log.warning(f"  Warning: {len(fixed) - len(validation)} of {len(fixed)} combos in "
                        f"{fixed_file} are not in this run's pool; validating on the rest")
        n_from_file = len(validation)
        # Systems the file doesn't cover get their own seeded draw, so a run
        # that adds systems still validates the new ones.
        covered = {combo_system_map.get(c, {}).get('name') for c in validation}
        in_file = set(validation)
        by_sys = _by_system([c for c in active
                             if combo_system_map.get(c, {}).get('name') not in covered
                             and c not in in_file], combo_system_map)
        validation += _draw_per_system(by_sys, int(config.get('validation_per_new_system', 6)))
        log.info('%s', f"Held-out validation set from {fixed_file}: {n_from_file} combos"
                 + (f" + {len(validation) - n_from_file} drawn for {len(by_sys)} system(s) "
                    "not in it" if len(validation) > n_from_file else "")
                 + " (excluded from training pool)")
    elif fresh and len(active) > 20:
        return _fraction_validation(config, active, path)
    elif path.exists():
        try:
            with open(path, 'r') as f:
                validation = json.load(f)
            held = set(validation)
            active = [c for c in active if c not in held]
            log.info(f"Restored held-out validation set from {path}: "
                     f"{len(validation)} combos (excluded from training pool)")
            return validation, active
        except Exception as err:
            log.warning(f"  Warning: could not restore validation combo set: {err}")
        return [], active
    else:
        return [], active

    held = set(validation)
    with open(path, 'w') as f:
        json.dump(validation, f)
    return validation, [c for c in active if c not in held]


def _fraction_validation(config: Dict, active: List[str],
                         path: Path) -> Tuple[List[str], List[str]]:
    """validation_fraction of the pool, capped by the first progression
    stage's combo budget (at least 10, at most max(100, budget x fraction))."""
    frac = config.get('validation_fraction', 0.1)
    progression = config.get('progression', {})
    max_stage_combos = 1000
    if progression and 'stages' in progression and len(progression['stages']) > 0:
        max_stage_combos = progression['stages'][0].get('max_train_combos_per_stage', 1000)
    size = min(max(100, int(max_stage_combos * frac)), max(10, int(len(active) * frac)))
    idx = np.random.RandomState(seed=42).choice(len(active), size=min(size, len(active)),
                                                replace=False)
    validation = [active[i] for i in sorted(idx)]
    remaining = [c for i, c in enumerate(active) if i not in idx]
    try:
        with open(path, 'w') as f:
            json.dump(validation, f)
    except Exception as err:
        log.warning(f"  Warning: could not save validation combo set: {err}")
    log.info(f"Held-out validation: {len(validation)} combos, fixed and evaluated "
             f"deterministically each epoch; training pool: {len(remaining)} combos")
    return validation, remaining


def check_algorithm(bandit: BanditConfig, use_bayesian_heads: bool) -> None:
    if bandit.algorithm not in BANDIT_ALGORITHMS:
        raise ValueError(f"bandit.algorithm must be one of {BANDIT_ALGORITHMS}, "
                         f"got {bandit.raw.get('algorithm')!r}")
    if use_bayesian_heads and bandit.algorithm != 'hindsight':
        log.warning("  Warning: training.policy.use_bayesian_heads=True but "
                    "bandit.algorithm != 'hindsight'; the policy will have "
                    "Bayesian heads but nothing will train them online.")
