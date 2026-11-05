"""Online CB-policy training: the run's state and its per-epoch phases.

``run_training(config_path)`` sets up a ``TrainingRun`` (``mllf.cb.training_setup``)
and, every epoch:

1. plans the epoch's pool: per-system stages (``curriculum.per_system``,
   always in deploy mode) skip empty stages, apply scouting, drop resolved
   or retired pairs and add ready groups; otherwise the global curriculum
   advances;
2. selects the epoch's combos (uncertainty, stratified or uniform), then
   revisits and per-system priorities;
3. trains (``mllf.cb.reinforce_trainer.train_epoch``);
4. records group runs and advances per-system stages;
5. validates on the held-out combos (with the regression validation gate);
6. exports finished systems, logs the epoch, checkpoints.
"""
import collections
import copy
import functools
import json
import logging
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from mllf.cb import epoch_log, group_correction, system_stages
from mllf.cb import training_setup as setup
from mllf.cb.bandit_config import BanditConfig
from mllf.cb.cost_ledger import LEDGER_FILE, load_ledger, record_runs, summarize_ledger
from mllf.cb.curriculum import (
    apply_stage_combo_filter,
    build_system_index,
    sample_combos_stratified,
)
from mllf.cb.epoch_runs import EpochOptions
from mllf.cb.export_biases import export_final_biases, label_final_biases
from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.hindsight import combo_node_labels, merge_revisits, pair_attention, select_revisits
from mllf.cb.reinforce_trainer import append_summary, format_pair_success, train_epoch
from mllf.cb.warning_summary import WarningSummary
from mllf.cb.workflow_utils import (
    action_std_score,
    bayesian_variance_score,
    load_bandit_state,
    save_bandit_state,
    select_combos_by_uncertainty,
)
from mllf.file_handling.archive import archive_config, archive_systems
from mllf.file_handling.combo_sets import reversed_pair
from mllf.file_handling.manifests import build_combo_dir_lookup, ensure_combo_dir_exists
from mllf.log import log_to_file


log = logging.getLogger(__name__)

# Every message of a run, with time stamps, in output.base_dir (the console
# shows progress, the epoch summaries, warnings and errors).
LOG_FILE = 'training.log'


# Stage keys that set the whole epoch's training; one batch mixes systems, so
# per-system stages cannot honour them.
_EPOCH_WIDE_STAGE_KEYS = ('update_policy', 'reward_override', 'policy_lr', 'bc_mode',
                          'bc_source_epoch',
                          'replay_from_epoch', 'deterministic_actions', 'bc_auxiliary_weight',
                          'max_train_combos')


@dataclass
class ValidationGate:
    """bandit.shared_update: regression moves the full network every epoch,
    so it is kept only while held-out pairs hold up; the best validated
    model is saved and is what the run ends on if the last one is worse."""
    accepted_state: Optional[dict] = None
    accepted_score: Optional[int] = None
    key: Optional[frozenset] = None          # the held-out set the scores refer to
    best_state: Optional[dict] = None
    best_score: Optional[int] = None
    best_epoch: Optional[int] = None
    last_score: Optional[int] = None


@dataclass
class TrainingRun:
    """Everything an online-training run keeps between epochs."""
    config: Dict
    bandit: BanditConfig
    device: str
    deploy: bool
    systems_cfg: List[Dict]
    combo_system_map: Dict[str, Dict]
    train_combos: List[str]
    combo_sets_by_system: Dict[str, Dict[str, List[str]]]
    explicit_combos: Dict[str, Dict]
    policy: torch.nn.Module
    rl_optimizer: torch.optim.Optimizer
    rl_scheduler: object
    checkpoint_dir: Path
    num_epochs: int
    start_epoch: int
    baseline_net: Optional[torch.nn.Module] = None
    baseline_optimizer: Optional[torch.optim.Optimizer] = None
    system_to_idx: Dict[str, int] = field(default_factory=dict)
    history_replay_cfg: Dict = field(default_factory=dict)
    bandit_state: Dict = field(default_factory=dict)
    # curriculum
    curriculum_cfg: Dict = field(default_factory=dict)
    stages: List[Dict] = field(default_factory=list)
    per_system: bool = False
    current_stage_idx: int = 0
    current_stage_epoch: int = 0
    stage_state: Dict[str, Dict] = field(default_factory=dict)
    combos_by_system: Dict[str, List[str]] = field(default_factory=dict)
    # pools
    active_train_combos: List[str] = field(default_factory=list)
    validation_combos: List[str] = field(default_factory=list)
    epoch_config: Dict = field(default_factory=dict)
    update_shared: bool = True
    # accumulated
    all_stats: List[Dict] = field(default_factory=list)
    ema_baseline: Dict = field(default_factory=dict)
    replay_cache: Dict[str, list] = field(default_factory=dict)
    gate: ValidationGate = field(default_factory=ValidationGate)
    # archive: block (mllf.file_handling.archive); systems archived so far
    archive: Optional[Dict] = None
    archived: Set[str] = field(default_factory=set)
    _pool_cache: Dict[tuple, List[str]] = field(default_factory=dict)
    _scout_tiers: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _scout_summaries: Dict[str, Dict] = field(default_factory=dict)

    @property
    def curriculum_enabled(self) -> bool:
        return bool(self.curriculum_cfg.get('enabled', False))

    @property
    def algorithm(self) -> str:
        return self.bandit.algorithm

    @property
    def seed(self) -> int:
        return self.config.get('seed', 42)

    @property
    def output_cfg(self) -> Dict:
        return self.config.get('output', {})

    @property
    def bandit_state_path(self) -> Path:
        return self.checkpoint_dir / 'bandit_state.json'

    @property
    def stage_state_path(self) -> Path:
        return self.checkpoint_dir / system_stages.STATE_FILE

    @property
    def summary_log_path(self) -> Path:
        return self.checkpoint_dir / 'epoch_summary.log'

    @property
    def ledger_path(self) -> Path:
        return self.checkpoint_dir / LEDGER_FILE

    @property
    def current_stage(self) -> Dict:
        return self.stages[self.current_stage_idx]

    def system_of(self, combo) -> Optional[str]:
        return self.combo_system_map.get(combo, {}).get('name')

    def save_bandit_state(self) -> None:
        if self.algorithm == 'hindsight':
            save_bandit_state(self.bandit_state_path, self.bandit_state)


@dataclass
class EpochPlan:
    """What one epoch trains on."""
    epoch: int
    active_systems: List[str] = field(default_factory=list)
    ref_keys: Dict[str, Set[str]] = field(default_factory=dict)   # reference pair keys per system
    ready_groups: List[str] = field(default_factory=list)
    stage: Dict = field(default_factory=dict)                     # global stage's epoch-wide keys
    max_combos: Optional[int] = None
    system_quota: Optional[Dict[str, int]] = None
    combos: List[str] = field(default_factory=list)
    detail_lines: List[str] = field(default_factory=list)

    @property
    def update_policy(self) -> bool:
        return self.stage.get('update_policy', True)


@dataclass
class EpochRecord:
    """What happened in one epoch, for the console and epoch_summary.log."""
    stage_lines: List[str] = field(default_factory=list)
    ended: Set[str] = field(default_factory=set)              # systems that left a stage
    to_export: List[str] = field(default_factory=list)        # systems past their last stage
    # (epoch within stage, stage max)
    ran: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    passed: List[str] = field(default_factory=list)
    reopened: List[Tuple[str, int]] = field(default_factory=list)
    holdoff: List[str] = field(default_factory=list)
    groupfix: List[str] = field(default_factory=list)
    rows: List[Dict] = field(default_factory=list)
    changes: List[Dict] = field(default_factory=list)
    val_lines: List[str] = field(default_factory=list)
    val_pair_success: Optional[Dict] = None
    gate: Optional[str] = None


# setup

def setup_run(config: Dict) -> TrainingRun:
    """Build the run's combo pool, policy, optimizers and curriculum state,
    resume from a checkpoint if one exists, and choose the validation set."""
    deploy = config.get('mode', 'train') == 'deploy'
    deploy_cfg = config.get('deploy') or {}
    systems_cfg = config.get('systems', [])
    if not systems_cfg:
        raise ValueError("Config must contain a 'systems:' list with at least one entry.")
    combo_system_map, all_combos = setup.build_manifests(config, systems_cfg)
    explicit_combos: Dict[str, Dict] = {}
    combo_sets_by_system = setup.build_all_combo_sets(config, systems_cfg, combo_system_map,
                                                      explicit_combos)
    curriculum_cfg = config.get('curriculum', {})
    setup.print_curriculum(curriculum_cfg)

    log.info("\nInitializing the policy")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")
    log.debug("Using pretrained Uni-Mol representation...")
    train_cfg = config.get('training', {})
    policy_cfg = train_cfg.get('policy', {})
    bandit = BanditConfig.from_config(config)
    setup.check_algorithm(bandit, policy_cfg.get('use_bayesian_heads', False))
    policy = setup.build_policy(config, device)
    setup.freeze_heads(policy, policy_cfg.get('freeze_heads', []))
    rl_optimizer, rl_scheduler = setup.make_rl_optimizer(
        policy, train_cfg.get('optimizer', {}).get('lr', 0.001))
    log.debug(f"Uni-Mol policy: {sum(p.numel() for p in policy.parameters())} params")
    log.debug(f"Uni-Mol dimension: {policy_cfg.get('unimol_dim', 512)}")
    log.debug(f"MLP hidden dimension: {policy_cfg.get('mlp_hidden', 64)}")
    system_to_idx = build_system_index(systems_cfg)
    baseline_net, baseline_optimizer = setup.make_context_baseline(train_cfg, bandit.algorithm,
                                                                   system_to_idx,
                                                                   device)
    history_replay_cfg = config.get('replay', {})
    if history_replay_cfg.get('enabled', False) and bandit.algorithm == 'hindsight':
        log.warning("  Warning: replay.enabled=True but bandit.algorithm == "
                    "'hindsight'; history replay folds an extra REINFORCE-"
                    "style loss into a gradient step hindsight doesn't take; ignoring.")
        history_replay_cfg = {}

    checkpoint_dir = Path(config.get('output', {}).get('base_dir', 'checkpoints'))
    # Per-pair offsets and per-combo failures; never read in reinforce mode.
    bandit_state = load_bandit_state(checkpoint_dir / 'bandit_state.json')
    pretrain_path = config.get('pretrain', {}).get('model_path', None)
    setup.check_pretrain_inputs(config, pretrain_path)
    start_epoch, start_stage_idx = setup.restore(policy, rl_optimizer, baseline_net,
                                                 baseline_optimizer,
                                                 checkpoint_dir, pretrain_path, device)

    log.info("\nTraining")
    curriculum_enabled = curriculum_cfg.get('enabled', False)
    stages = curriculum_cfg.get('stages', []) if curriculum_enabled else []
    if curriculum_enabled:
        num_epochs = sum(stage['epochs'] for stage in stages)
        log.info(f"Curriculum: {len(stages)} stages, {num_epochs} total epochs")
    else:
        num_epochs = train_cfg.get('num_epochs', 100)
        log.info(f"Training for {num_epochs} epochs")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    run = TrainingRun(
        archive=archive_config(config),
        config=config, bandit=bandit, device=device, deploy=deploy, systems_cfg=systems_cfg,
        combo_system_map=combo_system_map, train_combos=all_combos,
        combo_sets_by_system=combo_sets_by_system, explicit_combos=explicit_combos,
        policy=policy, rl_optimizer=rl_optimizer, rl_scheduler=rl_scheduler,
        checkpoint_dir=checkpoint_dir, num_epochs=num_epochs, start_epoch=start_epoch,
        baseline_net=baseline_net, baseline_optimizer=baseline_optimizer,
        system_to_idx=system_to_idx,
        history_replay_cfg=history_replay_cfg, bandit_state=bandit_state,
        curriculum_cfg=curriculum_cfg, stages=stages,
        # Shared-model updates: always in training; in deploy only on request.
        update_shared=(not deploy) or bool(deploy_cfg.get('update_shared_model', False)))
    # Lightweight per-epoch summary next to the (very large) job stdout.
    log.info(f"Epoch summary log: {run.summary_log_path}")
    _init_curriculum(run, start_stage_idx)
    _init_per_system(run)
    run.validation_combos, run.active_train_combos = setup.choose_validation(
        config, deploy=deploy, start_epoch=start_epoch, num_epochs=num_epochs,
        active=run.active_train_combos,
        train_combos=run.train_combos, combo_system_map=combo_system_map,
        combo_sets_by_system=combo_sets_by_system, curriculum_cfg=curriculum_cfg,
        curriculum_enabled=curriculum_enabled, seed=run.seed,
        path=checkpoint_dir / 'validation_combos.json')
    return run


def _stage_epoch_config(run: TrainingRun, stage: Dict) -> Dict:
    """The config an epoch of ``stage`` trains with (its reward_override applied)."""
    if stage.get('reward_override'):
        return {**run.config,
                'reward': {**run.config.get('reward', {}), **stage['reward_override']}}
    return run.config.copy()


def _set_lr(run: TrainingRun, lr: float) -> None:
    for pg in run.rl_optimizer.param_groups:
        pg['lr'] = lr


def _init_curriculum(run: TrainingRun, start_stage_idx: Optional[int]) -> None:
    """The global curriculum's current stage and the pool it starts from."""
    if not run.curriculum_enabled:
        run.active_train_combos = run.train_combos
        run.epoch_config = run.config
        return
    stages = run.stages
    if run.start_epoch > 0:
        if start_stage_idx is not None:
            run.current_stage_idx = start_stage_idx
        else:
            counter = 0
            run.current_stage_idx = len(stages) - 1
            for si, st in enumerate(stages):
                if counter + st['epochs'] > run.start_epoch:
                    run.current_stage_idx = si
                    break
                counter += st['epochs']
        run.current_stage_epoch = run.start_epoch - sum(stages[si]['epochs']
                                                        for si in range(run.current_stage_idx))
    stage = run.current_stage
    resumed = run.start_epoch > 0
    log.info(f"\nStage {run.current_stage_idx+1}/{len(stages)}: {stage['name']}"
             f"{' (resumed)' if resumed else ''}")
    # A combo_set stage (per-system only) has no manifest filter; per-system
    # pools are rebuilt every epoch.
    if stage.get('combo_set'):
        run.active_train_combos = list(run.train_combos)
    else:
        run.active_train_combos = apply_stage_combo_filter(
            run.train_combos, stage, run.curriculum_cfg, stage_idx=run.current_stage_idx,
            seed=run.seed, apply_transition_filter=(resumed and run.current_stage_idx > 0),
            combo_system_map=run.combo_system_map)
    if resumed and stage.get('policy_lr') is not None:
        _set_lr(run, stage['policy_lr'])
        log.info(f"  Policy LR restored to {stage['policy_lr']}")
    run.epoch_config = _stage_epoch_config(run, stage)


def _init_per_system(run: TrainingRun) -> None:
    """Per-system stages: each system advances through the stages on its own
    when its stage's ``advance`` criterion holds (mllf.cb.system_stages)."""
    run.per_system = run.curriculum_enabled and (
        run.deploy or bool(run.curriculum_cfg.get('per_system', False)))
    if (run.curriculum_enabled and not run.per_system
            and any(st.get('combo_set') for st in run.stages)):
        raise ValueError("a stage's combo_set needs curriculum.per_system: true")
    if not run.per_system:
        return
    run.epoch_config = run.config
    for c in run.train_combos:
        run.combos_by_system.setdefault(run.system_of(c), []).append(c)
    ignored = sorted({k for st in run.stages for k in st if k in _EPOCH_WIDE_STAGE_KEYS})
    if ignored:
        log.warning(f"  Warning: per-system stages ignore stage keys {ignored} "
                    "(they would apply to every system in the epoch)")
    state = (system_stages.load_state(run.stage_state_path) if run.start_epoch > 0 else None) or {}
    fresh = system_stages.init_state([s.get('name') for s in run.systems_cfg], run.stages)
    for name, st in fresh.items():
        state.setdefault(name, st)
    run.stage_state = state
    system_stages.save_state(run.stage_state_path, state)
    log.info("Per-system stages:")
    for name, st in state.items():
        where = ('done' if st['done'] else
                 f"{run.stages[st['stage']]['name']} (epoch {st['stage_epochs']} "
                 f"of {run.stages[st['stage']]['epochs']})")
        log.info(f"  {name}: {where}")


# per-system pools

def _system_stage(run: TrainingRun, sys_name: str) -> Optional[Dict]:
    st = run.stage_state.get(sys_name)
    if not st or st['done'] or st['stage'] is None:
        return None
    return run.stages[st['stage']]


def system_stage_pool(run: TrainingRun, sys_name: str) -> List[str]:
    """The system's combos for its current stage (stage filters, minus `systems`)."""
    st = run.stage_state.get(sys_name)
    if _system_stage(run, sys_name) is None:
        return []
    key = (sys_name, st['stage'])
    if key not in run._pool_cache:
        stage = dict(run.stages[st['stage']])
        stage.pop('systems', None)
        log.debug(f"  [{sys_name}] stage {stage['name']}:")
        if stage.get('combo_set'):
            # An explicit set (e.g. ALF's groups); the stage's site/sub
            # filters, if given, still apply to it.
            cands = run.combo_sets_by_system.get(sys_name, {}).get(stage['combo_set'], [])
            run._pool_cache[key] = (apply_stage_combo_filter(
                cands, stage, run.curriculum_cfg, stage_idx=st['stage'], seed=run.seed,
                apply_transition_filter=False) if 'min_sites' in stage else list(cands))
            log.debug(f"  combo set {stage['combo_set']}: {len(run._pool_cache[key])} combos")
        else:
            run._pool_cache[key] = apply_stage_combo_filter(
                run.combos_by_system.get(sys_name, []), stage, run.curriculum_cfg,
                stage_idx=st['stage'], seed=run.seed, apply_transition_filter=st['stage'] > 0)
    return run._pool_cache[key]


def system_group_pool(run: TrainingRun, sys_name: str) -> List[str]:
    """Groups (the current stage's ``group_set``) that join the pool once ready."""
    stage = _system_stage(run, sys_name)
    gset = stage.get('group_set') if stage else None
    return list(run.combo_sets_by_system.get(sys_name, {}).get(gset, [])) if gset else []


def _group_names(run: TrainingRun, sys_name: str) -> Set[str]:
    return {Path(g).name for g in system_group_pool(run, sys_name)}


def stage_names(run: TrainingRun) -> Dict[str, str]:
    """System -> the stage it is training in (per-pair stage counts and the ledger)."""
    if run.per_system:
        return {s: run.stages[st['stage']]['name'] for s, st in run.stage_state.items()
                if not st['done'] and st['stage'] is not None}
    if run.curriculum_enabled:
        return {s.get('name'): run.current_stage['name'] for s in run.systems_cfg}
    return {}


def retire_rule(plan: EpochPlan, stage: Dict, sys_name: str):
    """retire_requires_success as system_stages.retire_terms takes it: true /
    false for every pair, or 'reference' for the stage's reference pairs
    only (retire_min_successes good runs; reference_retire_after_epochs)."""
    v = stage.get('retire_requires_success', False)
    if v != 'reference':
        return bool(v)
    return {'keys': plan.ref_keys.get(sys_name, set()),
            'after_epochs': stage.get('reference_retire_after_epochs',
                                      stage.get('retire_after_epochs')),
            'min_successes': int(stage.get('retire_min_successes', 1))}


def export_final(run: TrainingRun, sys_name: str) -> None:
    """Production biases (and confidence labels) for a system that finished
    its last stage: one export per combo of its ``export_combo_set`` (e.g.
    each ALF group), else the whole system (or ``export_combo_dir``)."""
    if not run.curriculum_cfg.get('export_final_biases', True):
        return
    sys_cfg = next((s for s in run.systems_cfg if s.get('name') == sys_name), None)
    if sys_cfg is None:
        return
    cset = sys_cfg.get('export_combo_set')
    targets = ([(Path(c).name, c)
                for c in run.combo_sets_by_system.get(sys_name, {}).get(cset, [])]
               if cset else [(None, sys_cfg.get('export_combo_dir'))])
    for sub, cdir in targets:
        try:
            if sub is not None:
                cdir = ensure_combo_dir_exists(Path(cdir), sys_cfg, run.config,
                                               _lookup=run.explicit_combos)
            out = run.checkpoint_dir / 'final_biases' / sys_name
            path = export_final_biases(run.policy, sys_cfg, run.config, run.bandit_state,
                                       out / sub if sub else out, device=run.device,
                                       combo_dir=cdir)
            log.info(f"  {sys_name}: final biases written to {path}")
        except Exception as err:
            log.warning(f"  Warning: {sys_name}{'/' + sub if sub else ''}: export failed: {err}")
    # The labels read system_stages.json, so save it first (scouting exports
    # before the epoch's save).
    try:
        system_stages.save_state(run.stage_state_path, run.stage_state)
        n = label_final_biases(run.checkpoint_dir, run.config, [sys_name])
        if n:
            log.info(f"  {sys_name}: confidence labels for {n} group(s)")
    except Exception as err:
        log.warning(f"  Warning: {sys_name}: labelling failed: {err}")
    if run.archive and run.archive['when'] == 'system':
        archive_finished(run, [sys_name])


def archive_finished(run: TrainingRun, names: Optional[List[str]] = None) -> None:
    """Archive these systems' combo directories (default: every system not
    archived yet), per the config's archive: block."""
    todo = [n for n in (names if names is not None
                        else [s.get('name') for s in run.systems_cfg])
            if n not in run.archived]
    if not todo:
        return
    written = archive_systems(run.systems_cfg, run.archive['dir'], names=todo,
                              remove=run.archive['remove'])
    run.archived.update(Path(p).name[:-len('.tar.gz')] for p in written)


def _scout_tiers(run: TrainingRun, path) -> Dict[str, int]:
    """{pair key: tier} from a stage's scout_priority file (cached)."""
    if not path:
        return {}
    if path not in run._scout_tiers:
        run._scout_tiers[path] = json.loads(Path(path).read_text())['tiers']
    return run._scout_tiers[path]


def apply_scouting(run: TrainingRun, sn: str, epoch: int) -> Tuple[str, Dict]:
    """Stage key ``scouting: <dir>`` (training/scouting/scout_groups.py):
    before a system's first epoch in the stage, groups whose scouting run
    sampled every substituent have passed, pairs whose scouting run passed
    the full-run test are resolved (system_stages.apply_scouting), and the
    scouting runs enter the cost ledger at epoch - 1. A system whose group
    criterion already holds leaves the stage. Returns its log line and counts."""
    st = run.stage_state[sn]
    stg = run.stages[st['stage']]
    sdir = stg['scouting']
    if sdir not in run._scout_summaries:
        run._scout_summaries[sdir] = json.loads((Path(sdir) / 'scout_summary.json').read_text())
    summary = run._scout_summaries[sdir]
    gnames = [Path(g).name for g in system_group_pool(run, sn)]
    passed, resolved, runs = system_stages.apply_scouting(
        st, run.bandit_state if run.bandit_state is not None else {}, sn, stg['name'], summary,
        gnames)
    record_runs(run.ledger_path, [(sn, r, True) for r in runs], 'scout', epoch - 1,
                {sn: stg['name']})
    ns = sum(float(v.get('ns') or 0.0) for k, v in (summary.get('groups') or {}).items()
             if k.split('/', 1)[0] == sn)
    line = (f"  [SCOUT] {sn}: {len(passed)}/{len(gnames)} groups passed on their scouting run, "
            f"{len(resolved)} pairs resolved, {ns:.1f} ns")
    adv = stg.get('advance') or {}
    n_done = sum(1 for g in gnames if g in (st.get('groups_done') or {}))
    converged = False
    if (adv.get('type') == 'groups_done' and gnames
            and system_stages.criterion_met(adv, n_done, len(gnames))):
        nxt = system_stages.advance(st, run.stages, sn, epoch - 1, 'criterion',
                                    (n_done, len(gnames)))
        line += (f" -> {'done' if nxt is None else run.stages[nxt]['name']}"
                 " (criterion, by scouting)")
        converged = True
        if nxt is None:
            export_final(run, sn)
    log.info('%s', line)
    return line, {'groups': len(gnames), 'passed': len(passed), 'pairs': len(resolved), 'ns': ns,
                  'converged': converged, 'dir': sdir}


# 1. plan the epoch

def _skip_empty_stages(run: TrainingRun, epoch: int, held_out: Set[str]) -> None:
    """A stage with no combos for a system (filters leave none) is skipped."""
    for sn in sorted(run.stage_state):
        st = run.stage_state[sn]
        while not st['done'] and not [c for c in system_stage_pool(run, sn) if c not in held_out]:
            skipped = run.stages[st['stage']]['name']
            nxt = system_stages.advance(st, run.stages, sn, epoch, 'no_combos')
            log.info(f"  [STAGE] {sn}: no combos for stage {skipped}; skipped")
            if nxt is None:
                export_final(run, sn)


def _scout_systems(run: TrainingRun, epoch: int) -> None:
    lines, counts = [], []
    for sn in sorted(run.stage_state):
        st = run.stage_state[sn]
        if (st['done'] or not run.stages[st['stage']].get('scouting')
                or run.stages[st['stage']]['name'] in (st.get('scouting_applied') or [])):
            continue
        line, c = apply_scouting(run, sn, epoch)
        lines.append(line)
        counts.append(c)
    if not counts:
        return
    run.save_bandit_state()
    conv = sum(c['converged'] for c in counts)
    append_summary(run.summary_log_path, '\n'.join([
        f"\nScouting before epoch {epoch} ({', '.join(sorted({c['dir'] for c in counts}))}): "
        f"{sum(c['groups'] for c in counts)} groups scouted, "
        f"{sum(c['ns'] for c in counts):.1f} ns (cost ledger, epoch {epoch - 1}); "
        f"{sum(c['passed'] for c in counts)} groups passed on their scouting run; "
        f"{sum(c['pairs'] for c in counts)} pairs resolved; "
        f"{conv}/{len(counts)} systems converged by scouting"] + lines))


def plan_per_system(run: TrainingRun, epoch: int) -> Optional[EpochPlan]:
    """The epoch's pool under per-system stages; None when every system is done."""
    plan = EpochPlan(epoch)
    held_out = set(run.validation_combos)
    _skip_empty_stages(run, epoch, held_out)
    _scout_systems(run, epoch)
    system_stages.save_state(run.stage_state_path, run.stage_state)
    plan.active_systems = sorted(s for s, st in run.stage_state.items() if not st['done'])
    if not plan.active_systems:
        log.info(f"\nAll systems finished their last stage (epoch {epoch}); stopping.")
        return None
    run.active_train_combos = [c for s in plan.active_systems for c in system_stage_pool(run, s)
                               if c not in held_out]
    # Reference pairs: the stage's reference_set (e.g. each ALF group's
    # reference vs the others), else the stage's pool itself.
    for sn in plan.active_systems:
        rset = run.stages[run.stage_state[sn]['stage']].get('reference_set')
        src = (run.combo_sets_by_system.get(sn, {}).get(rset, []) if rset
               else system_stage_pool(run, sn))
        plan.ref_keys[sn] = {k for c in src for k in system_stages.combo_pair_keys(sn, c)}
    # A pairs_resolved / groups_done stage stops running resolved pairs and,
    # with retire_after_epochs, pairs that ran that many of the stage's
    # epochs without resolving (with retire_requires_success, only pairs
    # that had good runs).
    n_before = len(run.active_train_combos)
    keep = []
    for c in run.active_train_combos:
        sn = run.system_of(c)
        stg = run.stages[run.stage_state[sn]['stage']]
        adv = stg.get('advance') or {}
        if adv.get('type') in ('pairs_resolved', 'groups_done') and system_stages.combo_resolved(
                run.bandit_state, sn, c, stg['name'], adv, run.bandit.offset_prior_sd,
                retire_after=stg.get('retire_after_epochs'),
                retire_requires_success=retire_rule(plan, stg, sn)):
            continue
        keep.append(c)
    run.active_train_combos = keep
    if len(keep) < n_before:
        log.info(f"  Resolved or retired pairs left the pool: {n_before - len(keep)} combo(s)")
    # A stage with a group_set runs each group once group_ready_fraction of
    # its reference pairs are resolved (retired ones left out), until it
    # passes the group test; a group sent back to the pair phase
    # (group_max_failures) waits for its reopened pairs.
    for sn in plan.active_systems:
        stg = run.stages[run.stage_state[sn]['stage']]
        if not stg.get('group_set'):
            continue
        passed = run.stage_state[sn].get('groups_done') or {}
        holds = run.stage_state[sn].get('group_hold') or {}
        for g in system_group_pool(run, sn):
            if Path(g).name not in passed and system_stages.group_ready(
                    run.bandit_state, sn, g, plan.ref_keys[sn], stg['name'],
                    stg.get('advance') or {},
                    run.bandit.offset_prior_sd,
                    fraction=float(stg.get('group_ready_fraction', 1.0)),
                    retire_after=stg.get('retire_after_epochs'),
                    held=holds.get(Path(g).name),
                    retire_requires_success=retire_rule(plan, stg, sn),
                    min_resolved_per_site=int(stg.get('group_min_resolved_per_site', 0))):
                plan.ready_groups.append(g)
                holds.pop(Path(g).name, None)
    if plan.ready_groups:
        run.active_train_combos = run.active_train_combos + plan.ready_groups
        log.info(f"  Ready groups in the pool: {len(plan.ready_groups)}")
    return plan


def advance_curriculum(run: TrainingRun) -> None:
    """Global curriculum: move to the next stage once the current one has
    run its epochs (and, with progression.type reward, met its threshold)."""
    stages = run.stages
    if run.current_stage_epoch < stages[run.current_stage_idx]['epochs']:
        return
    progression = run.curriculum_cfg.get('progression', {})
    can_advance = progression.get('type', 'epoch') == 'epoch'
    if not can_advance:
        threshold = progression.get('reward_threshold', 0.0)
        recent = [s['avg_reward'] for s in run.all_stats[-5:] if 'avg_reward' in s]
        avg_recent = np.mean(recent) if recent else -999.0
        can_advance = avg_recent >= threshold
        if not can_advance:
            log.info(f"\nStage {run.current_stage_idx+1} kept: avg reward "
                     f"{avg_recent:.4f} < threshold {threshold}")
    if not (can_advance and run.current_stage_idx + 1 < len(stages)):
        return
    run.current_stage_idx += 1
    run.current_stage_epoch = 0
    stage = run.current_stage
    log.info(f"\nStage {run.current_stage_idx+1}/{len(stages)}: {stage['name']}")
    held_out = set(run.validation_combos)
    run.active_train_combos = [c for c in apply_stage_combo_filter(
        run.train_combos, stage, run.curriculum_cfg, stage_idx=run.current_stage_idx,
        seed=run.seed,
        apply_transition_filter=True, combo_system_map=run.combo_system_map) if c not in held_out]
    if stage.get('policy_lr') is not None:
        _set_lr(run, stage['policy_lr'])
        run.rl_scheduler = setup.make_scheduler(run.rl_optimizer)
        log.info(f"  Policy LR updated to {stage['policy_lr']}, scheduler reset")
    run.epoch_config = _stage_epoch_config(run, stage)


def print_epoch_header(run: TrainingRun, plan: EpochPlan) -> None:
    if run.per_system:
        counts = collections.Counter(run.stages[run.stage_state[s]['stage']]['name']
                                     for s in plan.active_systems)
        label = ("per-system stages: " + ', '.join(f"{n} {c}" for n, c in counts.items())
                 + f"; done {len(run.stage_state) - len(plan.active_systems)}")
    elif run.curriculum_enabled:
        label = (f"Stage {run.current_stage_idx+1}/{len(run.stages)} "
                 f"({run.current_stage['name']}) "
                 f"epoch {run.current_stage_epoch+1}/{run.current_stage['epochs']}")
    else:
        label = f"epoch {plan.epoch}"
    log.info(f"\nEPOCH {plan.epoch}/{run.num_epochs-1}  [{label}]")


# 2. select the epoch's combos

def _reference_quota(run: TrainingRun, plan: EpochPlan) -> None:
    """curriculum.combos_per_system: reference_pairs -- each system's share of
    the epoch is proportional to its number of reference pairs (at least
    min_combos_per_system); the budget grows if the floors exceed it."""
    if not (run.per_system and plan.max_combos is not None
            and run.curriculum_cfg.get('combos_per_system') == 'reference_pairs'):
        return
    nref = {s: len(plan.ref_keys.get(s, ())) for s in plan.active_systems}
    tot = sum(nref.values())
    if not tot:
        return
    qmin = int(run.curriculum_cfg.get('min_combos_per_system', 5))
    plan.system_quota = {s: max(qmin, int(round(plan.max_combos * n / tot)))
                         for s, n in nref.items()}
    plan.max_combos = max(plan.max_combos, sum(plan.system_quota.values()))
    log.debug('%s', "  Per-system shares (reference pairs): " + ', '.join(
        f"{s} {q}" for s, q in sorted(plan.system_quota.items())))


def _use_uncertainty_selection(run: TrainingRun) -> bool:
    """bandit.selection (uncertainty by default for hindsight, random for
    REINFORCE). Until the Bayesian heads have seen enough observations
    selection stays random -- except with shared_update: regression, which
    never adds posterior observations (n_obs stays 0) and keeps the
    pretraining-calibrated precision selection scores against."""
    use = run.bandit.selection_mode == 'uncertainty'
    if use and run.algorithm == 'hindsight' and not run.bandit.shared_regression:
        min_obs = run.bandit.min_observations_before_confident
        n_obs = [mlp.bayesian_head.n_obs.item() for mlp in run.policy.edge_mlp.mlps]
        if min(n_obs) < min_obs:
            log.info("  Bayesian heads: posterior not yet confident "
                     f"(min n_obs={min(n_obs):.0f} < {min_obs}); "
                     "falling back to random combo selection for this epoch")
            use = False
    return use


def _ensure_dir_for_scoring(lookup, combo_path, sys_cfg, cfg, verbose):
    return ensure_combo_dir_exists(combo_path, sys_cfg, cfg, verbose=verbose, _lookup=lookup)


def _pair_attention(run: TrainingRun, plan: EpochPlan, combo_dir, data):
    """Pair-state score of a combo; a stage's reference_pair_bonus ranks
    reference pairs ahead of others of similar uncertainty."""
    sn = run.combo_system_map.get(combo_dir, {}).get('name', 'unknown')
    att = pair_attention(run.bandit_state, sn, combo_node_labels(combo_dir), data.edge_index)
    if run.per_system and sn in run.stage_state and not run.stage_state[sn]['done']:
        bonus = float(run.stages[run.stage_state[sn]['stage']].get('reference_pair_bonus', 0.0))
        keys = system_stages.combo_pair_keys(sn, combo_dir)
        if bonus and keys and all(k in plan.ref_keys.get(sn, ()) for k in keys):
            att += bonus
    return att


def _by_system(run: TrainingRun, combos: List[str], default='unknown') -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    for c in combos:
        pools.setdefault(run.combo_system_map.get(c, {}).get('name', default), []).append(c)
    return pools


def select_epoch_combos(run: TrainingRun, plan: EpochPlan) -> None:
    """The epoch's subset of the active pool (which has already passed the
    curriculum / transition filters): uncertainty-driven (posterior variance
    for hindsight, action std for REINFORCE), stratified or uniform."""
    epoch, active, cfg = plan.epoch, run.active_train_combos, run.curriculum_cfg
    use_uncertainty = _use_uncertainty_selection(run)
    over_budget = plan.max_combos is not None and len(active) > plan.max_combos
    if over_budget and use_uncertainty:
        # Scoring a candidate costs a full Uni-Mol build, so only a reseeded,
        # stratified slice of the pool is scored each epoch.
        pool_size = (run.bandit.candidate_pool_size if run.bandit.candidate_pool_size is not None
                     else 5 * plan.max_combos)
        min_per_system = cfg.get('min_combos_per_system', 5)
        candidates = active
        if len(candidates) > pool_size:
            candidates = sample_combos_stratified(_by_system(run, candidates), n_total=pool_size,
                                                  min_per_system=min_per_system,
                                                  seed=run.seed + epoch + 20_000)
        # Most of a large pool was never created on disk: pre-scan metadata
        # once for this slice instead of a full scan per missing combo.
        lookup = {**run.explicit_combos, **build_combo_dir_lookup(candidates, run.combo_system_map,
                                                                  explicit=run.explicit_combos)}
        hindsight = run.algorithm == 'hindsight'
        plan.combos = select_combos_by_uncertainty(
            candidates, run.policy, run.combo_system_map, run.config,
            n_select=plan.max_combos, bandit_state=run.bandit_state,
            device=run.device, build_graph_fn=build_graph_and_data,
            ensure_dir_fn=functools.partial(_ensure_dir_for_scoring, lookup),
            min_per_system=min_per_system,
            score_fn=bayesian_variance_score if hindsight else action_std_score,
            attention_fn=functools.partial(_pair_attention, run, plan),
            attention_weight=float(run.bandit.selection_attention_weight),
            system_quota=plan.system_quota)
        log.info("  Epoch combo sample (uncertainty-driven, "
                 f"{'posterior variance' if hindsight else 'action std'}): "
                 f"{len(plan.combos)}/{len(active)} combos selected "
                 f"({len(candidates)} candidates scored, min_per_system={min_per_system})")
    elif over_budget and cfg.get('stratified_sampling', True):
        pools = _by_system(run, active)
        min_per_sys = cfg.get('min_combos_per_system', 5)
        plan.combos = sample_combos_stratified(pools, n_total=plan.max_combos,
                                               min_per_system=min_per_sys, seed=run.seed + epoch)
        log.info(f"  Epoch combo sample (stratified): {len(plan.combos)}/{len(active)} "
                 f"combos selected (min_per_system={min_per_sys}, seed={run.seed + epoch})")
        alloc = ', '.join(f"{s}:{sum(1 for c in plan.combos if run.system_of(c) == s)}"
                          for s in sorted(pools))
        log.debug(f"  Allocation per system: {alloc}")
    elif over_budget:
        rng = np.random.RandomState(run.seed + epoch)
        plan.combos = list(rng.choice(active, size=plan.max_combos, replace=False))
        log.info(f"  Epoch combo sample (uniform): {plan.max_combos}/{len(active)} "
                 f"combos selected (seed={run.seed + epoch})")
    else:
        plan.combos = active


def _combo_system(run: TrainingRun, c) -> str:
    return run.combo_system_map.get(c, {}).get('name') or Path(c).parent.name


def add_revisits(run: TrainingRun, plan: EpochPlan) -> None:
    """bandit.revisit: rerun in-progress pairs until their exploration is
    satisfactory, so per-pair corrections get consecutive visits; they take
    up to ``fraction`` of each system's minimum allocation."""
    rv = run.bandit.revisit
    if not (rv.get('enabled') and plan.update_policy and run.algorithm == 'hindsight'
            and plan.max_combos is not None):
        return
    active = run.active_train_combos
    rv_min = int(run.curriculum_cfg.get('min_combos_per_system', 5))
    sys_cfg = {s.get('name'): s for s in run.systems_cfg}
    systems = {run.system_of(c) for c in active}
    active_keys = {(run.system_of(c), Path(c).name) for c in active}
    held_keys = {(run.system_of(c), Path(c).name) for c in run.validation_combos}

    def allowed(c):
        k = (_combo_system(run, c), Path(c).name)
        return (k in active_keys and k not in held_keys and Path(c).is_dir()
                and not run.bandit_state.get(c, {}).get('deferred', False))

    quota = {s: int(round(float(rv.get('fraction', 0.5)) * rv_min)) for s in systems if s}
    revisits, counts = select_revisits(run.bandit_state, quota, allowed,
                                       successes_needed=int(rv.get('successes', 2)),
                                       max_visits=int(rv.get('max_visits', 8)))
    for c in revisits:   # out_dir paths (also after a restart)
        if c not in run.combo_system_map and Path(c).parent.name in sys_cfg:
            run.combo_system_map[c] = sys_cfg[Path(c).parent.name]
    plan.combos = merge_revisits(revisits, plan.combos, functools.partial(_combo_system, run),
                                 n_total=plan.max_combos, min_per_system=rv_min)
    msg = (f"  Revisits: {len(revisits)} in-progress pair(s) rerun "
           f"(pool: {counts['active']} active, {counts['satisfied']} satisfied, "
           f"{counts['exhausted']} exhausted); {len(plan.combos)} combos this epoch")
    log.info('%s', msg)
    plan.detail_lines = [msg]


def add_priorities(run: TrainingRun, plan: EpochPlan) -> None:
    """Per-system stages: ready groups, then pairs awaiting a full-length
    confirming run, then (reference_priority: strict) unresolved reference
    pairs and their reversed partners, then pairs not yet run in the stage
    (up to min_combos_per_system), then the selection."""
    if not (run.per_system and plan.max_combos is not None):
        return
    pmin = int(run.curriculum_cfg.get('min_combos_per_system', 5))
    priority, n_conf, n_new, n_ref, n_partner = list(plan.ready_groups), 0, 0, 0, 0
    rank = {(run.system_of(c), Path(c).name): n for n, c in enumerate(plan.combos)}
    store = (run.bandit_state or {}).get('_pair_offsets', {})
    for sn in plan.active_systems:
        stg = run.stages[run.stage_state[sn]['stage']]
        short_stage = bool(stg.get('run_steps'))
        first_new = bool(stg.get('prioritize_unvisited', True))
        strict = stg.get('reference_priority') == 'strict'
        # strict: the system's share goes first to its unresolved reference
        # pairs (and pairs a failed group sent back): not yet run first, then
        # in selection rank; a pair that got stuck or stranded (run_both) also
        # runs reversed, lambda starting on its other substituent.
        want = set(plan.ref_keys.get(sn, ()))
        for ks in (run.stage_state[sn].get('group_hold') or {}).values():
            want |= set(ks)
        # scout_priority: the lowest tier (likely hardest) goes first.
        tiers = _scout_tiers(run, stg.get('scout_priority') or (
            str(Path(stg['scouting']) / 'pair_priority.json') if stg.get('scouting') else None))
        refs, new = [], []
        for c in run.active_train_combos:
            if run.system_of(c) != sn or c in plan.ready_groups:
                continue
            awaiting, unvisited = system_stages.combo_pair_flags(run.bandit_state, sn, c,
                                                                 stg['name'])
            if strict:
                keys = system_stages.combo_pair_keys(sn, c)
                if keys and all(k in want for k in keys):
                    refs.append((0 if unvisited else 1,
                                 system_stages.combo_scout_tier(tiers, sn, c),
                                 rank.get((sn, Path(c).name), len(rank)), c, keys))
                    continue
            if awaiting and short_stage:     # confirmations exist only with short runs
                priority.append(c)
                n_conf += 1
            elif unvisited and first_new:
                new.append(c)
        if tiers:
            new.sort(key=lambda c: system_stages.combo_scout_tier(tiers, sn, c))
        if strict:
            share = (plan.system_quota or {}).get(sn, pmin)
            taken = 0
            for _, _, _, c, keys in sorted(refs, key=lambda t: t[:3]):
                if taken >= share:
                    break
                priority.append(c)
                taken += 1
                n_ref += 1
                if taken < share and len(keys) == 1 and (store.get(keys[0]) or {}).get('run_both'):
                    rev = reversed_pair(c, run.explicit_combos)
                    if rev:
                        run.combo_system_map[rev] = run.combo_system_map[c]
                        priority.append(rev)
                        taken += 1
                        n_partner += 1
        priority += new[:pmin]
        n_new += min(len(new), pmin)
    if not priority:
        return
    plan.combos = merge_revisits(priority, plan.combos, functools.partial(_combo_system, run),
                                 n_total=plan.max_combos, min_per_system=pmin)
    msg = (f"  Priority: {len(plan.ready_groups)} ready group(s), "
           + (f"{n_ref} reference pair(s) + {n_partner} reversed partner(s), " if n_ref else '')
           + f"{n_conf} pair(s) awaiting a full-length confirming run, "
           f"{n_new} not yet run; {len(plan.combos)} combos this epoch")
    log.info('%s', msg)
    plan.detail_lines.append(msg)


def run_steps(run: TrainingRun, plan: EpochPlan, sys_name: str, combo_name: str):
    """(equil, prod) MD steps for a combo's run this epoch: the stage's
    run_steps (short exploratory runs) or, for bracketed pairs,
    bracketed_run_steps; None (the template's full length) for group runs
    and pairs awaiting their full-length confirming run."""
    stage = _system_stage(run, sys_name) if run.per_system else None
    if stage is None:
        return None
    if combo_name in _group_names(run, sys_name):
        return None                      # group runs are full length
    rs = stage.get('run_steps')
    if not rs:
        return None
    awaiting, _ = system_stages.combo_pair_flags(run.bandit_state, sys_name, combo_name,
                                                 stage['name'])
    if awaiting:
        return None
    # Bracketed pairs (b seen on both sides) get longer runs: is the failure
    # a crossing-time problem rather than a bias one? Sticky long runs end
    # once the pair is resolved or retired.
    brs = stage.get('bracketed_run_steps')
    if brs and system_stages.combo_bracketed(
            run.bandit_state, sys_name, combo_name,
            sticky=bool(stage.get('bracketed_runs_sticky', False)),
            settled=lambda k, e: system_stages.pair_settled(
                e, stage['name'], stage.get('advance') or {}, run.bandit.offset_prior_sd,
                *system_stages.retire_terms(k, stage.get('retire_after_epochs'),
                                            retire_rule(plan, stage, sys_name)))):
        return int(brs['equil']), int(brs['prod'])
    return int(rs['equil']), int(rs['prod'])


def reserve_pool(run: TrainingRun, plan: EpochPlan) -> List[str]:
    """Combos that replace permanently failed ones on the fly."""
    size = run.config.get('max_failed_combos_per_epoch', 50)
    in_epoch = set(plan.combos)
    cands = [c for c in run.active_train_combos if c not in in_epoch]
    if not cands or size <= 0:
        return []
    rng = np.random.RandomState(run.seed + plan.epoch + 10_000)
    return list(rng.choice(cands, size=min(size, len(cands)), replace=False))


# 4. group runs and per-system stages

def _block_label(labels, block) -> str:
    n = int(block) - 2
    return f"s{labels[n][0]}:{labels[n][1]}" if 0 <= n < len(labels) else f"block{block}"


def _log_holdoff(run: TrainingRun, rec: EpochRecord, sys_name: str, st: Dict, cdir: Path, rd,
                 saved_data,
                 epoch: int) -> None:
    """bandit.group_holdoff: a group run whose dominant substituent's
    endpoint is held off by several partners at once (no pair profile sees
    it). Detected and logged."""
    held = system_stages.held_off_sites(saved_data, run.bandit.group_holdoff)
    if not held:
        return
    labels = combo_node_labels(cdir) or []
    recs = st.setdefault('group_holdoff', {}).setdefault(cdir.name, [])
    for h in held:
        recs.append({'epoch': epoch, 'run': Path(rd).name, 'sub': _block_label(labels, h['dom']),
                     'dom_frac': round(h['dom_frac'], 3), 'at_frac': round(h['at_frac'], 4),
                     'mean_dom': round(h['mean_dom'], 3),
                     'partners': {_block_label(labels, b): round(v, 3)
                                  for b, v in h['partners'].items()}})
        line = (f"  [HOLDOFF] {sys_name}/{cdir.name}: {_block_label(labels, h['dom'])} dominates "
                f"{h['dom_frac']:.0%} of frames at mean lambda {h['mean_dom']:.2f}, "
                f"endpoint held {h['at_frac']:.1%}; {len(h['partners'])} partners")
        log.info('%s', line)
        rec.stage_lines.append(line)
    del recs[:-20]
    rec.holdoff.append(f"{sys_name}/{cdir.name.split('_site')[0]} "
                       f"({', '.join(_block_label(labels, h['dom']) for h in held)})")


def _log_group_correction(run: TrainingRun, rec: EpochRecord, sys_name: str, cdir: Path, rd):
    """What a group run's correction (applied in train_epoch) did; returns
    its record, or None."""
    gc = group_correction.run_record(run.bandit_state, sys_name, cdir.name, rd)
    if not gc:
        return None
    what = ', '.join(f"{f['kind']} s{f['site']}" + (f" ({f['dom']})" if f.get('dom') else '')
                     + f" ratio {gc['ratio'].get(str(f['site']), 1.0):.2f}"
                     for f in gc.get('flagged') or [])
    what += ''.join(f", s{s} overshot: stepped back" for s in gc.get('guarded') or [])
    line = (f"  [GROUPFIX] {sys_name}/{cdir.name}: {what.lstrip(', ')}"
            + ("" if gc.get('stepped') else " (at the floor)"))
    log.info('%s', line)
    rec.stage_lines.append(line)
    rec.groupfix.append(f"{sys_name}/{cdir.name.split('_site')[0]} ({what.lstrip(', ')})")
    return gc


def _reopen_failed_group(run: TrainingRun, rec: EpochRecord, sys_name: str, st: Dict, stage: Dict,
                         cdir: Path, saved_data) -> None:
    """group_max_failures: a group failing that many runs goes back to the
    pair phase -- its pairs with a substituent its failed runs left empty are
    reopened and it waits for them."""
    gmax = stage.get('group_max_failures')
    fails = st.setdefault('group_failures', {})
    fails[cdir.name] = int(fails.get(cdir.name, 0)) + 1
    fpops = st.setdefault('group_failed_pops', {}).setdefault(cdir.name, [])
    fpops.append([float(v) for v in (saved_data.get('populations') or [])])
    if gmax is None or fails[cdir.name] < int(gmax):
        return
    keys = system_stages.group_failed_pairs(sys_name, cdir, [{'populations': p} for p in fpops])
    system_stages.reopen_pairs(run.bandit_state, keys, stage['name'])
    st.setdefault('group_hold', {})[cdir.name] = keys
    fails[cdir.name] = 0
    st['group_failed_pops'].pop(cdir.name, None)
    # runs from before the pairs are re-corrected are not pooled with later ones
    (st.get('group_window') or {}).pop(cdir.name, None)
    reopened = st.setdefault('groups_reopened', {})
    reopened[cdir.name] = int(reopened.get(cdir.name, 0)) + 1
    line = (f"  [GROUP] {sys_name}/{cdir.name}: failed {gmax} runs; back to the "
            f"pair phase ({len(keys)} pair(s) reopened)")
    log.info('%s', line)
    rec.stage_lines.append(line)
    rec.reopened.append((f"{sys_name}/{cdir.name.split('_site')[0]}", len(keys)))
    run.save_bandit_state()


def record_group_runs(run: TrainingRun, stats: Dict, epoch: int, rec: EpochRecord) -> None:
    """Record each combos_balanced / groups_done stage's latest runs: the
    balance test (pooled over group_pass_window runs), passes, hold-offs,
    group corrections and failures."""
    for sys_name, rd, ok in stats.get('run_dirs', []):
        st = run.stage_state.get(sys_name)
        if not st or st['done'] or not ok:
            continue
        stage = run.stages[st['stage']]
        adv = stage.get('advance') or {}
        groups = adv.get('type') == 'groups_done'
        if adv.get('type') not in ('combos_balanced', 'groups_done'):
            continue
        if groups and Path(rd).parent.name not in _group_names(run, sys_name):
            continue
        try:
            saved_data = torch.load(Path(rd) / 'epoch_results.pt', weights_only=False)
        except Exception:
            continue
        cdir = Path(rd).parent
        nsubs = [len(subs) for _, subs in system_stages.combo_sites(cdir)]
        if run.bandit.group_holdoff is not None and groups:
            _log_holdoff(run, rec, sys_name, st, cdir, rd, saved_data, epoch)
        # A run whose group correction stepped is not counted toward group_max_failures.
        gc = _log_group_correction(run, rec, sys_name, cdir, rd) if groups else None
        export_rd = str(rd)
        window = int(stage.get('group_pass_window', 1))
        min_ratio = float(adv.get('min_population_ratio', 0.5))
        min_trans = int(adv.get('min_site_transitions', 10))
        if groups and window > 1:
            # Pool the group's last runs: every substituent sampled in one of
            # them passes; the run that sampled most is the one exported.
            hist = st.setdefault('group_window', {}).setdefault(cdir.name, [])
            tr = saved_data.get('transitions') or []
            trans = list(tr.values()) if isinstance(tr, dict) else tr
            hist.append({'run': str(rd),
                         'populations': [float(v) for v in (saved_data.get('populations') or [])],
                         'transitions': [float(v) for v in trans]})
            del hist[:-window]
            passes, export_rd = system_stages.pooled_group_test(hist, nsubs, min_ratio, min_trans)
        else:
            passes = system_stages.site_balanced(saved_data, nsubs, min_ratio, min_trans)
        st['combos'][cdir.name] = passes
        if passes and groups:
            if cdir.name not in (st.get('groups_done') or {}):
                rec.passed.append(f"{sys_name}/{cdir.name.split('_site')[0]}")
            st.setdefault('groups_done', {}).setdefault(cdir.name, export_rd)
        elif (groups and cdir.name not in (st.get('groups_done') or {})
              and not (gc and gc.get('stepped'))):
            _reopen_failed_group(run, rec, sys_name, st, stage, cdir, saved_data)


def _stage_progress(run: TrainingRun, plan: EpochPlan, sys_name: str, st: Dict, stage: Dict,
                    kind: str,
                    held_out: Set[str]):
    """(progress, reference pairs resolved, retired, reference keys) of a
    system's stage criterion."""
    adv = stage.get('advance') or {}
    prog = n_res = n_ret = ref = None
    if kind == 'pairs_resolved':
        keys = {k for c in system_stage_pool(run, sys_name) if c not in held_out
                for k in system_stages.combo_pair_keys(sys_name, c)}
        prog = system_stages.pairs_progress(run.bandit_state, keys, stage['name'], adv,
                                            run.bandit.offset_prior_sd)
    elif kind == 'groups_done':
        names = [Path(g).name for g in system_group_pool(run, sys_name)]
        prog = (sum(1 for n in names if n in (st.get('groups_done') or {})), len(names))
        # reference pairs resolved (what group readiness depends on)
        ref = plan.ref_keys.get(sys_name, set())
        store = (run.bandit_state or {}).get('_pair_offsets', {})
        res = {k for k in ref if system_stages.pair_resolved(
            store.get(k), stage['name'], int(adv.get('successes', 2)), adv.get('max_offset_sd'),
            run.bandit.offset_prior_sd)}
        n_res = len(res)
        n_ret = sum(1 for k in ref if k not in res and system_stages.pair_retired(
            store.get(k), stage['name'], *system_stages.retire_terms(
                k, stage.get('retire_after_epochs'), retire_rule(plan, stage, sys_name))))
    elif kind == 'combos_balanced':
        prog = system_stages.combos_progress(st, adv, [Path(c).name
                                                       for c in system_stage_pool(run, sys_name)
                                                       if c not in held_out])
    return prog, n_res, n_ret, ref


def advance_system_stages(run: TrainingRun, plan: EpochPlan, epoch: int, rec: EpochRecord) -> None:
    """Advance systems whose stage criterion holds (or that used up the
    stage's epochs); a system past its last stage is done and exported."""
    held_out = set(run.validation_combos)
    for sys_name in plan.active_systems:
        st = run.stage_state[sys_name]
        stage = run.stages[st['stage']]
        adv = stage.get('advance') or {}
        kind = adv.get('type', 'epochs')
        st['stage_epochs'] += 1
        rec.ran[sys_name] = (st['stage_epochs'], int(stage['epochs']))
        prog, n_res, n_ret, ref = _stage_progress(run, plan, sys_name, st, stage, kind, held_out)
        met = (prog is not None and st['stage_epochs'] >= int(adv.get('min_epochs', 1))
               and system_stages.criterion_met(adv, *prog))
        what = {'pairs_resolved': 'pairs resolved',
                'groups_done': 'groups passed'}.get(kind, 'combos balanced')
        line = (f"  [STAGE] {sys_name}: {stage['name']} epoch "
                f"{st['stage_epochs']}/{stage['epochs']}"
                + (f", {prog[0]}/{prog[1]} {what}" if prog else '')
                + (f" ({n_res}/{len(ref)} reference pairs resolved"
                   + (f", {n_ret} retired" if stage.get('retire_after_epochs') is not None else '')
                   + ")" if kind == 'groups_done' else ''))
        if met or st['stage_epochs'] >= int(stage['epochs']):
            reason = 'criterion' if met else 'max_epochs' if prog is not None else 'epochs'
            nxt = system_stages.advance(st, run.stages, sys_name, epoch, reason, prog)
            line += f" -> {'done' if nxt is None else run.stages[nxt]['name']} ({reason})"
            rec.changes.append({'system': sys_name, 'stage': stage['name'],
                                'next': None if nxt is None else run.stages[nxt]['name'],
                                'reason': reason})
            rec.ended.add(sys_name)
            if nxt is None:
                rec.to_export.append(sys_name)
        log.info('%s', line)
        rec.stage_lines.append(line)
        rec.rows.append({
            'system': sys_name, 'stage': stage['name'],
            'stage_epoch': rec.ran[sys_name][0], 'stage_max': rec.ran[sys_name][1],
            'progress': ((prog[0], prog[1], epoch_log.PROGRESS_UNITS.get(kind, 'combos'))
                         if prog else None),
            'ref': (n_res, len(ref)) if ref is not None else None,
            'retired': n_ret if stage.get('retire_after_epochs') is not None else None})
    system_stages.save_state(run.stage_state_path, run.stage_state)


# 5. validation

def _epoch_validation_combos(run: TrainingRun, plan: EpochPlan, rec: EpochRecord) -> List[str]:
    """The held-out combos validated this epoch: those of the systems the
    current stage trains (eval_schedule: stage -- each system at the start,
    middle and end of each of its stages, and when it meets a criterion)."""
    stage_schedule = run.per_system and run.config.get('eval_schedule') == 'stage'
    if stage_schedule:
        systems = {s for s, (e, n) in rec.ran.items()
                   if e in (1, (n + 1) // 2, n) or s in rec.ended}
        if not systems:
            return []
    elif run.per_system:
        systems = set(plan.active_systems)
    else:
        systems = (set(run.current_stage['systems'])
                   if run.curriculum_enabled and run.current_stage.get('systems') else None)
    if not systems:
        return run.validation_combos
    return [c for c in run.validation_combos if run.system_of(c) in systems]


def _apply_gate(run: TrainingRun, epoch: int, val_combos: List[str], val_stats: Dict,
                rec: EpochRecord) -> None:
    """The validation gate: a pass scoring worse than the last accepted one
    by more than gate_margin restores the accepted model (per-pair offsets
    are kept); the best model is saved as best_model.pt."""
    g = run.gate
    key = frozenset(val_combos)
    if g.key is not None and key != g.key:
        # e.g. a new stage adds a system: scores aren't comparable
        log.info("  [VALIDATION GATE] held-out set changed (new stage): gate and best model reset")
        g.accepted_state = g.accepted_score = None
        g.best_state = g.best_score = g.best_epoch = None
    g.key = key
    score = int(val_stats.get('pair_success', {}).get('both', 0))
    margin = int(run.bandit.regression.get('gate_margin', 2))
    if g.accepted_score is not None and score < g.accepted_score - margin:
        run.policy.load_state_dict(g.accepted_state)
        msg = (f"  [VALIDATION GATE] held-out success {score} < accepted "
               f"{g.accepted_score} - {margin}: restored the last accepted model")
        rec.gate = f"below accepted {g.accepted_score} - {margin}: model restored"
    else:
        g.accepted_state = copy.deepcopy(run.policy.state_dict())
        g.accepted_score = score if g.accepted_score is None else max(g.accepted_score, score)
        msg = (f"  [VALIDATION GATE] held-out success {score}: model accepted "
               f"(reference {g.accepted_score})")
        rec.gate = "model accepted"
    g.last_score = score
    if g.best_score is None or score > g.best_score:
        g.best_state = copy.deepcopy(run.policy.state_dict())
        g.best_score, g.best_epoch = score, epoch
        torch.save({'policy_state_dict': g.best_state, 'epoch': epoch, 'val_success': score},
                   run.checkpoint_dir / 'best_model.pt')
        msg += f"; new best (epoch {epoch}) saved to best_model.pt"
        rec.gate += ", new best saved"
    log.info('%s', msg)
    rec.val_lines.append(msg)


def validate(run: TrainingRun, plan: EpochPlan, epoch: int, stats: Dict, rec: EpochRecord,
             stage_now: Dict[str, str]) -> None:
    """Deterministic pass over this epoch's held-out combos (every eval_freq
    epochs and after the last one), without updates. ``stage_now``: the
    stages the epoch trained in (for the cost ledger)."""
    val_combos = _epoch_validation_combos(run, plan, rec)
    stage_schedule = run.per_system and run.config.get('eval_schedule') == 'stage'
    if not val_combos or not (stage_schedule or epoch % run.config.get('eval_freq', 1) == 0
                              or epoch == run.num_epochs - 1):
        return
    log.info(f"\n  [VALIDATION] Evaluating on {len(val_combos)} held-out combos "
             "(deterministic)...")
    val_stats = train_epoch(run.policy, run.rl_optimizer, val_combos, run.combo_system_map, epoch,
                            run.epoch_config, run.device,
                            options=EpochOptions(update_policy=False, deterministic_actions=True,
                                                 explicit_combos=run.explicit_combos))
    record_runs(run.ledger_path, val_stats.get('run_dirs', []), 'validation', epoch, stage_now)
    nan = float('nan')
    val_reward, val_std = val_stats.get('avg_reward', nan), val_stats.get('reward_std', nan)
    train_reward, train_std = stats.get('avg_reward', nan), stats.get('reward_std', nan)
    val_loss, train_loss = val_stats.get('loss', nan), stats.get('loss', nan)
    lines = [f"  [VALIDATION] Train Reward: {train_reward:7.4f} +/- {train_std:6.4f}  |  "
             f"Val Reward: {val_reward:7.4f} +/- {val_std:6.4f}  "
             f"(delta: {val_reward - train_reward:+.4f})",
             f"  [VALIDATION] Train Loss: {train_loss:7.4f}  |  Val Loss: {val_loss:7.4f}"]
    pair_line = format_pair_success(val_stats.get('pair_success', {}), 'validation')
    if pair_line:
        lines.append(pair_line)
    for line in lines:
        log.info('%s', line)
    rec.val_lines.extend(lines)
    stats.update({'val_reward': val_reward, 'val_reward_std': val_std, 'val_loss': val_loss,
                  'val_n_combos': len(val_combos),
                  'val_pair_success': val_stats.get('pair_success', {})})
    rec.val_pair_success = stats['val_pair_success']
    if run.bandit.validation_gate:
        _apply_gate(run, epoch, val_combos, val_stats, rec)


# 6. log and checkpoint

def _cost_lines(run: TrainingRun, rec: EpochRecord) -> Dict:
    """Training ns per system so far (validation listed separately: it checks
    the shared model and is not part of converging a system)."""
    cost = summarize_ledger(load_ledger(run.ledger_path))
    if cost:
        rec.stage_lines.append("  [COST] training ns so far: " + ', '.join(
            f"{s} {c['train']['ns']:.1f}" for s, c in sorted(cost.items())))
        log.info('%s', rec.stage_lines[-1])
        vns = sum(c['validation']['ns'] for c in cost.values())
        if vns:
            rec.stage_lines.append(
                f"  [COST] validation ns so far (not in the totals above): {vns:.1f}")
            log.info('%s', rec.stage_lines[-1])
    return cost


def _epoch_log_block(run: TrainingRun, plan: EpochPlan, stats: Dict, rec: EpochRecord,
                     cost) -> str:
    """The epoch's epoch_summary.log block (mllf.cb.epoch_log)."""
    runs = {'groups': 0, 'reference': 0, 'reversed': 0, 'other': 0}
    for c in plan.combos:
        sn = _combo_system(run, c)
        if Path(c).name in _group_names(run, sn):
            runs['groups'] += 1
            continue
        want = set(plan.ref_keys.get(sn, ()))
        for ks in ((run.stage_state.get(sn) or {}).get('group_hold') or {}).values():
            want |= set(ks)
        keys = system_stages.combo_pair_keys(sn, c)
        if keys and all(k in want for k in keys):
            runs['reference'] += 1
            runs['reversed'] += bool(re.match(r'pair\d+r_', Path(c).name))
        else:
            runs['other'] += 1
    for r in rec.rows:
        r['ns'] = ((cost or {}).get(r['system']) or {}).get('train', {}).get('ns', 0.0)
    return epoch_log.format_epoch_block(
        plan.epoch, runs, failed=stats.get('permanently_failed', []),
        pair_short=stats.get('pair_success_short'), pair_full=stats.get('pair_success_full'),
        groups_run=runs['groups'], groups_passed=rec.passed, groups_reopened=rec.reopened,
        groups_holdoff=rec.holdoff, groups_corrected=rec.groupfix,
        validation=rec.val_pair_success, gate=rec.gate, regression=stats.get('regression'),
        hindsight=stats.get('hindsight'), systems=rec.rows, stage_changes=rec.changes,
        ns_train=sum(c['train']['ns'] for c in (cost or {}).values()) if cost else None,
        ns_validation=sum(c['validation']['ns'] for c in (cost or {}).values()) if cost else None)


def _summary_lines(run: TrainingRun, epoch: int, stats: Dict) -> List[str]:
    lines = [f"\nEpoch {epoch} Summary:",
             f"  Policy Loss: {stats['loss']:.4f}",
             f"  Avg Reward: {stats['avg_reward']:.4f} +/- {stats.get('reward_std', 0.0):.4f}"]
    if 'val_reward' in stats and not np.isnan(stats['val_reward']):
        lines.append(f"  Val Reward: {stats['val_reward']:.4f}"
                     f" +/- {stats.get('val_reward_std', 0.0):.4f} "
                     f"({stats.get('val_n_combos', len(run.validation_combos))} combos)")
    lines.append(f"  Success batch: {stats.get('n_success', 0)}")
    train_line = format_pair_success(stats.get('pair_success', {}), 'train batch')
    if train_line:
        lines.append(train_line)
    # Short exploratory and full-length runs, each judged by its own test.
    for key, label in (('pair_success_short', 'train short runs'),
                       ('pair_success_full', 'train full-length runs')):
        if (stats.get(key) or {}).get('total'):
            lines.append(format_pair_success(stats[key], label))
    if 'val_pair_success' in stats:
        val_line = format_pair_success(stats['val_pair_success'], 'validation')
        if val_line:
            lines.append(val_line)
    failed = stats.get('permanently_failed', [])
    if failed:
        lines.append(f"  Permanently failed ({len(failed)} combo(s)):")
        lines.extend(f"    {fc}" for fc in failed)
    updated = 'Yes' if stats.get('update_policy', True) else 'No'
    lines.append(f"  Policy updated: {'Yes (BC)' if stats.get('bc_mode') else updated}")
    return lines


def write_epoch_summary(run: TrainingRun, plan: EpochPlan, stats: Dict, rec: EpochRecord,
                        cost) -> None:
    """Print the epoch summary and append it to epoch_summary.log: the
    header first, then the epoch's detail lines (per-system stages: one
    concise block, the full detail stays in stdout)."""
    try:
        lines = _summary_lines(run, plan.epoch, stats)
        for line in lines:
            log.info('%s', line)
        sys.stdout.flush()
        try:
            if run.per_system:
                append_summary(run.summary_log_path,
                               '\n' + _epoch_log_block(run, plan, stats, rec, cost))
            else:
                append_summary(run.summary_log_path, '\n'.join(lines))
                details = (plan.detail_lines + stats.get('detail_lines', []) + rec.val_lines
                           + rec.stage_lines)
                if details:
                    append_summary(run.summary_log_path, '\n'.join(details))
        except Exception as err:
            log.warning(f"  Warning: Failed to write epoch summary to file: {err}")
            sys.stdout.flush()
    except Exception as err:
        log.error(f"\nERROR: Failed to print epoch summary: {err}")
        traceback.print_exc()
        sys.stdout.flush()


def step_scheduler(run: TrainingRun, stats: Dict) -> None:
    """ReduceLROnPlateau on the success count. rl_optimizer only steps in
    REINFORCE and BC; hindsight updates in closed form or with the
    regression's own optimizer."""
    if (stats.get('bc_mode') or not stats.get('update_policy', True)
            or run.algorithm == 'hindsight'):
        return
    prev = [pg['lr'] for pg in run.rl_optimizer.param_groups]
    run.rl_scheduler.step(stats.get('n_success', 0))
    new = [pg['lr'] for pg in run.rl_optimizer.param_groups]
    if new != prev:
        log.info(f"  ReduceLROnPlateau fired: LR {prev[0]:.2e} -> {new[0]:.2e}")


def _baseline_fields(run: TrainingRun) -> Dict:
    """Context-baseline state for a checkpoint (absent without one)."""
    if run.baseline_net is None:
        return {}
    return {'baseline_state_dict': run.baseline_net.state_dict(),
            'baseline_optimizer_state_dict': run.baseline_optimizer.state_dict()}


def save_checkpoint(run: TrainingRun, epoch: int, stats: Dict) -> None:
    if (not run.output_cfg.get('save_checkpoints', True)
            or epoch % run.output_cfg.get('checkpoint_freq', 5) != 0):
        return
    path = run.checkpoint_dir / f'checkpoint_{epoch:03d}.pt'
    torch.save({'epoch': epoch, 'policy_state_dict': run.policy.state_dict(),
                'rl_optimizer_state_dict': run.rl_optimizer.state_dict(),
                'current_stage_idx': run.current_stage_idx, 'stats': stats,
                **_baseline_fields(run)}, path)
    log.info(f"  Saved checkpoint: {path}")


def finish(run: TrainingRun) -> None:
    """End on the best validated model when the last validation scored
    lower, and save final_model.pt."""
    log.info("\nTraining complete")
    g = run.gate
    final_from = 'last epoch'
    if g.best_state is not None and g.last_score is not None and g.last_score < g.best_score:
        run.policy.load_state_dict(g.best_state)
        final_from = (f'best validated model (epoch {g.best_epoch}, '
                      f'held-out success {g.best_score})')
        log.info(f"Restored the best validated model: epoch {g.best_epoch}, held-out success "
                 f"{g.best_score} (last validation {g.last_score})")
    final_path = run.checkpoint_dir / 'final_model.pt'
    torch.save({'final_from': final_from, 'policy_state_dict': run.policy.state_dict(),
                'rl_optimizer_state_dict': run.rl_optimizer.state_dict(),
                'all_stats': run.all_stats,
                **_baseline_fields(run)}, final_path)
    log.info(f"Saved final model: {final_path}")
    if run.archive:
        archive_finished(run)


# the loop

def run_epoch(run: TrainingRun, epoch: int) -> bool:
    """Plan, train, record, validate and checkpoint one epoch. Returns False
    when every system has finished (per-system stages)."""
    if run.per_system:
        plan = plan_per_system(run, epoch)
        if plan is None:
            return False
    else:
        plan = EpochPlan(epoch)
        if run.curriculum_enabled:
            advance_curriculum(run)
    print_epoch_header(run, plan)
    plan.stage = run.current_stage if run.curriculum_enabled and not run.per_system else {}
    # Re-sample each epoch so max_train_combos_per_stage draws a fresh subset.
    plan.max_combos = plan.stage.get('max_train_combos',
                                     run.curriculum_cfg.get('max_train_combos_per_stage'))
    _reference_quota(run, plan)
    select_epoch_combos(run, plan)
    add_revisits(run, plan)
    add_priorities(run, plan)

    stage = plan.stage
    stats = train_epoch(
        run.policy, run.rl_optimizer, plan.combos, run.combo_system_map, epoch, run.epoch_config,
        run.device,
        options=EpochOptions(
            update_policy=plan.update_policy,
            bc_mode=stage.get('bc_mode', False),
            bc_source_epoch=stage.get('bc_source_epoch', None),
            replay_from_epoch=stage.get('replay_from_epoch', None),
            deterministic_actions=stage.get('deterministic_actions', False),
            bc_auxiliary_weight=stage.get('bc_auxiliary_weight', 0.0),
            ema_baseline=run.ema_baseline,
            reserve_pool=reserve_pool(run, plan),
            verbose=run.config.get('verbose', False),
            bandit_state=run.bandit_state,
            baseline_net=run.baseline_net, baseline_optimizer=run.baseline_optimizer,
            system_to_idx=run.system_to_idx,
            history_replay_cfg=run.history_replay_cfg,
            update_shared=run.update_shared,
            stage_of_system=stage_names(run),
            run_steps_fn=functools.partial(run_steps, run, plan) if run.per_system else None,
            stage_cfg_of_system=({s: run.stages[run.stage_state[s]['stage']]
                                  for s in plan.active_systems}
                                 if run.per_system else None),
            explicit_combos=run.explicit_combos,
            replay_cache=run.replay_cache))
    run.all_stats.append(stats)
    run.save_bandit_state()
    # Cost ledger (mllf.cb.cost_ledger): simulated ns per system and stage.
    stage_now = stage_names(run)
    record_runs(run.ledger_path, stats.get('run_dirs', []), 'train', epoch, stage_now)

    rec = EpochRecord()
    if run.per_system:
        record_group_runs(run, stats, epoch, rec)
        advance_system_stages(run, plan, epoch, rec)
    validate(run, plan, epoch, stats, rec, stage_now)
    for sys_name in rec.to_export:
        export_final(run, sys_name)
    cost = _cost_lines(run, rec)
    write_epoch_summary(run, plan, stats, rec, cost)
    step_scheduler(run, stats)
    if run.curriculum_enabled:
        run.current_stage_epoch += 1
    save_checkpoint(run, epoch, stats)
    return True


def run_training(config_path: str, verbose: Optional[bool] = None) -> None:
    """Run online policy training from a workflow YAML file (see
    :doc:`/docs/workflow`): build or load the combo manifests, initialize or
    resume the policy, run the configured (curriculum) epochs and save
    checkpoints. In deploy mode the model and output paths come from the
    ``deploy:`` block. ``verbose`` overrides the config's ``verbose`` key.
    Every message is also recorded in ``<output.base_dir>/training.log``."""
    config = setup.load_training_config(config_path)
    if verbose is not None:
        config['verbose'] = verbose
    log_file = Path(config.get('output', {}).get('base_dir', 'checkpoints')) / LOG_FILE
    with log_to_file(log_file), WarningSummary():
        log.info(f"Log: {log_file}")
        run = setup_run(config)
        for epoch in range(run.start_epoch, run.num_epochs):
            if not run_epoch(run, epoch):
                break
        finish(run)
