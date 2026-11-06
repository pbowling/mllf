"""Scouting runs: one run per group with the starting model's biases, before
a stage's pair phase, to rank the pairs that need the most work.

For each system of a workflow config and each group of the stage's
``group_set``, ``prepare``:

1. builds the group directory under ``<out>/simulations/<system>/`` (earlier
   simulation directories are not touched),
2. writes the model's mean prediction (``export_final_biases`` with no pair
   offsets) to ``<group>/run_001/variables.py``,
3. records every within-site pair's predicted b and c with their posterior
   predictive sd in ``<out>/uncertainty.json`` (Bayesian-head models),
4. submits the run.

``summarize`` (once the runs have finished) writes ``<out>/scout_summary.json``,
which a stage's ``scouting: <out>`` key applies before the system's first epoch
in the stage (``training_loop.apply_scouting``): each group's run, its
simulated ns and whether it sampled every substituent (such a group has passed
the group test), and the pairs whose scouting run passed the full-run test
(>= SUCCESS_MIN_TRANSITIONS crossings, minority fraction >= SUCCESS_MIN_FRACTION).
It also writes ``<out>/pair_priority.json``, the per-pair tiers that order the
pair queue (tier 0 runs first):

* 0: a substituent of the pair was never sampled, or 1-4 crossings
* 1: 5-9 crossings (and pairs the scouting runs did not cover)
* 2: both sampled without crossing, or 10+ crossings
"""
import copy
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from mllf.cb import system_stages as ss
from mllf.cb.cost_ledger import simulated_ns
from mllf.cb.export_biases import export_final_biases
from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.hindsight import (
    SUCCESS_MIN_FRACTION,
    SUCCESS_MIN_TRANSITIONS,
    combo_node_labels,
    node_transition_counts,
    pair_key,
)
from mllf.cb.slurm import submit_epoch_simulation
from mllf.cb.workflow_utils import set_msld_steps
from mllf.file_handling.combo_sets import build_combo_sets
from mllf.file_handling.manifests import ensure_combo_dir_exists


log = logging.getLogger(__name__)

RUN = 'run_001'
DIMS = {0: 'b', 1: 'c'}


def model_path(config: Dict) -> str:
    """The model a stage starts from: deploy.model_path, else pretrain.model_path."""
    return (config.get('deploy') or {}).get('model_path') or config['pretrain']['model_path']


def load_policy(config: Dict, device: str, path: Optional[str] = None):
    """The starting model, in eval mode."""
    from mllf.cb.training_setup import build_policy, check_pretrain_inputs
    path = path or model_path(config)
    check_pretrain_inputs(config, path)
    policy = build_policy(config, device, bayesian_default=True)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt.get('policy_state_dict') or ckpt['policy_state'])
    return policy.eval()


def scouting_stage(config: Dict, name: Optional[str] = None) -> Dict:
    """The stage scouting runs for: ``name``, else the first with a group_set."""
    stages = (config.get('curriculum') or {}).get('stages') or []
    for s in stages:
        if (name and s.get('name') == name) or (not name and s.get('group_set')):
            return s
    raise ValueError(f"no stage {name!r}" if name else "no curriculum stage has a group_set")


def stage_groups(config: Dict, sys_cfg: Dict, out: Path, stage: Dict,
                 registry: Optional[Dict] = None) -> Tuple[Dict, List[str], set]:
    """(system config with out_dir under <out>/simulations, group paths,
    reference pair keys). ``registry`` collects each combo's info, from which
    ``ensure_combo_dir_exists`` builds its directory."""
    sys_cfg = copy.deepcopy(sys_cfg)
    sys_cfg['out_dir'] = str(out / 'simulations' / sys_cfg['name']) + '/'
    sets, warns = build_combo_sets(sys_cfg, config.get('alf_root'), registry=registry)
    for w in warns:
        log.warning(f"  Warning: {sys_cfg['name']}: {w}")
    refs = {k for c in sets.get(stage.get('reference_set'), [])
            for k in ss.combo_pair_keys(sys_cfg['name'], c)}
    return sys_cfg, sets.get(stage['group_set'], []), refs


def scout_pairs(combo: Path, run_dir: Optional[Path] = None) -> Optional[Dict]:
    """{pair key: (bin, minority fraction, crossings)} from a scouting run, or
    None if it has not finished. Bins: crossing (both sampled, >= 1 crossing),
    sampled (both sampled, no crossing), one empty, both empty; crossings on a
    site of 3+ are the fewer of the two substituents' entries and exits
    (hindsight.node_transition_counts)."""
    from mllf.cli.collect_pretraining_data import parse_output_file
    from mllf.pretrain.rewards import _extract_highest_lambda_counts
    run_dir = run_dir or combo / RUN
    out = run_dir / 'output.out'
    if not out.exists():
        return None
    sr = parse_output_file(out, run_dir)
    if not sr or not sr.get('terminated_normally', True):
        return None
    pops = _extract_highest_lambda_counts(sr.get('populations', {}))
    saved = {'populations': pops, 'transition_pairs': sr.get('transition_pairs')}
    labels = combo_node_labels(combo)
    system = combo.parent.name
    res = {}
    for site, subs in ss.combo_sites(combo):
        nodes = [labels.index((site, s)) for s in subs]
        counts = node_transition_counts(saved, nodes) if len(nodes) > 2 else None
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                i, j = nodes[a], nodes[b]
                pi = pops[i] if i < len(pops) else 0
                pj = pops[j] if j < len(pops) else 0
                if counts is not None:
                    t = min(counts[i], counts[j])
                else:
                    key = f'{min(i, j) + 2}_{max(i, j) + 2}'
                    t = (saved['transition_pairs'] or {}).get(key) or 0
                if pi > 0 and pj > 0:
                    kind = 'crossing' if t >= 1 else 'sampled'
                else:
                    kind = 'one empty' if (pi > 0 or pj > 0) else 'both empty'
                frac = min(pi, pj) / (pi + pj) if pi + pj > 0 else 0.0
                res[pair_key(system, labels[i], labels[j])[0]] = (kind, frac, int(t))
    return res


def scout_tier(kind: str, frac: float, crossings: int) -> int:
    """Priority tier of a scouted pair (module docstring; 0 runs first)."""
    if kind in ('one empty', 'both empty') or (kind == 'crossing' and crossings < 5):
        return 0
    if kind == 'crossing' and crossings < 10:
        return 1
    return 2


def pair_uncertainty(policy, combo: Path, sys_cfg: Dict, config: Dict, device: str) -> Dict:
    """{pair key: {'b', 'c', 'b_sd', 'c_sd'}} for every within-site pair: the
    posterior predictive mean and sd in physical units (sd from the raw
    posterior +/-1 sd mapped through the output nonlinearity), b in the pair
    key's orientation. Empty for a model without Bayesian heads."""
    if not getattr(policy, 'use_bayesian_heads', False):
        return {}
    data, _ = build_graph_and_data(str(combo), sys_cfg, config, device)
    labels = combo_node_labels(combo)
    out = {}
    with torch.no_grad():
        z = policy.forward_features(data.x, data.edge_index, None)
        heads = {}
        for d in DIMS:
            m, v = policy.edge_mlp.mlps[d].bayesian_head.predict(z[:, d, :])
            sd = v.clamp_min(0).sqrt()
            spread = (policy._preact_to_action(m + sd, d)
                      - policy._preact_to_action(m - sd, d)).abs() / 2
            heads[d] = (policy._preact_to_action(m, d), spread)
    for e in range(data.edge_index.size(1)):
        i, j = int(data.edge_index[0, e]), int(data.edge_index[1, e])
        if i >= j or labels[i][0] != labels[j][0]:
            continue
        key, flipped = pair_key(sys_cfg['name'], labels[i], labels[j])
        b = float(heads[0][0][e])
        out[key] = {'b': -b if flipped else b, 'c': float(heads[1][0][e]),
                    'b_sd': float(heads[0][1][e]), 'c_sd': float(heads[1][1][e])}
    return out


def _systems(config: Dict, names: Optional[Iterable[str]]) -> List[Dict]:
    wanted = set(names) if names else None
    return [s for s in config.get('systems', []) if wanted is None or s.get('name') in wanted]


def prepare(config: Dict, out: Path, model: Optional[str] = None,
            systems: Optional[Iterable[str]] = None, stage_name: Optional[str] = None,
            run_steps: Optional[Tuple[int, int]] = None, submit: bool = True) -> Dict[str, str]:
    """Build, bias and (with ``submit``) submit one scouting run per group;
    returns {system/group: job id} of every submitted run (kept in
    <out>/jobs.json, so a repeated call submits only what is missing)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stage = scouting_stage(config, stage_name)
    policy = load_policy(config, device, model)
    log.info(f"Scouting stage {stage['name']} with model {model or model_path(config)}")
    unc_path, jobs_path = out / 'uncertainty.json', out / 'jobs.json'
    unc = json.loads(unc_path.read_text()) if unc_path.exists() else {}
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    for sys_cfg in _systems(config, systems):
        name = sys_cfg['name']
        registry: Dict[str, Dict] = {}
        sys_cfg, groups, refs = stage_groups(config, sys_cfg, out, stage, registry)
        for g in groups:
            combo = ensure_combo_dir_exists(Path(g), sys_cfg, config, _lookup=registry)
            if run_steps:
                set_msld_steps(combo, *run_steps)
            run_dir = combo / RUN
            tag = f'{name}/{combo.name}'
            if not (run_dir / 'variables.py').exists():
                export_final_biases(policy, sys_cfg, config, None, run_dir, device=device,
                                    combo_dir=combo)
            if tag not in unc:
                pairs = pair_uncertainty(policy, combo, sys_cfg, config, device)
                for k, v in pairs.items():
                    v['reference'] = k in refs
                unc[tag] = {'system': name, 'pairs': pairs}
                unc_path.write_text(json.dumps(unc, indent=1))
            if submit and tag not in jobs and not (run_dir / 'output.out').exists():
                jobs[tag] = submit_epoch_simulation(str(combo), str(run_dir), config,
                                                    sys_cfg.get('solvent_state', 'solv'))
                jobs_path.write_text(json.dumps(jobs, indent=1))
            log.info(f"  {tag}: {len(unc[tag]['pairs'])} pairs"
                     + (f", job {jobs[tag]}" if tag in jobs else ''))
    log.info(f"{len(unc)} group(s) prepared, {len(jobs)} submitted")
    return jobs


TERMINAL_STATES = ('BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
                   'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT')


def _queued(job_ids: Iterable[str]) -> set:
    sq = subprocess.run(['squeue', '--me', '--noheader', '--format=%i'],
                        capture_output=True, text=True)
    return {ln.strip() for ln in sq.stdout.splitlines() if ln.strip()} & set(job_ids)


def _ended(job_ids: Iterable[str]) -> Optional[set]:
    """The jobs sacct reports in a terminal state, or None without sacct."""
    try:
        sa = subprocess.run(['sacct', '-X', '-n', '-P', '-o', 'JobID,State',
                             '-j', ','.join(job_ids)], capture_output=True, text=True)
    except OSError:
        return None
    if sa.returncode != 0:
        return None
    ended = set()
    for line in sa.stdout.splitlines():
        job, _, state = line.partition('|')
        if state.split(' ')[0] in TERMINAL_STATES:
            ended.add(job.strip())
    return ended


def wait_for_jobs(job_ids: Iterable[str], poll_seconds: float = 60.0) -> None:
    """Return once every SLURM job has ended. A job counts as ended when it is
    out of the queue and, where sacct is available, sacct reports a terminal
    state (a requeued job can be briefly missing from squeue)."""
    pending = {str(j) for j in job_ids}
    while pending:
        queued = _queued(pending)
        ended = _ended(pending)
        pending = {j for j in pending
                   if j in queued or (ended is not None and j not in ended)}
        if pending:
            log.debug(f"waiting for {len(pending)} scouting job(s)")
            time.sleep(poll_seconds)


def summarize(config: Dict, out: Path, systems: Optional[Iterable[str]] = None,
              stage_name: Optional[str] = None) -> Dict:
    """Write scout_summary.json and pair_priority.json from the finished runs
    (module docstring); returns the summary."""
    out = Path(out)
    stage = scouting_stage(config, stage_name)
    tiers, counts, missing = {}, [0, 0, 0], []
    groups_out, pass_pairs = {}, set()
    for sys_cfg in _systems(config, systems):
        sys_cfg, groups, _ = stage_groups(config, sys_cfg, out, stage)
        for g in groups:
            combo = Path(sys_cfg['out_dir']) / Path(g).name
            scout = scout_pairs(combo)
            if scout is None:
                missing.append(f"{sys_cfg['name']}/{combo.name}")
                continue
            for k, v in scout.items():
                tiers[k] = min(tiers.get(k, 2), scout_tier(*v))
                if (v[0] == 'crossing' and v[2] >= SUCCESS_MIN_TRANSITIONS
                        and v[1] >= SUCCESS_MIN_FRACTION):
                    pass_pairs.add(k)
            run = combo / RUN
            groups_out[f"{sys_cfg['name']}/{combo.name}"] = {
                'run': str(run), 'ns': simulated_ns(run / 'output.out'),
                'all_sampled': bool(scout) and all(v[0] in ('crossing', 'sampled')
                                                   for v in scout.values())}
    for v in tiers.values():
        counts[v] += 1
    (out / 'pair_priority.json').write_text(json.dumps(
        {'stage': stage['name'], 'tiers': tiers, 'missing_runs': missing}, indent=1))
    summary = {'stage': stage['name'], 'groups': groups_out,
               'pass_pairs': sorted(pass_pairs), 'missing_runs': missing}
    (out / 'scout_summary.json').write_text(json.dumps(summary, indent=1))
    n_all = sum(g['all_sampled'] for g in groups_out.values())
    ns = sum(g['ns'] or 0 for g in groups_out.values())
    log.info(f"Scouting: {len(groups_out)} group(s), {ns:.1f} ns, {n_all} sampled every "
             f"substituent; {len(pass_pairs)} pair(s) passed the full-run test; "
             f"{len(tiers)} pairs in tiers 0/1/2: {counts[0]}/{counts[1]}/{counts[2]}"
             + (f"; {len(missing)} run(s) missing or unfinished" if missing else ''))
    return summary
