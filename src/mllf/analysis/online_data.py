"""Data of an online-training run for ``mllf.analysis.online``.

Everything is read from what the run wrote to ``output.base_dir``:
``epoch_summary.log`` (per-epoch success, per-system stages),
``cost_ledger.json`` (every run with its system, stage, kind, epoch and
simulated ns), ``system_stages.json`` (each system's stage history and passed
groups) and each run's ``epoch_results.pt`` (populations and transitions).
ALF's runs of the same groups come from the pretraining datasets
(``--pretraining-root``) and are cached, since those datasets do not change.

Run outcomes use the tests of online training. A pair run succeeds with both
substituents sampled, a minority fraction >= 0.2 and >= 10 crossings
(full-length runs) or the stage's short-run test (``explore_min_transitions``
/ ``explore_min_fraction``, default 2 and 0.2) for runs under 0.5 ns. A group
run passes when every site holds each substituent at >= ``min_population_ratio``
/ n of its population with >= ``min_site_transitions`` transitions (the
stage's ``advance``, default 0.5 and 10). With the stage's
``group_pass_window`` above 1, training pools a group's latest runs: a run
that fails alone but passes together with the group's previous runs is
"passed with previous run". A stage with ``advance.production`` instead
judges each group run alone by the production test
(``system_stages.production_balanced``: every substituent's time at its
endpoint and its visits per ns); a run that meets the balance test but not
the production test is "short of production test".
"""
import concurrent.futures
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


log = logging.getLogger(__name__)

PAIR_OUTCOMES = ['success', 'unbalanced', 'few crossings', 'one-sided', 'neither sampled']
GROUP_OUTCOMES = ['passed', 'passed with previous run', 'short of production test', 'uneven',
                  'few transitions',
                  'unsampled substituent']
SHORT_RUN_NS = 0.5
SCAN_VERSION = 4          # bump when scan_runs' rows change (rebuilds caches)
FULL_TEST = (10, 0.2)


@dataclass
class OnlineData:
    config: Dict
    run_dir: Path
    stages: List[Dict]
    env: Dict[str, str]                      # system -> environment
    epochs: 'object'                         # DataFrame, one row per epoch
    stage_rows: 'object'                     # DataFrame, one row per (epoch, system)
    ledger: Dict[str, Dict]
    stage_state: Dict[str, Dict]
    validation: Set[Tuple[str, str]] = field(default_factory=set)
    group_combos: Dict[str, List[str]] = field(default_factory=dict)
    ref_keys: Optional[Set[str]] = None
    alf_group_systems: Set[str] = field(default_factory=set)   # group_set = ALF's groups
    scouting_ns: float = 0.0

    @property
    def systems(self) -> List[str]:
        return [s['name'] for s in self.config.get('systems', [])]

    @property
    def stage_names(self) -> List[str]:
        return [s['name'] for s in self.stages]

    def group_test(self) -> Tuple[float, int]:
        adv = next((s.get('advance') or {} for s in self.stages
                    if (s.get('advance') or {}).get('type') in ('groups_done', 'combos_balanced')),
                   {})
        return (float(adv.get('min_population_ratio', 0.5)),
                int(adv.get('min_site_transitions', 10)))

    def pass_window(self) -> int:
        """The stages' group_pass_window (the most runs pooled by the group test)."""
        return max([int(s.get('group_pass_window', 1)) for s in self.stages] or [1])

    def stage_group_test(self, stage_name: Optional[str]) -> Dict:
        """The group test of one stage, as training applies it: the balance
        test (min_population_ratio / min_site_transitions), pooled over
        group_pass_window runs, or the production test (advance.production,
        one run, no pooling)."""
        st = next((s for s in self.stages if s.get('name') == stage_name), None)
        if st is None:
            ratio, trans = self.group_test()
            return {'ratio': ratio, 'trans': trans, 'production': None,
                    'window': self.pass_window()}
        adv = st.get('advance') or {}
        prod = adv.get('production') if adv.get('type') == 'groups_done' else None
        return {'ratio': float(adv.get('min_population_ratio', 0.5)),
                'trans': int(adv.get('min_site_transitions', 10)),
                'production': prod or None,
                'window': 1 if prod else int(st.get('group_pass_window', 1))}

    def short_test(self) -> Tuple[int, float]:
        st = next((s for s in self.stages if s.get('run_steps')), {})
        return (int(st.get('explore_min_transitions', 2)),
                float(st.get('explore_min_fraction', 0.2)))


def load_online(config: Dict, run_dir=None) -> OnlineData:
    """Read a run's outputs; ``run_dir`` defaults to the config's output.base_dir.
    The config gets training's defaults first (``with_hindsight_defaults``,
    e.g. each stage's group_pass_window), so runs are judged as training
    judged them."""
    import copy

    from mllf.cb.bandit_config import with_hindsight_defaults
    config = with_hindsight_defaults(copy.deepcopy(config))
    run_dir = Path(run_dir or (config.get('output') or {}).get('base_dir') or '.')
    log_path = run_dir / 'epoch_summary.log'
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} not found; is {run_dir} an online-training "
                                "output directory (output.base_dir)?")
    stages = (config.get('curriculum') or {}).get('stages') or []
    env = {}
    from mllf.analysis.common import environment_of
    for sc in config.get('systems', []):
        env[sc['name']] = environment_of(sc['name'], sc.get('solvent_state'))
    epochs, stage_rows, scouting_ns = parse_epoch_log(log_path)
    ledger = _read_json(run_dir / 'cost_ledger.json') or {}
    stage_state = _read_json(run_dir / 'system_stages.json') or {}
    val_combos = _read_json(run_dir / 'validation_combos.json') or []
    val = {tuple(Path(c).parts[-2:]) for c in val_combos}
    data = OnlineData(config=config, run_dir=run_dir, stages=stages, env=env, epochs=epochs,
                      stage_rows=stage_rows, ledger=ledger, stage_state=stage_state,
                      validation=val, scouting_ns=scouting_ns)
    _attach_combo_sets(data)
    return data


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _attach_combo_sets(data: OnlineData) -> None:
    """Reference pair keys and group combos of the stages' reference_set /
    group_set (None / empty when no stage names them)."""
    rset = next((s.get('reference_set') for s in data.stages if s.get('reference_set')), None)
    gset = next((s.get('group_set') for s in data.stages if s.get('group_set')), None)
    if not (rset or gset):
        return
    from mllf.cb.system_stages import combo_pair_keys
    from mllf.file_handling.combo_sets import ALF_GROUPS, build_combo_sets
    data.ref_keys = set() if rset else None
    for sc in data.config.get('systems', []):
        if gset and (sc.get('combo_sets') or {}).get(gset) == ALF_GROUPS:
            data.alf_group_systems.add(sc['name'])
        try:
            sets, _ = build_combo_sets(sc, data.config.get('alf_root'))
        except Exception as e:            # e.g. alf_root not readable here
            log.warning(f"  Warning: combo sets of {sc['name']} not built ({e})")
            continue
        if rset:
            data.ref_keys |= {k for c in sets.get(rset, [])
                              for k in combo_pair_keys(sc['name'], c)}
        if gset:
            data.group_combos[sc['name']] = [Path(g).name for g in sets.get(gset, [])]


# epoch_summary.log

_EPOCH_BLOCK_RE = re.compile(r"Epoch (\d+) Summary:[^\n]*\n(.*?)(?=\nEpoch \d+ Summary:|\Z)",
                             re.DOTALL)
_GROUPS_RE = re.compile(r"^\s*Groups: (\d+) run, (\d+) passed", re.M)
_SCOUT_RE = re.compile(r"\[SCOUT\] \S+: .*?, ([\d.]+) ns")


def parse_epoch_log(path: Path):
    """(per epoch, per epoch and system, scouting ns) from epoch_summary.log.

    Per epoch: pair runs good / run by length (short, full), groups run /
    passed, held-out pairs good / run, the validation gate's decision and the
    training ns. Per system: its stage, epoch within the stage, progress
    (done / total and what is counted), reference pairs resolved, and the
    stage change (next stage, reason) if it left the stage that epoch."""
    import pandas as pd

    from mllf.cb.epoch_log import parse_epoch_block
    text = path.read_text()
    rows, srows = [], []
    for m in _EPOCH_BLOCK_RE.finditer(text):
        epoch, block = int(m.group(1)), m.group(2)
        nb = parse_epoch_block(block)
        row = {'epoch': epoch}
        for k in ('short', 'full', 'validation'):
            good, total = nb.get(k, (np.nan, np.nan))
            row[f'{k}_good'], row[f'{k}_runs'] = good, total
        g = _GROUPS_RE.search(block)
        row['groups_run'], row['groups_passed'] = ((int(g.group(1)), int(g.group(2))) if g
                                                   else (0, 0))
        row['gate'] = nb.get('gate')
        sysrows = nb.get('systems') or []
        row['train_ns'] = sum(r['ns'] for r in sysrows) if sysrows else np.nan
        rows.append(row)
        changes = nb.get('changes') or {}
        for r in sysrows:
            nxt, why = changes.get(r['system'], (None, None))
            prog = r.get('progress')
            srows.append({'epoch': epoch, 'system': r['system'], 'stage': r['stage'],
                          'stage_epoch': r['stage_epoch'], 'stage_max': r['stage_max'],
                          'done': prog[0] if prog else np.nan,
                          'total': prog[1] if prog else np.nan,
                          'counted': prog[2] if prog else None,
                          'ref_resolved': (r.get('ref') or (np.nan, np.nan))[0],
                          'ref_total': (r.get('ref') or (np.nan, np.nan))[1],
                          'ns': r['ns'], 'next': nxt, 'reason': why})
    epochs = pd.DataFrame(rows)
    if not epochs.empty:
        epochs = epochs.sort_values('epoch').reset_index(drop=True)
        for k in ('short', 'full'):
            epochs[f'{k}_pct'] = 100 * epochs[f'{k}_good'] / epochs[f'{k}_runs']
        good = epochs[['short_good', 'full_good']].sum(axis=1, min_count=1)
        runs = epochs[['short_runs', 'full_runs']].sum(axis=1, min_count=1)
        epochs['pairs_good'], epochs['pairs_run'] = good, runs
        epochs['pairs_pct'] = 100 * good / runs
        epochs['groups_pct'] = (100 * epochs['groups_passed']
                                / epochs['groups_run'].replace(0, np.nan))
        epochs['validation_pct'] = 100 * epochs['validation_good'] / epochs['validation_runs']
    scouting_ns = sum(float(x) for x in _SCOUT_RE.findall(text.split('Epoch 0 Summary')[0]))
    return epochs, pd.DataFrame(srows), scouting_ns


# run outcomes

def combo_subs(name: str) -> List[Tuple[int, List[int]]]:
    """[(site, [subs])] of a combo name."""
    return [(int(s), [int(x) for x in subs.strip('_').split('_')])
            for s, subs in re.findall(r'site(\d+)_subs_([\d_]+?)(?=__site|$|_run)', name)]


def classify_pair(p0: float, p1: float, crossings: Optional[float],
                  test: Tuple[int, float] = FULL_TEST) -> str:
    total = p0 + p1
    if p0 <= 0 and p1 <= 0:
        return 'neither sampled'
    if p0 <= 0 or p1 <= 0:
        return 'one-sided'
    if min(p0, p1) / total < test[1]:
        return 'unbalanced'
    if (crossings or 0) < test[0]:
        return 'few crossings'
    return 'success'


def classify_group(pops: List[float], trans: List[float], nsubs: List[int],
                   ratio: float = 0.5, min_trans: int = 10) -> str:
    """Worst site decides (unsampled > uneven > few transitions > passed)."""
    rank = {o: i for i, o in enumerate(GROUP_OUTCOMES)}
    worst, start = 'passed', 0
    # (a run passes alone or not; 'passed with previous run' is set in scan_runs)
    for s, n in enumerate(nsubs):
        site = pops[start:start + n]
        start += n
        if len(site) < n or min(site, default=0) <= 0:
            out = 'unsampled substituent'
        elif ratio > 0 and min(site) / sum(site) < ratio / n:
            out = 'uneven'
        elif min_trans > 0 and (s >= len(trans) or trans[s] < min_trans):
            out = 'few transitions'
        else:
            out = 'passed'
        if rank[out] > rank[worst]:
            worst = out
    return worst


def _counts(saved: Dict) -> Tuple[List[float], List[float], Dict]:
    pops = saved.get('populations') or []
    if isinstance(pops, dict):
        pops = [pops[k] for k in sorted(pops, key=int)]
    trans = saved.get('transitions') or []
    if isinstance(trans, dict):
        trans = [trans[k] for k in sorted(trans, key=int)]
    return [float(p) for p in pops], [float(t) for t in trans], saved.get('transition_pairs') or {}


def _read_run(run_dir: Path) -> Optional[Dict]:
    """Populations and transitions of a finished run: epoch_results.pt, else
    the run's output (scouting runs have no epoch_results.pt)."""
    path = run_dir / 'epoch_results.pt'
    if path.exists():
        import torch
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            return None
    out = run_dir / 'output.out'
    if out.exists():
        from mllf.cli.collect_pretraining_data import parse_output_file
        from mllf.pretrain.rewards import _extract_highest_lambda_counts
        sr = parse_output_file(out, run_dir)
        if sr:
            tr = sr.get('transitions') or {}
            return {'populations': _extract_highest_lambda_counts(sr.get('populations', {})),
                    'transitions': [(tr[k] or {}).get('0.99', 0) if isinstance(tr[k], dict)
                                    else tr[k] for k in sorted(tr, key=int)],
                    'transition_pairs': sr.get('transition_pairs') or {}}
    return None


def _run_record(task) -> Optional[Dict]:
    run_dir, rec, role, short_test, gtest = task
    saved = _read_run(run_dir)
    if saved is None:
        return None
    pops, trans, tpairs = _counts(saved)
    combo = run_dir.parent.name
    sites = combo_subs(combo)
    nsubs = [len(s) for _, s in sites]
    ns = rec.get('ns')
    length = 'short' if ns is not None and ns < SHORT_RUN_NS else 'full'
    out = {'system': rec.get('system'), 'combo': combo, 'run': run_dir.name,
           'epoch': int(rec.get('epoch', 0)), 'stage': rec.get('stage'), 'kind': rec.get('kind'),
           'role': role, 'length': length, 'ns': ns}
    if role == 'group' or (nsubs and (len(nsubs) > 1 or nsubs[0] > 2)):
        out['outcome'] = classify_group(pops, trans, nsubs, gtest['ratio'], gtest['trans'])
        if gtest['production']:
            from mllf.cb.system_stages import production_balanced
            if production_balanced(saved, nsubs, gtest['production']):
                out['outcome'] = 'passed'
            elif out['outcome'] == 'passed':
                out['outcome'] = 'short of production test'
        out['test'] = 'group'
        out['_window'] = gtest['window']
        out['_ratio_trans'] = (gtest['ratio'], gtest['trans'])
        out['_counts'] = (pops, trans, nsubs)
    else:
        p0 = pops[0] if pops else 0.0
        p1 = pops[1] if len(pops) > 1 else 0.0
        t = tpairs.get('2_3', trans[0] if len(trans) == 1 else None)
        if re.match(r'pair\d+r_', combo):           # reverse-orientation run
            p0, p1 = p1, p0
        out['outcome'] = classify_pair(p0, p1, t, short_test if length == 'short' else FULL_TEST)
        out['test'] = 'pair'
    return out


def scan_runs(data: OnlineData, workers: int = 8):
    """One row per run in the cost ledger: system, combo, epoch, stage, role
    ('reference pair', 'other pair', 'group', 'validation', 'scouting'),
    length and outcome."""
    import pandas as pd

    from mllf.cb.system_stages import combo_pair_keys
    short = data.short_test()
    groups = {(s, g) for s, gs in data.group_combos.items() for g in gs}
    tasks = []
    for rd, rec in data.ledger.items():
        run_dir = Path(rd)
        if not run_dir.is_absolute() and not run_dir.exists():
            run_dir = data.run_dir / rd
        combo, sysn = run_dir.parent.name, rec.get('system')
        kind = rec.get('kind')
        if kind == 'validation' or (sysn, combo) in data.validation:
            role = 'validation'
        elif kind == 'scout':
            role = 'scouting'
        elif (sysn, combo) in groups or combo.startswith('group'):
            role = 'group'
        elif data.ref_keys is None or any(k in data.ref_keys
                                          for k in combo_pair_keys(sysn, combo)):
            role = 'reference pair'
        else:
            role = 'other pair'
        tasks.append((run_dir, rec, role, short, data.stage_group_test(rec.get('stage'))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(_run_record, tasks) if r]
    log.info(f"  read {len(rows)} of {len(tasks)} runs in the cost ledger")
    _pool_group_runs(rows)
    for r in rows:
        for k in ('_counts', '_window', '_ratio_trans'):
            r.pop(k, None)
    return pd.DataFrame(rows)


def _pool_group_runs(rows: List[Dict]) -> None:
    """A group run that failed alone but passes pooled with the group's
    previous runs (its stage's group_pass_window - 1 of them;
    ``system_stages.pooled_group_test``, as training judges it) becomes
    'passed with previous run'. Stages with the production test do not pool."""
    from mllf.cb.system_stages import pooled_group_test
    by_group: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        # training pools a group's training runs only (not its scouting run)
        if r.get('_counts') is not None and r['role'] == 'group':
            by_group.setdefault((r['system'], r['combo']), []).append(r)
    for runs in by_group.values():
        runs.sort(key=lambda r: r['epoch'])
        for k, r in enumerate(runs):
            window = int(r.get('_window', 1))
            if r['outcome'] == 'passed' or k == 0 or window <= 1:
                continue
            hist = [{'run': x['run'], 'populations': x['_counts'][0],
                     'transitions': x['_counts'][1]} for x in runs[max(0, k - window + 1):k + 1]]
            if pooled_group_test(hist, r['_counts'][2], *r['_ratio_trans'])[0]:
                r['outcome'] = 'passed with previous run'


# ALF

def alf_group_runs(dataset_dir: Path, ratio: float,
                   min_trans: int) -> List[Tuple[str, float, bool]]:
    """ALF's runs of one group in run order: (run, simulated ns, passed the
    group test). Runs without a readable length or results are skipped."""
    from mllf.cb.cost_ledger import _run_sort_key, simulated_ns
    from mllf.cb.system_stages import site_balanced
    out = []
    for run in sorted(Path(dataset_dir).glob('run[0-9]*'), key=_run_sort_key):
        meta = _read_json(run / 'metadata.json')
        res = _read_json(run / 'simulation_results.json')
        if meta is None or res is None:
            continue
        ns = res.get('simulated_ns')
        if ns is None and meta.get('source_run_dir'):
            ns = simulated_ns(Path(meta['source_run_dir']) / 'output')
        if ns is None:
            continue
        pops_d, trans_d = res.get('populations') or {}, res.get('transitions') or {}
        pops = [(pops_d[k].get('counts') or {}).get('0.99', 0) for k in sorted(pops_d, key=int)]
        trans = [trans_d[k].get('0.99', 0) for k in sorted(trans_d, key=int)]
        ok = site_balanced({'populations': pops, 'transitions': trans},
                           meta.get('nsubs_per_site') or [], ratio, min_trans)
        out.append((run.name, float(ns), bool(ok)))
    return out


def alf_cache_path(pretraining_root) -> Path:
    """One cache of ALF's runs per pretraining root, shared by every analysis
    (``$XDG_CACHE_HOME/mllf``, default ``~/.cache/mllf``): those datasets do
    not change, and reading them takes minutes."""
    import hashlib
    import os
    root = str(Path(pretraining_root).resolve())
    base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'mllf'
    return base / f"alf_runs_{hashlib.sha1(root.encode()).hexdigest()[:12]}.pkl"


def alf_runs(data: OnlineData, pretraining_root,
             cache_path: Optional[Path] = None) -> Dict[str, Dict[str, list]]:
    """{system: {ALF dataset: alf_group_runs}} for the run's systems, read once
    per group test into ``cache_path`` (default ``alf_cache_path``)."""
    from mllf.analysis.common import load_cache, save_cache
    from mllf.cb.cost_ledger import alf_dataset_dirs
    cache_path = cache_path or alf_cache_path(pretraining_root)
    cache = load_cache(cache_path)
    key = data.group_test()
    by_test = cache.setdefault(key, {})
    changed = False
    for s in data.systems:
        if s not in by_test:
            by_test[s] = {d.name: alf_group_runs(d, *key)
                          for d in alf_dataset_dirs(s, pretraining_root)}
            changed = True
            log.info(f"  ALF runs read: {s} ({len(by_test[s])} datasets)")
            save_cache(cache_path, cache)        # keep progress if interrupted
    if changed:
        save_cache(cache_path, cache)
    return {s: by_test[s] for s in data.systems if by_test.get(s)}


def _pass_epoch(run_dir: str, st: Dict) -> Optional[int]:
    """Epoch of a group's passing run: -1 for a scouting run, else run_NNN."""
    if run_dir in (st.get('scouted_groups') or {}).values():
        return -1
    m = re.search(r'run_(\d+)', str(run_dir))
    return int(m.group(1)) if m else None


def group_attributed_ns(system: str, ledger: Dict[str, Dict], groups: List[str],
                        through: Dict[str, Optional[int]]) -> Dict[str, float]:
    """Training ns of each group: its own runs plus its pair runs, a pair in
    several groups split equally among them, through ``through[group]`` (the
    epoch it passed; None counts every run). Validation runs are left out."""
    from mllf.cb.system_stages import combo_pair_keys
    keys = {g: set(combo_pair_keys(system, g)) for g in groups}
    out = {g: 0.0 for g in groups}
    for rd, rec in ledger.items():
        if rec.get('system') != system or rec.get('kind') == 'validation':
            continue
        ns, e, combo = float(rec.get('ns') or 0.0), int(rec.get('epoch', 0)), Path(rd).parent.name
        owners = [combo] if combo in out else [
            g for g in groups if any(k in keys[g] for k in combo_pair_keys(system, combo))]
        for g in owners:
            if through.get(g) is None or e <= through[g]:
                out[g] += ns / len(owners)
    return out


def convergence_tables(data: OnlineData, alf: Dict[str, Dict[str, list]]):
    """(per system, per group) simulated ns to convergence, MLLF vs ALF.

    MLLF, per system: training ns (scouting included, validation excluded)
    through the epoch the system met its stage criterion, or every ns used
    if it never did. Per group: ``group_attributed_ns`` through the epoch the
    group passed. ALF, per group: ns through its first run passing the same
    group test; per system: the sum over its groups (converged only if every
    group passed), and ALF's total."""
    import pandas as pd
    df = pd.DataFrame(list(data.ledger.values()))
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df[df['kind'] != 'validation']
    df['ns'] = df['ns'].fillna(0.0)
    cum = df.groupby(['system', 'epoch'])['ns'].sum().groupby(level=0).cumsum()
    sys_rows, grp_rows = [], []
    for sysn in data.systems:
        st = data.stage_state.get(sysn) or {}
        c = cum.loc[sysn] if sysn in cum.index.get_level_values(0) else None
        conv = bool(st.get('converged'))
        end = (st.get('history') or [{}])[-1].get('ended_epoch')
        if c is None:
            ours = np.nan
        elif conv and end is not None:
            ours = float(c[c.index <= end].iloc[-1]) if (c.index <= end).any() else 0.0
        else:
            ours = float(c.iloc[-1])
        groups = alf.get(sysn) or {}
        passed = st.get('groups_done') or {}
        gcombos = data.group_combos.get(sysn) or sorted(passed)
        through = {g: _pass_epoch(passed[g], st) if g in passed else None for g in gcombos}
        attributed = group_attributed_ns(sysn, data.ledger, gcombos, through) if gcombos else {}
        alf_pass, alf_total, all_pass = 0.0, 0.0, bool(groups)
        alf_sorted = sorted(groups.items(), key=lambda kv: _natural(kv[0]))
        # ALF's datasets are its groups, in the order the alf_groups set numbers ours
        matched = sysn in data.alf_group_systems
        for gi, gc in enumerate(gcombos, start=1):
            name, runs = (alf_sorted[gi - 1] if matched and gi <= len(alf_sorted)
                          else (None, []))
            ns_cum, ns_pass = 0.0, None
            for _, ns, ok in runs:
                ns_cum += ns
                if ok and ns_pass is None:
                    ns_pass = ns_cum
            grp_rows.append({
                'system': sysn, 'group': gi, 'combo': gc, 'alf_dataset': name,
                'mllf_passed': gc in passed,
                'mllf_by_scouting': gc in (st.get('scouted_groups') or {}),
                'mllf_ns': attributed.get(gc, np.nan),
                'alf_passed': ns_pass is not None if name else None,
                'alf_ns': (ns_pass if ns_pass is not None else ns_cum) if name else np.nan,
                'alf_ns_total': ns_cum if name else np.nan})
        for name, runs in alf_sorted:
            ns_cum, ns_pass = 0.0, None
            for _, ns, ok in runs:
                ns_cum += ns
                if ok and ns_pass is None:
                    ns_pass = ns_cum
            alf_total += ns_cum
            if ns_pass is None:
                all_pass = False
            else:
                alf_pass += ns_pass
        sys_rows.append({'system': sysn, 'environment': data.env.get(sysn),
                         'mllf_converged': conv, 'mllf_ns': ours,
                         'mllf_groups_passed': len(passed), 'groups': len(gcombos),
                         'alf_converged': all_pass if groups else None,
                         'alf_ns': (alf_pass if all_pass else alf_total) if groups else np.nan,
                         'alf_ns_total': alf_total if groups else np.nan,
                         'alf_groups_passed': sum(1 for _, r in alf_sorted
                                                  if any(ok for *_, ok in r))})
    return pd.DataFrame(sys_rows), pd.DataFrame(grp_rows)


def _natural(name: str):
    m = re.search(r'(\d+)$', name)
    return (int(m.group(1)) if m else 10 ** 9, name)
