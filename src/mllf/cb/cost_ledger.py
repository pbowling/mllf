"""Simulation cost ledger: ns simulated per system and stage, and ALF's cost.

The paper's comparison is compute to converged biases, so both sides are
measured the same way: the simulated time in each run's CHARMM output (every
dynamics call prints ``NSTEP = n ... NSAVC`` followed by ``TIME STEP = ... PS``;
equilibration counts, since it is compute spent).

Online training records every finished run in ``<checkpoint dir>/cost_ledger.json``
keyed by its run directory (so a resumed epoch is not counted twice):
``{run_dir: {system, stage, kind, epoch, ok, ns}}`` with kind 'train' or
'validation'. ``summarize_ledger`` totals it per system and stage.

ALF's cost comes from the pretraining datasets (``pretraining/<system>_group*``
etc.): each ``run*/metadata.json`` names the original ALF run directory, whose
``output`` file is parsed the same way (``alf_cost``).
"""
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_NSTEP_RE = re.compile(r"^\s+NSTEP\s*=\s*(\d+)\s+NSAVC")
_TIMESTEP_RE = re.compile(r"TIME STEP\s*=\s*\S+\s+AKMA\s+(\S+)\s+PS")

LEDGER_FILE = 'cost_ledger.json'


def simulated_ns(output_file) -> Optional[float]:
    """Simulated time (ns) in one CHARMM/pyCHARMM output: the sum over its
    dynamics calls of NSTEP x that call's time step. None if the file is
    missing or holds no dynamics."""
    path = Path(output_file)
    if not path.is_file():
        return None
    total_ps = 0.0
    pending = None          # NSTEP of the dynamics call awaiting its time step
    found = False
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = _NSTEP_RE.match(line)
            if m:
                pending = int(m.group(1))
                continue
            if pending is not None:
                t = _TIMESTEP_RE.search(line)
                if t:
                    total_ps += pending * float(t.group(1).replace('D', 'E'))
                    pending = None
                    found = True
    return total_ps / 1000.0 if found else None


def load_ledger(path) -> Dict[str, Dict]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def record_runs(path, run_dirs: Iterable[Tuple[str, str, bool]], kind: str, epoch: int,
                stage_of_system: Optional[Dict[str, str]] = None,
                output_name: str = 'output.out') -> int:
    """Add finished runs ((system, run_dir, ok) from train_epoch's
    ``run_dirs``) to the ledger at ``path``. Returns the number added."""
    ledger = load_ledger(path)
    n = 0
    for system, run_dir, ok in run_dirs:
        if run_dir in ledger:
            continue
        ledger[run_dir] = {
            'system': system,
            'stage': (stage_of_system or {}).get(system),
            'kind': kind,
            'epoch': int(epoch),
            'ok': bool(ok),
            'ns': simulated_ns(Path(run_dir) / output_name),
        }
        n += 1
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ledger, indent=1))
    return n


def summarize_ledger(ledger: Dict[str, Dict]) -> Dict[str, Dict]:
    """Per system: {'stages': {stage: {'runs', 'ns'}}, 'validation': {...},
    'train': {...} (training runs only, the cost of converging the system),
    'total': {...} (training + validation), 'failed_runs'}. Runs whose output
    gave no time count as runs with 0 ns."""
    out: Dict[str, Dict] = {}
    for rec in ledger.values():
        s = out.setdefault(rec.get('system') or 'unknown', {
            'stages': {}, 'validation': {'runs': 0, 'ns': 0.0}, 'train': {'runs': 0, 'ns': 0.0},
            'total': {'runs': 0, 'ns': 0.0}, 'failed_runs': 0})
        ns = float(rec.get('ns') or 0.0)
        bucket = (s['validation'] if rec.get('kind') == 'validation'
                  else s['stages'].setdefault(str(rec.get('stage')), {'runs': 0, 'ns': 0.0}))
        for b in (bucket, s['total']) + (() if rec.get('kind') == 'validation' else (s['train'],)):
            b['runs'] += 1
            b['ns'] += ns
        if not rec.get('ok', True):
            s['failed_runs'] += 1
    return out


# ── ALF baseline ─────────────────────────────────────────────────────────────

def alf_dataset_dirs(system: str, pretraining_root, explicit: Optional[List[str]] = None) -> List[Path]:
    """Pretraining dataset dirs holding ALF runs for ``system``: ``explicit``
    (names or paths) if given, else ``<root>/<system>`` and
    ``<root>/<system>_group*`` (``_vac`` also tried as ``_vacuum``) that
    contain ``run*`` directories."""
    root = Path(pretraining_root)
    if explicit:
        return [Path(e) if Path(e).is_absolute() else root / e for e in explicit]
    names = {system}
    if system.endswith('_vac'):
        names.add(system[:-4] + '_vacuum')
    dirs = []
    for name in sorted(names):
        for d in [root / name] + sorted(root.glob(f"{name}_group*")):
            if d.is_dir() and any(d.glob('run[0-9]*')) and d not in dirs:
                dirs.append(d)
    return dirs


def _run_sort_key(p: Path):
    m = re.match(r'run(\d+)(\D*)$', p.name)
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, p.name)


def alf_cost(dataset_dirs: Iterable[Path]) -> Dict[str, Dict]:
    """ALF's simulated time per dataset (group): {'runs', 'ns', 'missing'},
    from each run's original output (``metadata.json`` -> source_run_dir /
    output). ``missing`` counts runs whose output could not be read (e.g.
    production replicas whose outputs were removed)."""
    out = {}
    for d in dataset_dirs:
        rec = {'runs': 0, 'ns': 0.0, 'missing': 0}
        for run in sorted(Path(d).glob('run[0-9]*'), key=_run_sort_key):
            src = None
            meta = run / 'metadata.json'
            if meta.exists():
                try:
                    src = json.loads(meta.read_text()).get('source_run_dir')
                except Exception:
                    src = None
            ns = None
            for cand in ([Path(src) / 'output'] if src else []) + [run / 'output', run / 'output.out']:
                ns = simulated_ns(cand)
                if ns is not None:
                    break
            if ns is None:
                rec['missing'] += 1
                continue
            rec['runs'] += 1
            rec['ns'] += ns
        out[Path(d).name] = rec
    return out


def main() -> None:
    """Per-system cost of an online run next to ALF's::

        python -m mllf.cb.cost_ledger training/checkpoints_x --pretraining-root pretraining
    """
    import argparse
    p = argparse.ArgumentParser(description='Simulated ns per system: online run vs ALF')
    p.add_argument('checkpoint_dir', help='online run output dir (holds cost_ledger.json)')
    p.add_argument('--pretraining-root', default=None,
                   help='pretraining datasets dir, for ALF cost (omit to skip ALF)')
    p.add_argument('--json', default=None, help='also write the table to this JSON file')
    args = p.parse_args()

    summary = summarize_ledger(load_ledger(Path(args.checkpoint_dir) / LEDGER_FILE))
    stages_path = Path(args.checkpoint_dir) / 'system_stages.json'
    stages = json.loads(stages_path.read_text()) if stages_path.exists() else {}
    rows = {}
    print(f"{'system':28s} {'train ns':>9s} {'runs':>6s} {'valid ns':>9s} {'ALF ns':>9s} {'ALF runs':>9s}  stages")
    for system in sorted(summary):
        s = summary[system]
        alf = None
        if args.pretraining_root:
            per_group = alf_cost(alf_dataset_dirs(system, args.pretraining_root))
            if per_group:
                alf = {'runs': sum(g['runs'] for g in per_group.values()),
                       'ns': sum(g['ns'] for g in per_group.values()),
                       'missing': sum(g['missing'] for g in per_group.values()),
                       'groups': per_group}
        st = stages.get(system) or {}
        hist = ' > '.join(f"{h['stage']}({h['epochs']}, {h['reason']})" for h in st.get('history', []))
        rows[system] = {'online': s, 'alf': alf, 'stages': st}
        print(f"{system:28s} {s['train']['ns']:9.1f} {s['train']['runs']:6d} {s['validation']['ns']:9.1f} "
              f"{(alf['ns'] if alf else float('nan')):9.1f} {(alf['runs'] if alf else 0):9d}  {hist}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
