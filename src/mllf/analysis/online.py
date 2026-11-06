"""``mllf analyze online``: figures and a report for an online-training run.

Figures (``<out>/figures``):

* ``success_over_epochs``: percent of pair runs, group runs and (optional)
  held-out pairs that passed their test, per epoch
* ``outcomes_by_epoch`` / ``outcomes_by_system``: training runs by outcome
  (counts)
* ``stage_timeline``: the epochs each system spent in each stage, and whether
  it converged
* ``convergence_systems`` / ``convergence_groups``: simulated ns to
  convergence for MLLF and ALF, per system and per group (needs the
  pretraining datasets, ``--pretraining-root``)
* ``free_energy/<system>_<group>_site<N>``: free-energy profiles of each
  passed group's run next to ALF's best run of the group (only with
  ``free_energy``: they read every run's trajectory, which takes minutes per
  system)

``report.txt`` summarizes the run, and ``tables/`` holds the data of each
figure as CSV.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

from mllf.analysis.common import Output, table_text


log = logging.getLogger(__name__)

FIGURES = ('success_over_epochs', 'outcomes_by_epoch', 'outcomes_by_system', 'stage_timeline',
           'convergence_systems', 'convergence_groups', 'free_energy')


def analyze_online(config: Dict, out_dir, run_dir=None, pretraining_root=None,
                   only: Optional[Sequence[str]] = None, validation: bool = True,
                   workers: int = 8, rescan: bool = False,
                   free_energy: Optional[Sequence[str]] = None, fmt: str = 'png',
                   grayscale: bool = False) -> Path:
    """Write the figures, tables and report of an online run; returns the
    report's path. ``free_energy``: None skips the free-energy profiles, an
    empty list draws them for every system, a list of names for those."""
    import pandas as pd

    from mllf.analysis import online_data as od
    from mllf.analysis import online_plots as op
    from mllf.analysis import style as bs
    from mllf.analysis.common import load_cache, save_cache
    bs.apply_style()
    data = od.load_online(config, run_dir)
    out = Output(out_dir, only=_expand(only), fmt=fmt, grayscale=grayscale)
    cache_path = out.dir / 'cache' / 'runs.pkl'
    cache = {} if rescan else load_cache(cache_path)
    key = (od.SCAN_VERSION, len(data.ledger))
    if cache.get('key') != key or 'runs' not in cache:
        cache = {'key': key, 'runs': od.scan_runs(data, workers)}
        save_cache(cache_path, cache)
    runs = cache['runs']
    alf = {}
    if pretraining_root:
        alf = od.alf_runs(data, pretraining_root)
    sys_df, grp_df = od.convergence_tables(data, alf)

    out.table('epochs', data.epochs)
    out.table('stages', data.stage_rows)
    if not runs.empty:
        out.table('runs', runs)
    if not sys_df.empty:
        out.table('convergence_systems', sys_df)
    if not grp_df.empty:
        out.table('convergence_groups', grp_df)

    log.info(f"Figures in {out.figures}")
    out.figure('success_over_epochs', lambda: op.success_over_epochs(data.epochs, validation))
    out.figure('outcomes_by_epoch', lambda: op.outcomes_by_epoch(runs))
    out.figure('outcomes_by_system', lambda: op.outcomes_by_system(runs, data.env))
    totals = {s: len(g) for s, g in data.group_combos.items()}
    out.figure('stage_timeline', lambda: op.stage_timeline(data.stage_rows, data.stage_state,
                                                           data.stage_names, data.env, totals))
    out.figure('convergence_systems', lambda: op.convergence_systems(sys_df, data.env))
    pages = op.convergence_group_pages(grp_df, data.env)
    for k, systems in enumerate(pages, start=1):
        name = 'convergence_groups' if len(pages) == 1 else f'convergence_groups/part{k}'
        out.figure(name, lambda s=systems: op.convergence_groups(grp_df, sys_df, data.env, s))
    if free_energy is not None and out.wanted('free_energy'):
        from mllf.analysis import free_energy as fe
        for rec in fe.group_runs(data, pretraining_root, list(free_energy) or None):
            try:
                for name, make in fe.figures(rec):
                    out.figure(name, make)
            except (ValueError, OSError) as e:
                log.warning(f"  Warning: free-energy profiles of {rec['system']}/{rec['combo']} "
                            f"not drawn ({e})")
    report = report_text(data, runs, sys_df, grp_df, out, pd)
    path = out.report(report)
    log.info(f"Report: {path}")
    return path


def analyze_after_training(config: Dict, run_dir) -> Optional[Path]:
    """The ``analysis:`` config key: analyze the run when training finishes.
    ``analysis: true`` writes to ``<run dir>/analysis`` and compares with ALF
    under the config's ``alf_root``; a mapping may set ``out``,
    ``pretraining_root`` (or ``none``) and ``validation``. A failure is
    reported as a warning, never as a training error."""
    opts = config.get('analysis')
    opts = opts if isinstance(opts, dict) else {}
    root = opts.get('pretraining_root', config.get('alf_root'))
    if root and str(root).lower() == 'none':
        root = None
    out = Path(opts.get('out') or Path(run_dir) / 'analysis')
    try:
        from mllf.analysis.common import require_plotting
        require_plotting()
        log.info(f"\nAnalysis -> {out}")
        fe = opts.get('free_energy')
        fe = [] if fe is True else (list(fe) if fe else None)
        return analyze_online(config, out, run_dir=run_dir, pretraining_root=root,
                              validation=bool(opts.get('validation', True)), free_energy=fe,
                              fmt=opts.get('format', 'png'))
    except Exception as e:
        log.warning(f"  Warning: analysis not completed ({type(e).__name__}: {e}); run "
                    f"'mllf analyze online CONFIG --run-dir {run_dir}' to retry")
        return None


def _expand(only):
    if not only:
        return None
    unknown = [o for o in only if o not in FIGURES]
    if unknown:
        raise ValueError(f"unknown figure(s) {', '.join(unknown)}; "
                         f"choose from {', '.join(FIGURES)}")
    return list(only)


def _sum(series) -> float:
    return float(series.fillna(0).sum()) if series is not None else 0.0


def report_text(data, runs, sys_df, grp_df, out: Output, pd) -> str:
    """The plain-text report (module docstring)."""
    lines = [f"Online training analysis: {data.run_dir}", '']
    ep = data.epochs
    n_epochs = int(ep['epoch'].max()) + 1 if not ep.empty else 0
    ledger = pd.DataFrame(list(data.ledger.values())) if data.ledger else pd.DataFrame()
    train = ledger[ledger['kind'] != 'validation'] if not ledger.empty else ledger
    val = ledger[ledger['kind'] == 'validation'] if not ledger.empty else ledger
    scout = ledger[ledger['kind'] == 'scout'] if not ledger.empty else ledger
    lines += ['Run', '---',
              f"Systems: {len(data.systems)}; epochs: {n_epochs}; stages: "
              f"{', '.join(data.stage_names) or 'none'}",
              f"Simulated time: {_sum(train.get('ns')):.1f} ns of training in {len(train)} runs "
              f"(scouting {_sum(scout.get('ns')):.1f} ns in {len(scout)} runs included); "
              f"validation {_sum(val.get('ns')):.1f} ns in {len(val)} runs, not counted.", '']
    if not runs.empty:
        tr = runs[~runs['role'].isin(['validation', 'scouting'])]
        lines += ['Run outcomes (training runs)', '----------------------------']
        for test, label in (('pair', 'Pair runs'), ('group', 'Group runs')):
            sub = tr[tr['test'] == test]
            if sub.empty:
                continue
            counts = sub['outcome'].value_counts()
            parts = ', '.join(f"{k} {int(v)}" for k, v in counts.items())
            good = int(counts.get('success', 0) + counts.get('passed', 0)
                       + counts.get('passed with previous run', 0))
            lines.append(f"{label}: {good}/{len(sub)} passed "
                         f"({100 * good / len(sub):.0f}%); {parts}")
        lines.append('')
    if not sys_df.empty:
        has_alf = sys_df['alf_ns'].notna().any()
        m_conv = int(sys_df['mllf_converged'].sum())
        lines += ['Convergence', '-----------',
                  f"MLLF: {m_conv}/{len(sys_df)} systems converged, "
                  f"{int(sys_df['mllf_groups_passed'].sum())}/{int(sys_df['groups'].sum())} "
                  "groups passed"]
        if has_alf:
            a = sys_df[sys_df['alf_ns'].notna()]
            a_conv = int(a['alf_converged'].fillna(False).astype(bool).sum())
            lines.append(f"ALF:  {a_conv}/{len(a)} systems converged (same group test), "
                         f"{_sum(a['alf_ns_total']):.1f} ns run in all")
            alf_conv = sys_df['alf_converged'].fillna(False).astype(bool)
            both = sys_df[sys_df['mllf_converged'] & alf_conv]
            if not both.empty:
                lines.append(f"Systems converged by both ({len(both)}): MLLF "
                             f"{_sum(both['mllf_ns']):.1f} ns, "
                             f"ALF {_sum(both['alf_ns']):.1f} ns to "
                             "convergence")
        lines.append('')
        rows = [[r.system, r.environment, r.mllf_converged, r.mllf_ns,
                 f"{r.mllf_groups_passed}/{r.groups}" if r.groups else '-',
                 r.alf_converged, r.alf_ns, r.alf_ns_total]
                for r in sys_df.itertuples()]
        lines += ['Per system (ns: to convergence if converged, else every ns run)',
                  table_text(rows, ['System', 'Env', 'MLLF conv', 'MLLF ns', 'Groups',
                                    'ALF conv', 'ALF ns', 'ALF total ns']), '']
    if not ep.empty:
        last = ep.iloc[-1]
        lines += ['Last epoch', '----------',
                  f"Epoch {int(last['epoch'])}: pair runs good "
                  f"{_frac(last.get('pairs_good'), last.get('pairs_run'))}, groups passed "
                  f"{_frac(last.get('groups_passed'), last.get('groups_run'))}, "
                  "held-out pairs good "
                  f"{_frac(last.get('validation_good'), last.get('validation_runs'))}", '']
    lines += ['Outputs', '-------', f"Figures: {out.figures} ({len(out.written)})",
              f"Tables: {out.tables}"]
    for name, why in out.skipped.items():
        lines.append(f"  {name} not drawn: {why}")
    return '\n'.join(lines) + '\n'


def _frac(n, d) -> str:
    if d is None or d != d or not d:
        return '-'
    return f"{int(n)}/{int(d)} ({100 * n / d:.0f}%)"
