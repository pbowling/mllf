"""``mllf analyze pretrain``: figures and a report for a pretrained model.

The model predicts the runs it was trained on (``mllf.analysis.pretrain_data``),
and each prediction is compared with the run's own coefficient where the run's
biases were good for it.

Figures (``<out>/figures``):

* ``parity_all``: predicted vs actual for every family, one panel per
  coefficient; the largest errors are labelled with their system
* ``parity_systems/<system>``: the same per system
* ``embedding_pca``: principal components of the substituents' Uni-Mol
  ligand embeddings
* ``similarity``: distance between systems in Uni-Mol ligand-embedding space
  (``similarity_full``: per system and environment, with the full 515-D node
  input)
* ``dataset_stats`` (only with ``dataset_stats``: it lists every pretraining
  directory, minutes for a full set; cached): runs per dataset, substituents
  and sites per run, and the coefficients' distributions

Tables (``<out>/tables``): ``predictions`` (every directed pair and
coefficient of every analyzed run, with ``scored`` marking the targets),
``metrics_by_system``, ``embedding_pca`` and ``similarity``. ``report.txt``
summarizes the model, its accuracy and the data.
"""
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from mllf.analysis.common import Output, table_text


log = logging.getLogger(__name__)

FIGURES = ('parity_all', 'parity_systems', 'embedding_pca', 'similarity', 'dataset_stats')
# the figures drawn by default (dataset_stats needs dataset_stats=True)
DEFAULT_FIGURES = FIGURES[:-1]


def analyze_pretrain(model_dir: Path, out_dir, model: Optional[str] = None,
                     config_path: Optional[str] = None, device: Optional[str] = None,
                     max_runs: Optional[int] = 10, only: Optional[Sequence[str]] = None,
                     workers: int = 8, rescan: bool = False, fmt: str = 'png',
                     grayscale: bool = False, dataset_stats: bool = False,
                     similarity_full: bool = False) -> Path:
    """Write the figures, tables and report of a pretrained model; returns the
    report's path."""
    from mllf.analysis import pretrain_data as pdata
    from mllf.analysis import pretrain_plots as pp
    from mllf.analysis import style as bs
    bs.apply_style()
    model_dir = Path(model_dir)
    if not only:
        only = list(DEFAULT_FIGURES) + (['dataset_stats'] if dataset_stats else [])
    out = Output(out_dir, only=only, fmt=fmt, grayscale=grayscale)
    cache = out.dir / 'cache' / 'predictions.pkl'
    if rescan and cache.exists():
        cache.unlink()
    data = pdata.collect(model_dir, cache, model=model, config_path=config_path,
                         device=device, max_runs=10 if max_runs is None else max_runs,
                         workers=workers)
    records = data['runs']
    df = prediction_table(records)
    out.table('predictions', df)
    scored = df[df['scored']]
    by_system = metric_table(scored, 'system')
    by_family = metric_table(scored, 'family')
    out.table('metrics_by_system', by_system)
    out.table('metrics_by_family', by_family)

    subs, X = pp.substituent_table(records)
    systems, D = pp.system_distances(subs, X)
    full_groups, full_D = [], None
    if similarity_full:
        try:
            full_groups, full_D = pp.system_distances(*pp.substituent_table(records, full=True))
        except ValueError as e:
            out.skipped['similarity_full'] = str(e)
    if len(X) >= 4:
        scores, _ = pp.pca(X, 3)
        tab = subs.copy()
        for k in range(scores.shape[1]):
            tab[f'pc{k + 1}'] = scores[:, k]
        out.table('embedding_pca', tab)
    if systems:
        import pandas as pd
        out.table('similarity', pd.DataFrame(D, index=systems, columns=systems)
                  .reset_index().rename(columns={'index': 'system'}))
    stats = None
    if out.wanted('dataset_stats'):
        stats = _cached_manifest_stats(model_dir, out.dir / 'cache' / 'dataset_stats.pkl',
                                       workers, rescan)

    log.info(f"Figures in {out.figures}")
    out.figure('parity_all', lambda: pp.parity_all(scored))
    if out.wanted('parity_systems'):
        for s in sorted(scored['system'].unique()):
            out.figure(f'parity_systems/{s}', lambda s=s: pp.parity_system(scored, s))
    out.figure('embedding_pca', lambda: pp.embedding_pca(subs, X))
    out.figure('similarity', lambda: pp.similarity_heatmap(systems, D))
    if full_groups:
        if out.only is not None:
            out.only.add('similarity_full')
        out.figure('similarity_full', lambda: pp.similarity_heatmap(full_groups, full_D))
        import pandas as pd
        out.table('similarity_full', pd.DataFrame(full_D, index=full_groups, columns=full_groups)
                  .reset_index().rename(columns={'index': 'system'}))
    out.figure('dataset_stats', lambda: pp.dataset_stats(stats, scored))
    families = dict(zip(subs['group'], subs['family'])) if len(subs) else {}
    near = pp.nearest_systems(systems, D, families)
    path = out.report(report_text(data, df, scored, by_family, by_system, near, stats, out))
    log.info(f"Report: {path}")
    return path


def prediction_table(records):
    """One row per (run, directed pair, coefficient)."""
    import pandas as pd

    from mllf.analysis.pretrain_data import BIASES
    rows = []
    for r in records:
        nodes = r['nodes']
        for e, (i, j) in enumerate(r['pairs']):
            ni, nj = nodes[i] if i < len(nodes) else {}, nodes[j] if j < len(nodes) else {}
            for d, bias in enumerate(BIASES):
                rows.append({'dataset': r['dataset'], 'system': r['system'], 'family': r['family'],
                             'environment': r['environment'], 'run_dir': r['run_dir'],
                             'site': ni.get('site'), 'sub_i': ni.get('sub'),
                             'sub_j': nj.get('sub'), 'bias': bias,
                             'actual': float(r['actual'][e, d]),
                             'predicted': float(r['pred'][e, d]),
                             'sd': float(r['sd'][e, d]), 'scored': bool(r['good'][e, d])})
    return pd.DataFrame(rows)


def metric_table(scored, by: str):
    """N, R², MUE and RMSE per coefficient for each value of ``by``."""
    import pandas as pd

    from mllf.analysis.pretrain_plots import metrics
    rows = []
    for key, g in scored.groupby(by):
        row = {by: key}
        for bias, gb in g.groupby('bias'):
            m = metrics(gb['actual'].to_numpy(), gb['predicted'].to_numpy())
            for k in ('n', 'r2', 'mue', 'rmse'):
                row[f'{bias}_{k}'] = m[k]
        rows.append(row)
    return pd.DataFrame(rows)


def _cached_manifest_stats(model_dir: Path, cache: Path, workers: int, rescan: bool) -> Dict:
    from mllf.analysis.common import load_cache, save_cache
    manifest = model_dir / 'pretraining_manifest.json'
    key = (str(manifest.resolve()), manifest.stat().st_mtime)
    data = {} if rescan else load_cache(cache)
    if data.get('key') != key:
        log.info("Scanning the pretraining directories for dataset statistics")
        data = {'key': key, 'stats': manifest_stats(model_dir, workers)}
        save_cache(cache, data)
    return data['stats']


def _count_runs(d: Path) -> int:
    try:
        return sum(1 for p in d.iterdir() if p.name.startswith('run') and p.is_dir())
    except OSError:
        return -1


def manifest_stats(model_dir: Path, workers: int = 8) -> Dict:
    """Runs per dataset, substituents and sites per run, from the manifest's
    directory names and run subdirectories (no run files are read)."""
    import concurrent.futures

    from mllf.analysis.pretrain_data import manifest_dirs
    from mllf.pretrain.runs import system_label
    per_dataset: Counter = Counter()
    subs, sites = Counter(), Counter()
    dirs = manifest_dirs(model_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        counts = list(ex.map(_count_runs, dirs))
    for d, n in zip(dirs, counts):
        if n < 0:
            continue
        per_dataset[system_label(d)] += n
        groups = re.findall(r'site\d+_subs_([\d_]+?)(?=__site|$)', d.name)
        if groups:
            subs[sum(len(g.strip('_').split('_')) for g in groups)] += n
            sites[len(groups)] += n
    return {'runs_per_dataset': dict(per_dataset), 'subs_per_run': dict(subs),
            'sites_per_run': dict(sites), 'runs': sum(per_dataset.values()),
            'datasets': len(per_dataset)}


def report_text(data, df, scored, by_family, by_system, near, stats, out: Output) -> str:
    from mllf.analysis.pretrain_data import BIASES
    from mllf.analysis.pretrain_plots import metrics
    st = data['settings']
    records = data['runs']
    lines = [f"Pretrained model analysis: {st['model']}", '',
             'Model', '-----',
             f"Inputs: env_input {st['env_input']}, site size {st['site_size_input']}, other-site "
             f"references {st['ligand_other_sites']}, {st['structure_selection']} structures", '',
             'Data', '----',
             (f"Pretraining set: {stats['runs']} runs in {stats['datasets']} datasets "
              f"({st['directories']} directories)" if stats else
              f"Pretraining set: {st['directories']} directories (--dataset-stats counts "
              "their runs)"),
             f"Analyzed: {len(records)} runs ({st['max_runs'] or 'all'} per dataset at most, "
             f"those with the most well-sampled pairs), {len(df) // 4} directed pairs, "
             f"{len(scored)} scored targets", '']
    rows = []
    for bias in BIASES:
        g = scored[scored['bias'] == bias]
        m = metrics(g['actual'].to_numpy(), g['predicted'].to_numpy())
        sd = g['sd'].to_numpy()
        rows.append([bias, m['n'], m['r2'], m['r'], m['mue'], m['rmse'],
                     float(np.nanmedian(sd)) if np.isfinite(sd).any() else None])
    head = 'Accuracy on scored targets (b in kcal/mol)'
    lines += [head, '-' * len(head),
              table_text(rows, ['Coefficient', 'N', 'R²', 'r', 'MUE', 'RMSE', 'Median sd']), '']
    if not by_family.empty:
        cols = [c for c in ('linear_n', 'linear_mue', 'linear_r2', 'quadratic_mue',
                            'quadratic_r2') if c in by_family]
        t = by_family.sort_values(cols[1] if len(cols) > 1 else 'family', ascending=False)
        lines += ['Per family (sorted by linear MUE, largest first)',
                  table_text([[r['family']] + [r.get(c) for c in cols] for _, r in t.iterrows()],
                             ['Family'] + [c.replace('_', ' ') for c in cols]), '']
    near = {k: v for k, v in near.items() if v}
    if near:
        lines += ['Closest systems of other families (embedding distance; lower = more similar)',
                  table_text([[s] + [f"{n} ({d:.1f})" for n, d in v] for s, v in near.items()],
                             ['System', '1st', '2nd', '3rd'][
                                 :1 + max(len(v) for v in near.values())]),
                  '']
    lines += ['Outputs', '-------', f"Figures: {out.figures} ({len(out.written)})",
              f"Tables: {out.tables}"]
    for name, why in out.skipped.items():
        lines.append(f"  {name} not drawn: {why}")
    return '\n'.join(lines) + '\n'
