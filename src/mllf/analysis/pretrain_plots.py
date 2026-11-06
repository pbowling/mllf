"""Figures of a pretrained model (``mllf.analysis.pretrain``)."""
from typing import Dict, List, Tuple

import numpy as np

from mllf.analysis import style as bs
from mllf.analysis.pretrain_data import BIASES


LABELS = {'linear': 'b', 'quadratic': 'c', 'skew': 'x', 'end': 's'}
N_OUTLIERS = 4          # labelled per parity panel
N_FAMILIES_SHOWN = 5    # highlighted in the embedding PCA


def _plt():
    import matplotlib.pyplot as plt
    return plt


def metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """N, R² (1 - SS_res / SS_tot), Pearson r, MUE and RMSE."""
    n = len(actual)
    if n < 2:
        return {'n': n, 'r2': np.nan, 'r': np.nan, 'mue': np.nan, 'rmse': np.nan}
    err = pred - actual
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    r = float(np.corrcoef(actual, pred)[0, 1]) if np.std(pred) > 0 and np.std(actual) > 0 \
        else np.nan
    return {'n': n, 'r2': r2, 'r': r, 'mue': float(np.abs(err).mean()),
            'rmse': float(np.sqrt((err ** 2).mean()))}


def _axis_label(kind: str, bias: str) -> str:
    """Axis label of a coefficient; only b (linear) is in kcal/mol."""
    sym = LABELS[bias]
    unit = ' (kcal/mol)' if bias == 'linear' else ''
    return rf"{kind + ' ' if kind else ''}$\mathbfit{{{sym}}}${unit}"


def _count_axis(ax, which: str = 'y') -> None:
    """Integer ticks on a count axis, ending on a tick."""
    from matplotlib.ticker import MaxNLocator
    axis = ax.yaxis if which == 'y' else ax.xaxis
    lo, hi = ax.get_ylim() if which == 'y' else ax.get_xlim()
    loc = MaxNLocator(nbins=5, integer=True, min_n_ticks=2)
    ticks = loc.tick_values(0, hi)
    (ax.set_ylim if which == 'y' else ax.set_xlim)(0, ticks[-1])
    axis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=2))


def _parity_panel(ax, df, bias: str, label_outliers: int = 0, dense: bool = True):
    sub = df[df['bias'] == bias]
    x, y = sub['actual'].to_numpy(), sub['predicted'].to_numpy()
    if len(x) == 0:
        ax.text(0.5, 0.5, 'No targets', transform=ax.transAxes, ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        ax._bowling_no_snap = True
        return
    kw = bs.sns_kwargs('scatter_dense') if dense and len(x) > 300 else bs.sns_kwargs('scatter')
    kw = {k: v for k, v in kw.items() if k in ('s', 'linewidth', 'alpha', 'edgecolor')}
    ax.scatter(x, y, color=bs.PALETTE['plum'], zorder=2, **kw)
    bs.parity(ax, x=x, y=y, band=1.0)
    ax._bowling_no_zero = True          # y = x is the reference here
    ax.set_xlabel(_axis_label('Actual', bias))
    ax.set_ylabel(_axis_label('Predicted', bias))
    m = metrics(x, y)
    bs.stat_text(ax, [(f"MUE = {m['mue']:.2f}", bs.INK), (f"R² = {m['r2']:.2f}", bs.INK),
                      (f"N = {m['n']}", bs.INK)], loc='lower right',
                 size=bs.mpl.rcParams['font.size'])
    if label_outliers:
        err = np.abs(y - x)
        order = np.argsort(-err)
        seen = set()
        for k in order:
            name = sub['system'].iloc[k]
            if name in seen:
                continue
            seen.add(name)
            ax.annotate(name, xy=(x[k], y[k]), xytext=(4, 4), textcoords='offset points',
                        fontsize=bs.mpl.rcParams['font.size'] - 2)
            if len(seen) >= label_outliers:
                break


def parity_all(df):
    """Predicted vs actual for every scored target, one panel per coefficient;
    the systems of the largest errors are labelled. Gray band: ±1 of the
    coefficient's value (kcal/mol for b)."""
    if df.empty:
        return None
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=bs.figsize('double', 0.5, 2))
    for ax, bias in zip(axes.flat, BIASES):
        _parity_panel(ax, df, bias, label_outliers=N_OUTLIERS)
    bs.label_panels(axes)
    return fig


def parity_system(df, system: str):
    """The same for one system's targets (all four coefficients)."""
    sub = df[df['system'] == system]
    if sub.empty:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=bs.figsize('double', 0.27))
    for ax, bias in zip(axes, BIASES):
        _parity_panel(ax, sub, bias, dense=False)
    fig.suptitle(system, fontweight='bold')
    bs.label_panels(axes)
    return fig


# embeddings

def substituent_table(records: List[Dict]):
    """One row per substituent (system, family, environment, site, sub,
    fragment) with its ligand embedding (mean over the runs that hold it)."""
    import pandas as pd
    rows, embs = {}, {}
    for r in records:
        for k, node in enumerate(r['nodes']):
            key = (r['system'], node['site'], node['fragment'] or f"sub{node['sub']}")
            embs.setdefault(key, []).append(r['ligand'][k])
            rows.setdefault(key, {'system': r['system'], 'family': r['family'],
                                  'environment': r['environment'], 'site': node['site'],
                                  'sub': node['sub'], 'fragment': node['fragment']})
    keys = list(rows)
    df = pd.DataFrame([rows[k] for k in keys])
    X = np.array([np.mean(embs[k], axis=0) for k in keys]) if keys else np.zeros((0, 512))
    return df, X


def pca(X: np.ndarray, n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """(scores [N, n], percent of variance per component [n])."""
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    n = min(n, len(s))
    return Xc @ vt[:n].T, 100 * var[:n] / var.sum()


def embedding_pca(subs, X):
    """Principal components of the substituents' ligand embeddings: PC1 vs
    PC2 (A) and PC1 vs PC3 (B), the most common families highlighted and
    the rest gray; percent of variance per component (C)."""
    if len(X) < 4:
        return None
    scores, var = pca(X)
    fams = subs['family'].value_counts().index[:N_FAMILIES_SHOWN].tolist()
    styles = dict(zip(fams, bs.styles(len(fams), 'marker')))
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=bs.figsize('double', 0.36),
                             gridspec_kw={'width_ratios': [1, 1, 0.8]})
    for ax, (i, j) in zip(axes[:2], ((0, 1), (0, 2))):
        ax._bowling_no_zero = True
        other = ~subs['family'].isin(fams).to_numpy()
        if other.any():
            ax.scatter(scores[other, i], scores[other, j], s=10, c=bs.MUTED, edgecolors=bs.INK,
                       linewidths=0.3, label='Other families', zorder=1)
        for f in fams:
            m = (subs['family'] == f).to_numpy()
            ax.scatter(scores[m, i], scores[m, j], s=16, linewidths=0.4, label=f, zorder=2,
                       **styles[f])
        ax.set_xlabel(f'PC{i + 1} ({var[i]:.0f}%)')
        ax.set_ylabel(f'PC{j + 1} ({var[j]:.0f}%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=min(len(labels), 6))
    ax = axes[2]
    k = np.arange(1, len(var) + 1)
    ax.bar(k, var, color=bs.PALETTE['lavender'], edgecolor=bs.INK)
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Variance explained (%)')
    ax.set_xticks(k[::2] if len(k) > 6 else k)
    bs.label_panels(axes)
    return fig


def system_distances(subs, X):
    """Distance between systems: for each substituent of system A, the
    distance to the nearest substituent of system B (ligand embeddings, each
    dimension standardized over all substituents), averaged over A's
    substituents and symmetrized. Returns (systems, matrix)."""
    if len(X) == 0:
        return [], np.zeros((0, 0))
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)
    # all substituent pairs at once: |a - b|^2 = |a|^2 + |b|^2 - 2 a.b
    sq = (Z ** 2).sum(axis=1)
    full = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0, None))
    systems = sorted(subs['system'].unique(), key=lambda s: (_family(subs, s), s))
    idx = {s: np.where(subs['system'].to_numpy() == s)[0] for s in systems}
    n = len(systems)
    D = np.zeros((n, n))
    for a in range(n):
        rows = full[idx[systems[a]]]
        for b in range(n):
            if a != b:
                D[a, b] = rows[:, idx[systems[b]]].min(axis=1).mean()
    return systems, (D + D.T) / 2


def _family(subs, system):
    return subs.loc[subs['system'] == system, 'family'].iloc[0]


def similarity_heatmap(systems: List[str], D: np.ndarray):
    """Distance between systems' substituents (``system_distances``); darker
    = more similar. Systems are ordered by family."""
    if len(systems) < 2:
        return None
    plt = _plt()
    n = len(systems)
    size = min(bs.WIDTHS['double'], 1.2 + 0.13 * n)
    fig, ax = plt.subplots(figsize=(size + 0.8, size))
    M = D.copy()
    np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap='bowling_purple_r', interpolation='nearest')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fs = max(4, min(bs.mpl.rcParams['font.size'], 140 / n))
    ax.set_xticklabels(systems, rotation=90, fontsize=fs)
    ax.set_yticklabels(systems, fontsize=fs)
    bs.colorbar(fig, im, ax, label='Embedding distance', shrink=0.8)
    if n <= 7:
        bs.annotate_cells(ax, np.nan_to_num(M), fmt='{:.1f}')
    return fig


# dataset statistics

def dataset_stats(stats: Dict, df):
    """Collected runs per dataset (A), substituents per run (B), sites per
    run (C) and the distribution of each scored coefficient (D-G)."""
    plt = _plt()
    fig = plt.figure(figsize=bs.figsize('double', 0.62))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0:2])
    per_ds = stats['runs_per_dataset']
    per = np.array(sorted(per_ds.values()))
    if 0 < len(per) < 10:
        names = sorted(per_ds, key=per_ds.get)
        ax.barh(range(len(names)), [per_ds[n] for n in names], color=bs.PALETTE['lavender'],
                edgecolor=bs.INK)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Runs')
    elif len(per):
        bins = np.unique(np.round(np.logspace(0, np.log10(max(per.max(), 2)), 20)))
        ax.hist(per, bins=bins, color=bs.PALETTE['lavender'], edgecolor=bs.INK)
        ax.set_xscale('log')
        ax.set_xlabel('Runs per dataset')
        ax.set_ylabel('Datasets')
        _count_axis(ax)
    axes = [ax]
    for col, key, lab in ((2, 'subs_per_run', 'Substituents per run'),
                          (3, 'sites_per_run', 'Sites per run')):
        ax = fig.add_subplot(gs[0, col])
        vals = stats[key]
        if vals:
            k = sorted(vals)
            ax.bar(k, [vals[v] for v in k], color=bs.PALETTE['lavender'], edgecolor=bs.INK)
            ax.set_xticks(k if len(k) <= 8 else k[::max(1, len(k) // 6)])
        ax.set_xlabel(lab)
        ax.set_ylabel('Runs')
        _count_axis(ax)
        axes.append(ax)
    for col, bias in enumerate(BIASES):
        ax = fig.add_subplot(gs[1, col])
        vals = df.loc[df['bias'] == bias, 'actual'].to_numpy() if not df.empty else []
        if len(vals):
            ax.hist(vals, bins='auto', color=bs.PALETTE['lavender'], edgecolor=bs.INK)
            _count_axis(ax)
        ax.set_xlabel(_axis_label('', bias))
        ax.set_ylabel('Targets' if col == 0 else '')
        axes.append(ax)
    bs.label_panels(axes)
    return fig


def nearest_systems(systems: List[str], D: np.ndarray, families: Dict[str, str],
                    k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
    """Each system's ``k`` nearest systems of other families."""
    out = {}
    for a, s in enumerate(systems):
        cand = [(systems[b], D[a, b]) for b in range(len(systems))
                if b != a and families.get(systems[b]) != families.get(s)]
        out[s] = sorted(cand, key=lambda t: t[1])[:k]
    return out
