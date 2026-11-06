"""Figures of an online-training run (``mllf.analysis.online``).

Every function returns a matplotlib figure, or None when the run has no data
for it; ``Output.figure`` saves it (``mllf.analysis.style``).
"""
from typing import Dict, List, Optional

import numpy as np

from mllf.analysis import style as bs
from mllf.analysis.common import system_order
from mllf.analysis.online_data import GROUP_OUTCOMES, PAIR_OUTCOMES


MLLF, ALF = 'MLLF', 'ALF'
_ROW_IN = 0.24          # inches per system row in horizontal bar charts


def _plt():
    import matplotlib.pyplot as plt
    return plt


# success over epochs

def success_over_epochs(epochs, validation: bool = True):
    """Percent of pair runs and group runs that passed their test per epoch,
    and of held-out validation pairs (optional). Pair runs are each judged by
    their own length's test; group runs by the stage's group test."""
    if epochs is None or epochs.empty:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=bs.figsize('single', 0.75))
    series = [('Pair runs', 'pairs_pct'), ('Group runs', 'groups_pct')]
    if validation:
        series.append(('Held-out pairs', 'validation_pct'))
    series = [(lab, col) for lab, col in series if col in epochs and epochs[col].notna().any()]
    if not series:
        return None
    for st, (lab, col) in zip(bs.styles(len(series), 'line'), series):
        d = epochs.dropna(subset=[col])
        ax.plot(d['epoch'], d[col], label=lab, **st)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Runs passing their test (%)')
    _int_axis(ax, 'x', epochs['epoch'].min(), epochs['epoch'].max())
    _legend_above(ax, ncol=len(series))
    return fig


# run outcomes

# Outcome fills, best first: success dark green, then lighter shades with a
# hatch each (no darkest purple next to the darkest green), so the outcomes
# stay apart in grayscale and for colour-blind readers.
_OUTCOME_FILLS = [
    {'color': bs.PALETTE['forest']},
    {'color': bs.PALETTE['orange']},
    {'color': bs.PALETTE['apricot'], 'hatch': '///'},
    {'color': bs.EXTENDED['plum_mid'], 'hatch': '...'},
    {'color': bs.PALETTE['lavender'], 'hatch': 'xxx'},
]


def _outcome_styles(order: List[str]) -> Dict[str, Dict]:
    fills = list(_OUTCOME_FILLS)
    if 'passed with previous run' in order:    # a second pass: the lighter green
        fills.insert(1, {'color': bs.PALETTE['sage'], 'hatch': '\\\\\\'})
    return {o: dict(fills[i], edgecolor=bs.INK) for i, o in enumerate(order)}


def outcomes_by_epoch(runs):
    """Number of training runs per epoch by outcome: pair runs (top) and
    group runs (bottom). Validation and scouting runs are left out."""
    if runs is None or runs.empty:
        return None
    train = runs[~runs['role'].isin(['validation', 'scouting'])]
    panels = [(lab, train[train['test'] == t], order)
              for lab, t, order in (('Pair runs', 'pair', PAIR_OUTCOMES),
                                    ('Group runs', 'group', GROUP_OUTCOMES))
              if (train['test'] == t).any()]
    if not panels:
        return None
    plt = _plt()
    fig, axes = plt.subplots(len(panels), 1, figsize=bs.figsize('onehalf', 0.42, len(panels)),
                             sharex=True, squeeze=False)
    epochs = sorted(train['epoch'].unique())
    for ax, (lab, sub, order) in zip(axes[:, 0], panels):
        t = (sub.groupby(['epoch', 'outcome']).size().unstack(fill_value=0)
             .reindex(epochs, fill_value=0))
        styles = _outcome_styles(order)
        bottom = np.zeros(len(t))
        for o in order:
            if o in t:
                ax.bar(t.index, t[o], bottom=bottom, width=0.8, label=o, **styles[o])
                bottom += t[o].values
        ax.set_title(lab)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        _int_axis(ax, 'y', 0, float(bottom.max()) if len(bottom) else 1)
    axes[-1, 0].set_xlabel('Epoch')
    _int_axis(axes[-1, 0], 'x', min(epochs), max(epochs), pad=0.6)
    bs.shared_label(fig, axes[:, 0], 'Runs')
    return fig


def outcomes_by_system(runs, env: Dict[str, str]):
    """Number of training runs per system by outcome: pair runs (left) and
    group runs (right)."""
    if runs is None or runs.empty:
        return None
    train = runs[~runs['role'].isin(['validation', 'scouting'])]
    panels = [(lab, train[train['test'] == t], order)
              for lab, t, order in (('Pair runs', 'pair', PAIR_OUTCOMES),
                                    ('Group runs', 'group', GROUP_OUTCOMES))
              if (train['test'] == t).any()]
    if not panels:
        return None
    systems = system_order(train['system'], env)
    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                             figsize=(bs.WIDTHS['double'], _ROW_IN * len(systems) + 2.0))
    y = np.arange(len(systems))
    for ax, (lab, sub, order) in zip(axes[0], panels):
        t = (sub.groupby(['system', 'outcome']).size().unstack(fill_value=0)
             .reindex(systems, fill_value=0))
        styles = _outcome_styles(order)
        left = np.zeros(len(t))
        for o in order:
            if o in t:
                ax.barh(y, t[o], left=left, height=0.75, label=o, **styles[o])
                left += t[o].values
        ax.set_xlabel('Runs')
        _int_axis(ax, 'x', 0, float(left.max()) if len(left) else 1)
        _legend_above(ax, ncol=2, title=lab)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(systems)
    axes[0, 0].invert_yaxis()
    return fig


# stage timeline

def _lighten(color, amount: float = 0.55) -> str:
    """``color`` blended toward white (the fill of a hatched, unconverged bar)."""
    from matplotlib.colors import to_hex, to_rgb
    return to_hex([c + (1 - c) * amount for c in to_rgb(color)])


def stage_timeline(stage_rows, stage_state: Dict[str, Dict], stage_names: List[str],
                   env: Dict[str, str], group_totals: Optional[Dict[str, int]] = None):
    """Per system, the epochs it spent in each stage: solid where the system
    met the stage's criterion (converged), lighter and hatched where it did
    not (stopped at the stage's epoch limit, or still running). A diamond
    marks a system that converged on its scouting runs, before epoch 0. The
    right axis gives the groups passed."""
    if stage_rows is None or stage_rows.empty:
        return None
    # systems that converged on their scouting runs never ran an epoch in a stage
    scouted = {s for s, st in stage_state.items()
               if st.get('done') and s not in set(stage_rows['system'])}
    systems = system_order(set(stage_rows['system']) | scouted, env)
    names = [n for n in stage_names if n in set(stage_rows['stage'])] or \
        sorted(set(stage_rows['stage']))
    fills = (bs.styles(len(names), 'fill', ordered=True) if len(names) > 1
             else [{'color': bs.PALETTE['lavender'], 'edgecolor': bs.INK}])
    color = dict(zip(names, fills))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(bs.WIDTHS['onehalf'], _ROW_IN * len(systems) + 1.6))
    last = int(stage_rows['epoch'].max())
    for y, s in enumerate(systems):
        rows = stage_rows[stage_rows['system'] == s].sort_values('epoch')
        for stage, seg in rows.groupby((rows['stage'] != rows['stage'].shift()).cumsum()):
            name = seg['stage'].iloc[0]
            lo, hi = seg['epoch'].min(), seg['epoch'].max()
            face = color.get(name, fills[0])['color']
            # the system's stage history records why each stage ended (the
            # log's stage-change line can be missing)
            hist = [h for h in (stage_state.get(s) or {}).get('history') or []
                    if h.get('stage') == name]
            converged = (any(h.get('reason') == 'criterion' for h in hist)
                         or (seg['reason'] == 'criterion').any())
            ax.barh(y, hi - lo + 1, left=lo - 0.5, height=0.7, edgecolor=bs.INK,
                    color=face if converged else _lighten(face),
                    hatch=None if converged else '///')
        if s in scouted:
            ax.plot(-0.5, y, marker='D', ms=6, mec=bs.INK, mfc=bs.INK, ls='', clip_on=False,
                    zorder=4)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems)
    ax.set_ylim(len(systems) - 0.5, -0.5)
    first = int(min(stage_rows['epoch'].min(), 0))
    _int_axis(ax, 'x', first, last, pad=0.5)
    ax.set_xlabel('Epoch')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(label=f'{n}, converged' if len(names) == 1 else n,
                     facecolor=color[n]['color'], edgecolor=bs.INK) for n in names]
    handles.append(Patch(label='Not converged', hatch='///', edgecolor=bs.INK,
                         facecolor=_lighten(color[names[0]]['color'])))
    if scouted:
        handles.append(Line2D([], [], marker='D', ls='', mfc=bs.INK, mec=bs.INK,
                              label='Converged by scouting'))
    _legend_above(ax, handles=handles, ncol=min(len(handles), 3))
    if group_totals:
        right = ax.twinx()
        right.set_ylim(ax.get_ylim())
        right.set_yticks(range(len(systems)))
        right.set_yticklabels(
            [f"{len((stage_state.get(s) or {}).get('groups_done') or {})}/{group_totals[s]}"
             if group_totals.get(s) else '-' for s in systems])
        right.set_ylabel('Groups passed')
    return fig


# convergence vs ALF

def _ns_bars(ax, y, values, converged, color, light, height, offset):
    """Horizontal ns bars: solid when converged, light and hatched when not."""
    values = np.nan_to_num(np.asarray(values, float), nan=0.0)
    conv = np.asarray([bool(c) if c is not None and c == c else False for c in converged])
    ax.barh(y[conv] + offset, values[conv], height=height, color=color, edgecolor=bs.INK)
    ax.barh(y[~conv] + offset, values[~conv], height=height, color=light, edgecolor=bs.INK,
            hatch='///')


def _convergence_legend(ax, **kw):
    from matplotlib.patches import Patch
    P = bs.PALETTE
    handles = [Patch(facecolor=P['plum'], edgecolor=bs.INK, label=f'{MLLF}, converged'),
               Patch(facecolor=P['lavender'], edgecolor=bs.INK, hatch='///',
                     label=f'{MLLF}, not converged'),
               Patch(facecolor=P['orange'], edgecolor=bs.INK, label=f'{ALF}, converged'),
               Patch(facecolor=P['apricot'], edgecolor=bs.INK, hatch='///',
                     label=f'{ALF}, not converged')]
    return ax.legend(handles=handles, **kw)


def convergence_systems(sys_df, env: Dict[str, str]):
    """Per system, simulated ns to convergence for MLLF and ALF. Solid bars:
    converged (every group passed); hatched bars: never converged, showing
    every ns that method ran. ALF bars cover the same systems' ALF datasets."""
    if sys_df is None or sys_df.empty:
        return None
    df = sys_df.set_index('system').loc[system_order(sys_df['system'], env)]
    has_alf = df['alf_ns'].notna().any()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(bs.WIDTHS['onehalf'], 0.34 * len(df) + 1.6))
    y = np.arange(len(df))
    P = bs.PALETTE
    h = 0.38 if has_alf else 0.7
    _ns_bars(ax, y, df['mllf_ns'], df['mllf_converged'], P['plum'], P['lavender'], h,
             -h / 2 if has_alf else 0)
    if has_alf:
        _ns_bars(ax, y, df['alf_ns'], df['alf_converged'], P['orange'], P['apricot'], h, h / 2)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_ylim(len(df) - 0.5, -0.5)
    ax.set_xlabel('Simulated time (ns)')
    bs.value_grid(ax, axis='x')
    _convergence_legend(ax, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    _value_labels(ax, y - h / 2 if has_alf else y, df['mllf_ns'])
    if has_alf:
        _value_labels(ax, y + h / 2, df['alf_ns'])
    return fig


def _value_labels(ax, y, values) -> None:
    """The value at the end of each bar (ns, one decimal), with 15% headroom
    on the axis for the labels."""
    vmax = np.nanmax(np.asarray(values, float)) if len(values) else np.nan
    if vmax == vmax and vmax > 0:
        ax.update_datalim([[1.15 * vmax, y[0]]])
        ax.autoscale_view()
    for yy, v in zip(y, values):
        if v == v and v is not None:
            ax.annotate(f'{v:.1f}', xy=(v, yy), xytext=(3, 0), textcoords='offset points',
                        va='center', fontsize=bs.mpl.rcParams['font.size'] - 1)


PANELS_PER_PAGE = 12


def convergence_group_pages(grp_df, env: Dict[str, str]) -> List[List[str]]:
    """The systems of each page of ``convergence_groups`` (PANELS_PER_PAGE each)."""
    if grp_df is None or grp_df.empty:
        return []
    systems = system_order(grp_df['system'], env)
    return [systems[k:k + PANELS_PER_PAGE] for k in range(0, len(systems), PANELS_PER_PAGE)]


def convergence_groups(grp_df, sys_df, env: Dict[str, str], systems: Optional[List[str]] = None,
                       ncols: int = 3):
    """One panel per system (``systems``, default all): per group, simulated
    ns to pass the group test for MLLF (its group runs plus its share of its
    pair runs) and ALF. Solid bars: passed; hatched: never passed (every ns
    used). Each panel has its own x-axis, as the comparison is within a system."""
    if grp_df is None or grp_df.empty:
        return None
    systems = systems or system_order(grp_df['system'], env)
    if not systems:
        return None
    nrows = int(np.ceil(len(systems) / ncols))
    counts = grp_df.groupby('system').size()
    heights = [max(counts[s] for s in systems[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
    plt = _plt()
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(bs.WIDTHS['double'],
                                      sum(0.3 * h + 0.9 for h in heights) + 0.6),
                             gridspec_kw={'height_ratios': heights})
    P = bs.PALETTE
    conv = ({} if sys_df is None or sys_df.empty
            else sys_df.set_index('system')['mllf_converged'])
    for k, (ax, s) in enumerate(zip(axes.flat, systems)):
        g = grp_df[grp_df['system'] == s].sort_values('group')
        slots = heights[k // ncols]          # same bar thickness across a row
        y = np.arange(len(g))
        has_alf = g['alf_ns'].notna().any()
        h = 0.38 if has_alf else 0.7
        _ns_bars(ax, y, g['mllf_ns'], g['mllf_passed'], P['plum'], P['lavender'], h,
                 -h / 2 if has_alf else 0)
        if has_alf:
            _ns_bars(ax, y, g['alf_ns'], g['alf_passed'], P['orange'], P['apricot'], h, h / 2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"G{int(r)}" for r in g['group']])
        ax.set_ylim(slots - 0.5, -0.5)
        n_pass = int(g['mllf_passed'].sum())
        ax.set_title(f"{s} ({n_pass}/{len(g)}" + (', converged)' if conv.get(s) else ')'),
                     fontsize=bs.mpl.rcParams['font.size'])
    for ax in list(axes.flat)[len(systems):]:
        ax.set_visible(False)
    bs.shared_label(fig, [a for a in axes.flat if a.get_visible()], 'Simulated time (ns)',
                    axis='x')
    _convergence_legend(fig, loc='outside upper center', ncol=4)
    fig.get_layout_engine().set(h_pad=0.08)     # room between legend and titles
    return fig


def _int_axis(ax, which: str, lo: float, hi: float, pad: float = 0.0) -> None:
    """Integer ticks (epochs, run counts) with limits on a tick (plus ``pad``,
    for bars centred on the end ticks); the axis is then left unsnapped."""
    from matplotlib.ticker import MultipleLocator
    span = max(hi - lo, 1)
    step = next(s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 10 ** 9)
                if span / s <= 6)
    lo2, hi2 = np.floor(lo / step) * step, np.ceil(hi / step) * step
    if hi2 == lo2 and pad == 0:
        hi2 = lo2 + step              # a count axis needs a range
    (ax.set_xlim if which == 'x' else ax.set_ylim)(lo2 - pad, hi2 + pad)
    (ax.xaxis if which == 'x' else ax.yaxis).set_major_locator(MultipleLocator(step))


def _legend_above(ax, **kw):
    """A legend centred above the axes (outside the data, whatever its height)."""
    kw.setdefault('loc', 'lower center')
    kw.setdefault('bbox_to_anchor', (0.5, 1.02))
    return ax.legend(**kw)
