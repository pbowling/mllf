"""``mllf production analyze``: end-state free energies of a production.

Reads each replica's lambda file (``res/*_flat.lmd``), drops the first
``discard_ns`` of production (a quarter by default) and estimates the free
energy of every combination by WHAM and by a Potts model
(``mllf.analysis.estimators``), with bootstrap uncertainties over the
replicas. Writes, under ``<production>/analysis``:

* ``tables/free_energies.csv``: per combination, frames, G by WHAM and by
  the Potts model with their SDs, and each replica's own histogram value
* ``tables/sampling.csv`` and ``tables/replicas.csv``: endpoint fractions,
  entries and transitions per ns, per replica
* ``tables/potts_h.csv`` / ``potts_J.csv``: the Potts model (kT units), from
  which the free energy of any combination can be rebuilt
* ``figures/populations`` (frames at each substituent's end state, the
  replicas stacked), ``figures/lambda_trajectories`` (lambda over time, per
  replica and site), ``figures/profiles`` (the biased lambda profiles of
  each site, flat when the biases are good) and, for two or more sites,
  ``figures/estimator_parity``
* ``report.txt``
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from mllf.analysis import estimators as est
from mllf.analysis.common import Output, table_text


log = logging.getLogger(__name__)

FIGURES = ('populations', 'lambda_trajectories', 'end_state_occupancy', 'profiles',
           'estimator_parity')
MIN_FRACTION = 0.01              # endpoint fraction flagged below this
MIN_ENTRIES_PER_NS = 2.0         # entries per ns flagged below this


def _system_names(rep_dir: Path, nsubs: Sequence[int]) -> Optional[List[List[str]]]:
    """Each substituent's number in the full system, from mapping.json."""
    p = rep_dir / 'mapping.json'
    if not p.is_file():
        return None
    try:
        rows = json.loads(p.read_text())
    except ValueError:
        return None
    names = [[str(j + 1) for j in range(n)] for n in nsubs]
    for r in rows:
        try:
            names[int(r['new_site']) - 1][int(r['new_sub']) - 1] = str(r['original_sub'])
        except (KeyError, IndexError, ValueError):
            continue
    return names


def load_replicas(out_dir: Path, nsubs: Sequence[int], discard_ns: float, stride: int = 1):
    """[replica dicts], notes. Each: 'name', 'L', 'terms', 'dt_ps', 'ns_total',
    'ns_used', 'temperature'."""
    from mllf.file_handling.read_bias_coeff import parse_new
    from mllf.file_handling.read_lambda import read_lmd
    reps, notes = [], []
    for d in sorted(Path(out_dir).glob('replica_*')):
        lmd = sorted((d / 'res').glob('*_flat.lmd'))
        if not lmd:
            notes.append(f"{d.name}: no lambda file; left out")
            continue
        try:
            L, hdr = read_lmd(lmd[0])
        except (ValueError, OSError) as e:
            notes.append(f"{d.name}: {lmd[0].name} not read ({e}); left out")
            continue
        if L.shape[1] != sum(nsubs):
            notes.append(f"{d.name}: {L.shape[1]} substituent blocks, expected {sum(nsubs)}; "
                         "left out")
            continue
        skip = int(round(discard_ns * 1000.0 / hdr['dt_ps']))
        if skip >= len(L):
            notes.append(f"{d.name}: {len(L) * hdr['dt_ps'] / 1000:.2f} ns, all within the "
                         f"{discard_ns:g} ns discarded; left out")
            continue
        bias = parse_new(str(d / 'variables.py'))
        reps.append({'name': d.name, 'L': L[skip::stride], 'terms': est.bias_terms(bias, nsubs),
                     'dt_ps': hdr['dt_ps'] * stride, 'temperature': hdr['temperature'],
                     'ns_total': len(L) * hdr['dt_ps'] / 1000.0,
                     'ns_used': (len(L) - skip) * hdr['dt_ps'] / 1000.0,
                     'stopped': len(L) < hdr['frames_in_header']})
    for a in range(len(reps)):
        for b in range(a + 1, len(reps)):
            n = min(len(reps[a]['L']), len(reps[b]['L']), 2000)
            if n and np.array_equal(reps[a]['L'][:n], reps[b]['L'][:n]):
                notes.append(f"{reps[a]['name']} and {reps[b]['name']} have identical "
                             "trajectories (same random seed); they are not independent")
    return reps, notes


def analyze_production(out_dir, out=None, discard_ns: Optional[float] = None,
                       cutoff: Optional[float] = None, n_boot: Optional[int] = None,
                       stride: Optional[int] = None, potts: Optional[bool] = None,
                       fmt: str = 'png', only=None) -> Path:
    """Analyze the production in ``out_dir``; returns the report's path.
    Unset options come from the production's ``analysis:`` settings."""
    import pandas as pd

    from mllf.analysis import style as bs
    from mllf.production import DISCARD_FRACTION, load_config
    out_dir = Path(out_dir)
    cfg = load_config(out_dir)
    a = cfg.get('analysis') or {}
    nsubs = [int(n) for n in cfg['nsubs']]
    discard_ns = (discard_ns if discard_ns is not None else a.get('discard_ns'))
    if discard_ns is None:
        discard_ns = DISCARD_FRACTION * float(cfg['ns'])
    cutoff = float(cutoff if cutoff is not None else a.get('cutoff', 0.99))
    n_boot = int(n_boot if n_boot is not None else a.get('bootstrap', 50))
    stride = int(stride if stride is not None else a.get('stride', 1))
    potts = bool(potts if potts is not None else a.get('potts', True))

    bs.apply_style()
    output = Output(Path(out or out_dir / 'analysis'), only=only, fmt=fmt)
    reps, notes = load_replicas(out_dir, nsubs, discard_ns, stride)
    if not reps:
        raise ValueError(f"{out_dir}: no replica with frames to analyze"
                         + ('; ' + '; '.join(notes) if notes else ''))
    temperature = float(cfg.get('temperature') or reps[0]['temperature'])
    names = _system_names(out_dir / reps[0]['name'], nsubs)
    log.info(f"{len(reps)} replicas, {sum(r['ns_used'] for r in reps):.2f} ns after "
             f"discarding {discard_ns:g} ns of each")

    res = est.analyze_replicas(reps, nsubs, temperature, cutoff=cutoff, n_boot=n_boot,
                               potts=potts)
    samp = [est.sampling(r['L'], nsubs, r['dt_ps'], cutoff) for r in reps]

    # tables
    combos = res['combos']
    df = pd.DataFrame({f'site{i + 1}': combos[:, i] for i in range(len(nsubs))})
    if names:
        for i in range(len(nsubs)):
            df[f'site{i + 1}_system'] = [names[i][s - 1] for s in combos[:, i]]
    df['label'] = [est.combo_label(c, names) for c in combos]
    df['frames'] = res['counts'].astype(int)
    df['G_wham'] = res['wham']
    df['G_wham_sd'] = res['wham_sd']
    if res['potts'] is not None:
        df['G_potts'] = res['potts']
        df['G_potts_sd'] = res['potts_sd']
    for k, r in enumerate(reps):
        df[f'G_{r["name"]}'] = res['replica_g'][k]
    output.table('free_energies', df)
    rows = []
    for r, s in zip(reps, samp):
        for i, n in enumerate(nsubs):
            for j in range(n):
                rows.append({'replica': r['name'], 'site': i + 1, 'sub': j + 1,
                             'system_sub': names[i][j] if names else j + 1,
                             'fraction': s['fraction'][i][j],
                             'entries_per_ns': s['entries_per_ns'][i][j]})
    output.table('sampling', pd.DataFrame(rows))
    output.table('replicas', pd.DataFrame([
        {'replica': r['name'], 'ns_run': r['ns_total'], 'ns_analyzed': r['ns_used'],
         'stopped_early': r['stopped'], 'physical_fraction': s['physical'],
         **{f'site{i + 1}_transitions_per_ns': t for i, t in enumerate(s['transitions_per_ns'])}}
        for r, s in zip(reps, samp)]))
    if res['potts_model'] is not None:
        h, J = est.potts_tables(res['potts_model'])
        output.table('potts_h', pd.DataFrame(h, columns=['site', 'state', 'h']))
        output.table('potts_J', pd.DataFrame(J, columns=['site_i', 'state_i', 'site_j',
                                                         'state_j', 'J']))

    # figures
    output.figure('populations', lambda: population_figure(reps, nsubs, names, cutoff))
    output.figure('lambda_trajectories',
                  lambda: trajectory_figure(reps, nsubs, names, discard_ns, cutoff))
    output.figure('end_state_occupancy',
                  lambda: occupancy_figure(reps, nsubs, names, discard_ns, cutoff))
    output.figure('profiles', lambda: profile_figure(reps, nsubs, names))
    if len(nsubs) > 1:
        output.figure('estimator_parity', lambda: parity_figure(df))

    report = report_text(out_dir, cfg, reps, samp, res, df, names, notes, discard_ns, cutoff,
                         n_boot, stride, output)
    path = output.report(report)
    log.info(f"Report: {path}")
    return path


# Report

def report_text(out_dir, cfg, reps, samp, res, df, names, notes, discard_ns, cutoff, n_boot,
                stride, output) -> str:
    nsubs = [int(n) for n in cfg['nsubs']]
    ref = df.iloc[res['reference']]
    lines = [f"Production analysis: {out_dir}", '',
             'Inputs', '------',
             f"Prep: {cfg.get('prep')}", f"Script: {cfg.get('script')}",
             f"Variables: {cfg.get('variables')}",
             f"Sites: {len(nsubs)}, substituents {nsubs}; temperature {cfg.get('temperature')} K",
             '', 'Sampling', '--------',
             f"Replicas analyzed: {len(reps)} of {cfg.get('replicas')}; first {discard_ns:g} ns "
             f"of each discarded; {sum(r['ns_used'] for r in reps):.2f} ns analyzed"
             + (f" (every {stride}th frame)" if stride > 1 else '')
             + f"; end state: lambda >= {cutoff:g}"]
    rows = []
    low = []
    for r, s in zip(reps, samp):
        fr = [f for site in s['fraction'] for f in site]
        en = [e for site in s['entries_per_ns'] for e in site]
        rows.append([r['name'], f"{r['ns_used']:.2f}" + (' (stopped)' if r['stopped'] else ''),
                     f"{100 * s['physical']:.1f}",
                     ', '.join(f"{t:.1f}" for t in s['transitions_per_ns']),
                     f"{100 * min(fr):.2f}", f"{min(en):.1f}"])
        for i, n in enumerate(nsubs):
            for j in range(n):
                if (s['fraction'][i][j] < MIN_FRACTION
                        or s['entries_per_ns'][i][j] < MIN_ENTRIES_PER_NS):
                    sub = names[i][j] if names else str(j + 1)
                    low.append(f"{r['name']} site {i + 1} sub {sub}")
    lines += [table_text(rows, ['Replica', 'ns', 'Physical %', 'Transitions / ns (per site)',
                                'Min endpoint %', 'Min entries / ns'])]
    if low:
        lines.append(f"Below {100 * MIN_FRACTION:g}% of frames or {MIN_ENTRIES_PER_NS:g} entries "
                     f"/ ns ({len(low)}): " + '; '.join(low[:30])
                     + (' ...' if len(low) > 30 else ''))
    lines += ['', 'Free energies', '-------------',
              f"Relative to {ref['label']} (kcal/mol); uncertainty: SD over {n_boot} bootstrap "
              "samples of the replicas" + (' (one replica: none)' if len(reps) < 2 else '')]
    if res['bias_sets'] > 1:
        lines.append(f"The replicas ran with {res['bias_sets']} different bias sets; "
                     "WHAM combines them")
    if res['potts'] is None:
        lines.append('Potts model: not fitted' + (f" ({res['potts_note']})"
                                                  if res['potts_note'] else ''))
    has_p = 'G_potts' in df
    show = df if len(df) <= 60 else df.sort_values('G_wham').head(60)
    rows = []
    for r in show.itertuples():
        row = [r.label, r.frames, _pm(r.G_wham, r.G_wham_sd)]
        if has_p:
            row.append(_pm(r.G_potts, r.G_potts_sd))
        rows.append(row)
    header = ['Combination' + (' (system subs)' if names else ''), 'Frames', 'WHAM']
    if has_p:
        header.append('Potts')
    if len(df) > 60:
        lines.append(f"{len(df)} combinations; the 60 lowest by WHAM are listed "
                     "(all in tables/free_energies.csv)")
    lines += [table_text(rows, header)]
    unsampled = int(np.sum(~np.isfinite(df['G_wham'])))
    if unsampled:
        lines.append(f"Never sampled: {unsampled} of {len(df)} combinations (no WHAM value"
                     + (', Potts value where each substituent and pair was sampled)'
                        if has_p else ')'))
    if notes:
        lines += ['', 'Notes', '-----'] + notes
    lines += ['', 'Outputs', '-------', f"Figures: {output.figures} ({len(output.written)})",
              f"Tables: {output.tables}"]
    for name, why in output.skipped.items():
        lines.append(f"  {name} not drawn: {why}")
    return '\n'.join(lines) + '\n'


def _pm(v, sd) -> str:
    if v is None or not np.isfinite(v):
        return '-'
    return f"{v:7.2f}" + (f" +/- {sd:.2f}" if sd is not None and np.isfinite(sd) else '')


# Figures

def _runs(mask: np.ndarray):
    """(start, length) of each run of True in mask, in frames."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    return starts, np.flatnonzero(d == -1) - starts


def occupancy_figure(reps, nsubs, names, discard_ns, cutoff=0.99):
    """When each substituent is at its end state (lambda >= the cutoff): one
    lane per substituent, lit while it is there, and a gray lane below for
    frames with no substituent at the cutoff. One row per replica (the first
    eight), one column per site."""
    import matplotlib.pyplot as plt

    from mllf.analysis import free_energy as fe
    from mllf.analysis import style as bs
    shown = reps[:MAX_TRACE_REPLICAS]
    n = len(nsubs)
    lane_h = 0.17
    fig, axes = plt.subplots(len(shown), n, squeeze=False, sharex=True,
                             figsize=(bs.WIDTHS['double'] if n > 1 else bs.WIDTHS['onehalf'] + 1.4,
                                      0.7 + len(shown) * (0.45 + lane_h * (max(nsubs) + 1))))
    for k, r in enumerate(shown):
        st = est.frame_states(r['L'], nsubs, cutoff)
        dt = r['dt_ps'] / 1000.0
        for i in range(n):
            ax = axes[k, i]
            lanes = [(0, 'none', bs.MUTED)] + [
                (j + 1, f"{names[i][j] if names else j + 1}", fe._sub_style(j)['color'])
                for j in range(nsubs[i])]
            for y, (state, _, color) in enumerate(lanes):
                starts, lengths = _runs(st[:, i] == state)
                ax.broken_barh(list(zip(discard_ns + starts * dt, lengths * dt)), (y - 0.4, 0.8),
                               facecolors=color, edgecolors='none')
            ax.set_yticks(range(len(lanes)))
            ax.set_yticklabels([lab for _, lab, _ in lanes],
                               fontsize=bs.mpl.rcParams['font.size'] - 2)
            ax.set_ylim(-0.6, len(lanes) - 0.4)
            ax.set_xlim(discard_ns, discard_ns + len(r['L']) * dt)
            ax.tick_params(axis='y', length=0)
            ax._bowling_no_snap = True
            rep = r['name'].replace('replica_', 'Replica ')
            ax.set_title(f'{rep}, site {i + 1}' if n > 1 else rep, loc='left',
                         fontsize=bs.mpl.rcParams['font.size'] - 1)
            if i == 0:
                ax.set_ylabel(f'At λ ≥ {cutoff:g}')
    for ax in axes[-1]:
        ax.set_xlabel('Time (ns)')
    if len(reps) > len(shown):
        fig.text(0.99, 0.0, f'first {len(shown)} of {len(reps)} replicas', ha='right', va='bottom',
                 fontsize=bs.mpl.rcParams['font.size'] - 2)
    return fig


def _replica_fills(n):
    """Bar styles for n replicas: the house fills, hatched again past six."""
    from mllf.analysis import style as bs
    hatches = ['', '///', '...', 'xxx']
    out = []
    for i in range(n):
        d = {'color': bs.FILL_COLORS[i % len(bs.FILL_COLORS)], 'edgecolor': bs.INK}
        h = bs.HATCHES[i % len(bs.HATCHES)] if i < len(bs.FILL_COLORS) else \
            hatches[(i // len(bs.FILL_COLORS)) % len(hatches)]
        if h:
            d['hatch'] = h
        out.append(d)
    return out


def population_figure(reps, nsubs, names, cutoff):
    """Frames at each substituent's end state, one bar per substituent with
    each replica's frames stacked; one panel per site. The dashed line is the
    height every bar would have with the site's end-state frames split
    evenly, so a short bar is an undersampled substituent and a bar of one
    color a substituent only one replica reached."""
    import matplotlib.pyplot as plt

    from mllf.analysis import style as bs
    n = len(nsubs)
    o = est.offsets(nsubs)
    counts = np.array([[np.sum(r['L'][:, o[i]:o[i] + nsubs[i]] >= cutoff, axis=0)
                        for i in range(n)] for r in reps], dtype=object)
    widths = [max(1.6, 0.35 * k + 0.8) for k in nsubs]
    fig_w = min(bs.WIDTHS['double'], sum(widths) + 1.6)
    fig, axes = plt.subplots(1, n, squeeze=False, sharey=True, figsize=(fig_w, 2.6),
                             gridspec_kw={'width_ratios': widths})
    fills = _replica_fills(len(reps))
    for i in range(n):
        ax = axes[0, i]
        x = np.arange(nsubs[i])
        bottom = np.zeros(nsubs[i])
        for k, r in enumerate(reps):
            c = np.asarray(counts[k, i], float) / 1000.0
            ax.bar(x, c, bottom=bottom, width=0.75, label=r['name'].replace('replica_', 'Replica '),
                   **fills[k])
            bottom += c
        ax.axhline(bottom.mean(), color=bs.INK, ls='--', lw=1.0, label='Even split')
        ax.set_xticks(x)
        ax.set_xticklabels(names[i] if names else [str(j + 1) for j in x])
        ax.set_xlim(-0.6, nsubs[i] - 0.4)
        ax.set_xlabel('Substituent')
        if n > 1:
            ax.set_title(f'Site {i + 1}')
    axes[0, 0].set_ylabel(f'Frames at λ ≥ {cutoff:g} (thousands)')
    h, lab = axes[0, 0].get_legend_handles_labels()
    axes[0, -1].legend(h, lab, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    return fig


MAX_TRACE_POINTS = 4000           # per replica and substituent in lambda_trajectories
MAX_TRACE_REPLICAS = 8


END_FLOOR = 1e-4                  # 1 - lambda below this is drawn at the top of the end scale


def _end_scale(ax) -> None:
    """y = -log10(1 - lambda): 0, 0.9, 0.99 and 0.999 equally spaced, so the
    end state takes most of the height and the frames between end states
    share the bottom."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    def fwd(v):
        return -np.log10(np.clip(1.0 - np.asarray(v, float), END_FLOOR, None))

    def inv(u):
        return 1.0 - 10.0 ** (-np.asarray(u, float))
    ax.set_yscale('function', functions=(fwd, inv))
    ax.set_ylim(0.0, 1.0 - END_FLOOR)
    ticks = [0.0, 0.9, 0.99, 0.999]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def trajectory_figure(reps, nsubs, names, discard_ns, cutoff=0.99):
    """lambda of every substituent over the analyzed production, one row per
    replica (the first eight) and one column per site, on a scale that
    stretches the end state (``_end_scale``); the gray band is lambda >= the
    cutoff. Frequent crossings between substituents mean a
    well-mixed replica; a substituent absent for long stretches, or a replica
    that stays at one end state, shows up here before it shows in the
    averages."""
    import matplotlib.pyplot as plt

    from mllf.analysis import free_energy as fe
    from mllf.analysis import style as bs
    shown = reps[:MAX_TRACE_REPLICAS]
    n = len(nsubs)
    o = est.offsets(nsubs)
    fig, axes = plt.subplots(len(shown), n, squeeze=False, sharex=True, sharey=True,
                             figsize=(bs.WIDTHS['double'] if n > 1 else bs.WIDTHS['onehalf'] + 1.4,
                                      0.6 + 1.6 * len(shown)))
    for k, r in enumerate(shown):
        step = max(1, len(r['L']) // MAX_TRACE_POINTS)
        t = discard_ns + np.arange(0, len(r['L']), step) * r['dt_ps'] / 1000.0
        for i in range(n):
            ax = axes[k, i]
            for j in range(nsubs[i]):
                sub = names[i][j] if names else str(j + 1)
                ax.plot(t, r['L'][::step, o[i] + j], lw=0.6, alpha=0.85,
                        color=fe._sub_style(j)['color'], label=f'Sub {sub}')
            ax.axhspan(cutoff, 1.0, color=bs.BAND_GRAY, zorder=0, lw=0)
            _end_scale(ax)
            ax._bowling_no_snap = True
            if k == 0 and n > 1:
                ax.set_title(f'Site {i + 1}')
            if i == 0:
                ax.set_ylabel(r['name'].replace('replica_', 'Rep. '))
    for ax in axes[-1]:
        ax.set_xlabel('Time (ns)')
    for i in range(n):
        if nsubs[i] <= 12:
            h, lab = axes[0, i].get_legend_handles_labels()
            above = 1.02 if n == 1 else 1.12
            leg = axes[0, i].legend(h, lab, loc='lower left', bbox_to_anchor=(0, above),
                                    ncol=min(nsubs[i], 6), handlelength=1.2, columnspacing=0.8,
                                    fontsize=bs.mpl.rcParams['font.size'] - 2)
            for line in leg.get_lines():
                line.set_linewidth(2.0)
    fig.supylabel('λ', x=0.0)
    if len(reps) > len(shown):
        fig.text(0.99, 0.0, f'first {len(shown)} of {len(reps)} replicas', ha='right', va='bottom',
                 fontsize=bs.mpl.rcParams['font.size'] - 2)
    return fig


def profile_figure(reps, nsubs, names):
    """The biased 1D lambda profile of each substituent, all replicas
    pooled; one panel per site, one y range."""
    import matplotlib.pyplot as plt

    from mllf.analysis import free_energy as fe
    from mllf.analysis import style as bs
    L = np.concatenate([r['L'] for r in reps])
    n = len(nsubs)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharey=True,
                             figsize=(bs.WIDTHS['double'] if ncol > 1 else bs.WIDTHS['onehalf'],
                                      2.4 * nrow + 0.3))
    curves, used = [], []
    o = est.offsets(nsubs)
    for i in range(n):
        ax = axes.flat[i]
        used.append(ax)
        for j in range(nsubs[i]):
            x, g, _ = fe.profile(L[:, o[i] + j])
            curves.append(g - g.min())
            sub = names[i][j] if names else str(j + 1)
            ax.plot(x, curves[-1], label=f'Sub {sub}', **fe._sub_style(j))
        ax.set_xlim(0, 1)
        ax.set_xlabel('λ')
        ax.set_title(f'Site {i + 1}')
        if nsubs[i] <= 12:
            ax.legend(loc='upper center', ncol=2, fontsize=bs.mpl.rcParams['font.size'] - 2)
    for ax in axes.flat[n:]:
        ax.axis('off')
    axes[0, 0].set_ylabel('G (kcal/mol)')
    fe._shared_top(used, curves)
    return fig


def parity_figure(df):
    """Potts vs WHAM, the combinations with both."""
    import matplotlib.pyplot as plt

    from mllf.analysis import style as bs
    if 'G_potts' not in df:
        return None
    d = df[np.isfinite(df['G_wham']) & np.isfinite(df['G_potts'])]
    if len(d) < 2:
        return None
    fig, ax = plt.subplots(figsize=bs.figsize('single', aspect=1.0))
    ax.errorbar(d['G_wham'], d['G_potts'], xerr=d['G_wham_sd'], yerr=d['G_potts_sd'],
                ls='none', marker='o', ms=3, color=bs.STRONG[0])
    bs.parity(ax, d['G_wham'], d['G_potts'])
    ax.set_xlabel('ΔG, WHAM (kcal/mol)')
    ax.set_ylabel('ΔG, Potts (kcal/mol)')
    return fig
