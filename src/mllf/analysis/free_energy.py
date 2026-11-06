"""Free-energy profiles of a group's run, MLLF next to ALF's best run of the
same group (as ALF's PlotFreeEnergy draws them).

Each run's own lambda trajectory gives the biased-ensemble profile
G = -kT ln P (a pseudocount fills empty bins), from the LAMBDA lines of its
output only. A well-flattened run has flat profiles with similar endpoints.

* 1D profiles: G(lambda_i) for every substituent of a site.
* Transition profiles: for each pair i, j, the frames on the pair's edge
  (lambda_i + lambda_j >= 0.9), at x = lambda_i / (lambda_i + lambda_j), so
  x = 1 has substituent i on.

ALF's best run of a group is its 1 ns run with the most pairs passing the
full-run test, ties broken by total transitions.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mllf.analysis import style as bs


log = logging.getLogger(__name__)

KT = 0.001987 * 298.15          # kcal/mol at 298.15 K
NBINS = 40
EDGE = 0.9
PSEUDO = 0.5


def profile(values, lo: float = 0.0, hi: float = 1.0):
    """(bin centres, G in kcal/mol, counts)."""
    h, edges = np.histogram(values, bins=NBINS, range=(lo, hi))
    g = -KT * np.log((h + PSEUDO) / (h.sum() + PSEUDO * NBINS))
    return (edges[:-1] + edges[1:]) / 2, g, h


def _site_columns(labels: List[Tuple[int, int]]):
    sites: Dict[int, List[Tuple[int, int]]] = {}
    for col, (site, sub) in enumerate(labels):
        sites.setdefault(site, []).append((col, sub))
    return sorted(sites.items())


def run_profiles(output_file: Path, labels: List[Tuple[int, int]]):
    """{site: ({sub: profile}, {(sub i, sub j): profile or None})}, frames."""
    from mllf.file_handling.read_output import parse_lambda_trajectory
    _, frames = parse_lambda_trajectory(Path(output_file).read_text(errors='replace'))
    F = np.array(frames)
    if F.ndim != 2 or F.shape[1] < len(labels):
        raise ValueError(f"{output_file}: {F.shape} lambda frames for {len(labels)} substituents")
    out = {}
    for site, cols in _site_columns(labels):
        one_d = {sub: profile(F[:, c]) for c, sub in cols}
        trans = {}
        for k, (ci, si) in enumerate(cols):
            for cj, sj in cols[k + 1:]:
                s = F[:, ci] + F[:, cj]
                on = s >= EDGE
                trans[(si, sj)] = profile(F[on, ci] / s[on]) if on.sum() >= 20 else None
        out[site] = (one_d, trans)
    return out, len(F)


def combo_labels(name: str) -> List[Tuple[int, int]]:
    labels = []
    for m in re.finditer(r'site(\d+)_subs_([\d_]+?)(?=__site|$)', name):
        labels += [(int(m.group(1)), int(x)) for x in m.group(2).strip('_').split('_')]
    return labels


def pairs_passing(sim_results: Dict, nsubs: List[int]) -> Tuple[int, int]:
    """(pairs passing the success test for the run's length, pairs)."""
    import torch

    from mllf.cb.graph_utils import build_directed_pairs
    from mllf.pretrain.rewards import run_pair_success
    pairs = build_directed_pairs(nsubs)
    ok = run_pair_success({'sim_results': sim_results}, torch.tensor(pairs).T, {})
    if ok is None:
        return 0, 0
    half = [k for k, (i, j) in enumerate(pairs) if i < j]
    return int(sum(bool(ok[k]) for k in half)), len(half)


def best_alf_run(dataset: Path, cache: Optional[Dict] = None) -> Optional[Dict]:
    """ALF's best full-length run of a group dataset (module docstring),
    remembered in ``cache`` (keyed by the dataset path)."""
    from mllf.pretrain.rewards import run_length
    key = str(Path(dataset).resolve())
    if cache is not None and key in cache:
        return dict(cache[key]) if cache[key] else None
    best = None
    runs = (p for p in Path(dataset).iterdir() if (p / 'simulation_results.json').exists())
    for r in sorted(runs):
        sr = json.loads((r / 'simulation_results.json').read_text())
        if run_length(sr)[1] or sr.get('simulated_ns') is None:
            continue
        md = json.loads((r / 'metadata.json').read_text())
        n_pass, n_pairs = pairs_passing(sr, md['nsubs_per_site'])
        score = (n_pass, md.get('total_transitions', 0))
        if best is None or score > best['score']:
            best = {'score': score, 'run': r, 'pass': n_pass, 'pairs': n_pairs,
                    'ns': sr.get('simulated_ns')}
    if cache is not None:
        cache[key] = best
    return best


def group_runs(data, pretraining_root=None, systems=None) -> List[Dict]:
    """Each passed group's MLLF run (of ``systems``, default every system),
    with ALF's best run of the same group when the stage's groups are ALF's
    (``OnlineData.alf_group_systems``). ALF's best runs are cached next to
    ALF's runs (``online_data.alf_cache_path``)."""
    from mllf.analysis.common import load_cache, save_cache
    from mllf.analysis.online_data import alf_cache_path
    from mllf.cb.cost_ledger import alf_dataset_dirs
    from mllf.cli.collect_pretraining_data import parse_output_file
    from mllf.file_handling.combo_sets import read_alf_groups
    cache_path = (alf_cache_path(pretraining_root).with_name(
        alf_cache_path(pretraining_root).stem + '_best.pkl') if pretraining_root else None)
    best_cache = load_cache(cache_path) if cache_path else {}
    n_cached = len(best_cache)
    out = []
    for sc in data.config.get('systems', []):
        sysn = sc['name']
        if systems and sysn not in systems:
            continue
        passed = (data.stage_state.get(sysn) or {}).get('groups_done') or {}
        if not passed:
            continue
        alf_groups, alf_dirs = [], []
        if pretraining_root and sysn in data.alf_group_systems:
            alf_groups, _ = read_alf_groups(sysn, pretraining_root, sc['input_dir'])
            alf_dirs = alf_dataset_dirs(sysn, pretraining_root)
        for combo, run in sorted(passed.items()):
            run_dir = Path(run)
            if not (run_dir / 'output.out').exists():
                log.debug(f"{sysn}/{combo}: no output.out in {run_dir}; skipped")
                continue
            labels = combo_labels(combo)
            sr = parse_output_file(run_dir / 'output.out', run_dir) or {}
            nsubs = [len(c) for _, c in _site_columns(labels)]
            rec = {'system': sysn, 'combo': combo, 'run': run_dir, 'labels': labels,
                   'ns': sr.get('simulated_ns'), 'pass': pairs_passing(sr, nsubs), 'alf': None}
            m = re.match(r'group(\d+)', combo)
            gi = int(m.group(1)) if m else None
            if gi and gi <= len(alf_dirs) and gi <= len(alf_groups):
                best = best_alf_run(alf_dirs[gi - 1], best_cache)
                if best:
                    g = alf_groups[gi - 1]
                    best['labels'] = [(site, sub) for site in sorted(g) for sub in g[site]]
                    best['dataset'] = alf_dirs[gi - 1].name
                    rec['alf'] = best
            out.append(rec)
    if cache_path and len(best_cache) > n_cached:
        save_cache(cache_path, best_cache)
    return out


# Line styles of the substituents: six line-safe shades (light colours are
# never lines) with a line style each, so neighbouring substituents differ in
# both and the darkest shades are not used for every line.
SUB_COLORS = [bs.PALETTE['plum'], bs.PALETTE['orange'], bs.PALETTE['forest'],
              bs.EXTENDED['plum_mid'], bs.EXTENDED['orange_dark'], bs.EXTENDED['forest_mid']]
SUB_LINESTYLES = ['-', '--', ':', '-.']
RUN_STYLE = {'MLLF': dict(color=bs.PALETTE['plum'], ls='-'),
             'ALF': dict(color=bs.PALETTE['orange'], ls='--')}


def _sub_style(i: int) -> Dict:
    return {'color': SUB_COLORS[i % len(SUB_COLORS)],
            'ls': SUB_LINESTYLES[i % len(SUB_LINESTYLES)]}


def profile_figure(rec: Dict, site: int, mllf_prof, alf_prof):
    """One site of one group: 1D profiles of the MLLF run (A) and ALF's run
    (B, when available) on one y scale, then the transition profile of each
    pair (MLLF solid, ALF dashed), titled "i / j" with x = 1 at substituent i.
    Legends sit to the right of the panels."""
    import matplotlib.pyplot as plt
    one_d, trans = mllf_prof
    a_one, a_trans = alf_prof if alf_prof else ({}, {})
    subs = list(one_d)
    n = len(subs)
    k = max(n - 1, 1)
    width = bs.WIDTHS['double']
    cell = min(1.25, (width - 1.4) / max(k, 2))
    fig = plt.figure(figsize=(width, 2.4 + cell * k + 0.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, cell * k + 0.4])
    top = gs[0].subgridspec(1, 2)
    ax_m = fig.add_subplot(top[0, 0])
    ax_a = fig.add_subplot(top[0, 1], sharey=ax_m)
    styles = {sub: _sub_style(i) for i, sub in enumerate(subs)}
    for ax, prof, name in ((ax_m, one_d, 'MLLF'), (ax_a, a_one, 'ALF')):
        for sub in subs:
            if sub in prof:
                x, g, _ = prof[sub]
                ax.plot(x, g - g.min(), label=f'Sub {sub}', **styles[sub])
        ax.set_xlim(0, 1)
        ax.set_xlabel('λ')
        ax.set_title(f'{name} run' if prof else f'{name}: no matching run')
    ax_m.set_ylabel('G (kcal/mol)')
    handles, labels = ax_m.get_legend_handles_labels()
    ax_a.legend(handles, labels, loc='center left', bbox_to_anchor=(1.03, 0.5))
    grid = gs[1].subgridspec(k, k + 1, width_ratios=[1] * k + [0.6])
    first = None
    small = bs.mpl.rcParams['font.size'] - 1
    for i, si in enumerate(subs):
        for j, sj in enumerate(subs):
            if j <= i:
                continue
            ax = fig.add_subplot(grid[i, j - 1], sharex=first, sharey=first)
            first = first or ax
            for prof, name in ((trans.get((si, sj)), 'MLLF'), (a_trans.get((si, sj)), 'ALF')):
                if prof is not None:
                    x, g, _ = prof
                    ax.plot(x, g - g.min(), label=name, **RUN_STYLE[name])
            ax.set_xlim(0, 1)
            ax.set_title(f'{si} / {sj}', fontsize=small, pad=2)
            ax.tick_params(labelsize=small)
            diagonal = j == i + 1        # leftmost in its row and lowest in its column
            ax.tick_params(labelleft=diagonal, labelbottom=diagonal)
            if i == 0 and j == 1:
                ax.set_ylabel('G (kcal/mol)')
            if i == n - 2:
                ax.set_xlabel(r'$\lambda_i / (\lambda_i + \lambda_j)$')
    if first is not None:
        lax = fig.add_subplot(grid[0, k])
        lax.axis('off')
        lax._bowling_no_snap = lax._bowling_no_zero = True
        lax.legend(*first.get_legend_handles_labels(), loc='upper left',
                   title='Pair i / j\n(x = 1 at i)')
    bs.label_panels([ax_m, ax_a])
    return fig


def figures(rec: Dict):
    """[(name, figure)] of one group, one per site."""
    mllf, _ = run_profiles(rec['run'] / 'output.out', rec['labels'])
    alf = None
    if rec['alf']:
        from mllf.file_handling.read_output import resolve_output_file
        try:
            alf, _ = run_profiles(resolve_output_file(rec['alf']['run']), rec['alf']['labels'])
        except (ValueError, OSError, TypeError) as e:
            log.warning(f"  Warning: ALF run {rec['alf']['run']} not read ({e})")
    group = rec['combo'].split('_site')[0]
    return [(f"free_energy/{rec['system']}_{group}_site{site}",
             lambda site=site, prof=prof: profile_figure(rec, site, prof, (alf or {}).get(site)))
            for site, prof in mllf.items()]
