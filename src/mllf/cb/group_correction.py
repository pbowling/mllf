"""Group-level end-term correction (bandit.group_correction).

In a group run, each substituent's endpoint collects the end terms (x, s) of
all its partners at the site. Each pair's terms are fine in its own pair run,
but together they can hold an endpoint off:

* one endpoint (``endpoint``): a substituent dominates the site yet almost
  never reaches lambda 0.99, its partners' terms acting there together
  (``system_stages.held_off_sites``; e.g. 1benz g01/g02);
* the whole site (``site``): no substituent at the site reaches 0.99 and it
  barely transitions, while another site in the same run samples normally
  (e.g. indolizine site 1: halving the site's x and s let all 8 substituents
  sample, 103 transitions; zeroing them trapped endpoints).

Pair runs cannot see either, so the correction lives with the group, in
``bandit_state[GROUP_KEY]['<system>|<group combo>']``: per directed pair
(i -> j, original subs) the x[i][j] / s[i][j] the group first ran with
(``base_x`` / ``base_s``) and a ratio ``r`` (1 = uncorrected). A flagged run
lowers r by ``step`` (scaled by the partner's lambda share for an endpoint)
down to ``floor``; a later run at a corrected site that has instead trapped
endpoints (substituents unsampled, the rest sampled, few transitions) undoes
half of each pair's last step. The group's runs then use base x r for those
pairs (``apply_group_values``) in place of the model + pair offsets, and the
corrected values are the model's training targets for those edges
(``correction_targets``), so the model learns what works in the group.
c and b are not touched.
"""
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from mllf.cb.hindsight import OFFSET_DIMS, pair_key
from mllf.cb.system_stages import held_off_sites

GROUP_KEY = 'group_offsets'
DEFAULTS = {
    'step': 0.25,                  # ratio lowered per flagged run (x partner weight)
    'floor': 0.5,                  # lowest ratio (most total shrink 50%)
    # endpoint trigger: as bandit.group_holdoff (system_stages.held_off_sites)
    'dominant_min': 0.7, 'at_max': 0.05, 'partner_min': 0.02, 'min_partners': 2,
    # whole-site trigger
    'site_min_subs': 3,            # sites of 2 are left to the pair rules
    'site_max_transitions': 2,
    'other_min_transitions': 10,   # another site in the run sampled normally
    # over-correction guard: a corrected site with unsampled substituents and
    # some sampled, trapped at endpoints: fewer transitions than
    # guard_transitions, or its dominant substituent at lambda >= 0.99 in at
    # least guard_at_min of its frames (indolizine, x / s zeroed: 3 of 8
    # unsampled, 19 transitions, 0.48 at the endpoint; halved: all sampled, 0.06)
    'guard_transitions': 5,
    'guard_at_min': 0.3,
    # training target sd: max(target_sd_frac x |value|, target_sd_min)
    'target_sd_frac': 0.1, 'target_sd_min': 0.1,
    'history': 20,
}
END_COLS = (OFFSET_DIMS['skew'], OFFSET_DIMS['end'])   # x, s


def correction_config(bandit_cfg: Optional[Dict]) -> Optional[Dict]:
    """bandit.group_correction merged over DEFAULTS; None when not set."""
    cfg = (bandit_cfg or {}).get('group_correction')
    if cfg is None or cfg is False:
        return None
    return {**DEFAULTS, **(cfg if isinstance(cfg, dict) else {})}


def group_id(system: str, combo_name: str) -> str:
    return f"{system}|{combo_name}"


def edge_id(system: str, a: Tuple[int, int], b: Tuple[int, int]) -> str:
    """Directed pair a -> b (original (site, sub) labels)."""
    key, forward = pair_key(system, a, b)
    return f"{key}|{'fwd' if forward else 'rev'}"


def run_sites(labels: Sequence[Tuple[int, int]]) -> List[Tuple[int, List[int]]]:
    """(original site, nodes) in node order, the order of the output's per-site transitions."""
    out: List[Tuple[int, List[int]]] = []
    for n, (site, _) in enumerate(labels):
        if not out or out[-1][0] != site:
            out.append((site, []))
        out[-1][1].append(n)
    return out


def _site_transitions(saved_data: Dict) -> List[float]:
    tr = saved_data.get('transitions') or []
    return [float(v) for v in (tr.values() if isinstance(tr, dict) else tr)]


def flagged_sites(saved_data: Dict, labels: Sequence[Tuple[int, int]], cfg: Dict,
                  block_offset: int = 2) -> List[Dict]:
    """Sites of a group run whose end terms hold an endpoint off.

    Returns [{'kind': 'endpoint' | 'site', 'site', 'nodes', 'edges': {(i, j):
    weight}}]: the directed edges (node indices) whose x[i][j] / s[i][j] are
    shrunk, weighted 0-1. Endpoint: each partner j -> the held-off substituent
    v, weighted by j's lambda share over the largest partner's. Site: every
    directed edge within the site, weight 1."""
    pops = [float(p) for p in (saved_data.get('populations') or [])]
    trans = _site_transitions(saved_data)
    sites = run_sites(labels)
    out: Dict[int, Dict] = {}
    for idx, (site, nodes) in enumerate(sites):
        if len(nodes) < int(cfg['site_min_subs']) or idx >= len(trans):
            continue
        others = [t for k, t in enumerate(trans) if k != idx]
        if (sum(pops[n] for n in nodes if n < len(pops)) <= 0
                and trans[idx] <= float(cfg['site_max_transitions'])
                and others and max(others) >= float(cfg['other_min_transitions'])):
            out[site] = {'kind': 'site', 'site': site, 'nodes': nodes,
                         'edges': {(i, j): 1.0 for i in nodes for j in nodes if i != j}}
    for h in held_off_sites(saved_data, cfg):
        v = int(h['dom']) - block_offset
        if not 0 <= v < len(labels) or labels[v][0] in out:
            continue
        shares = {int(b) - block_offset: float(s) for b, s in h['partners'].items()}
        shares = {j: s for j, s in shares.items() if 0 <= j < len(labels)}
        if not shares:
            continue
        top = max(shares.values())
        site = labels[v][0]
        out[site] = {'kind': 'endpoint', 'site': site, 'dom': v,
                     'nodes': next(n for s, n in sites if s == site),
                     'edges': {(j, v): s / top for j, s in shares.items()}}
    return list(out.values())


def _overcorrected(saved_data: Dict, nodes: Sequence[int], site_idx: int, cfg: Dict,
                   block_offset: int = 2) -> bool:
    pops = [float(p) for p in (saved_data.get('populations') or [])]
    trans = _site_transitions(saved_data)
    sampled = [n < len(pops) and pops[n] > 0 for n in nodes]
    if not any(sampled) or all(sampled):
        return False
    if site_idx < len(trans) and trans[site_idx] < float(cfg['guard_transitions']):
        return True
    blocks = {n + block_offset for n in nodes}
    return any(int(h.get('dom', -1)) in blocks and float(h.get('at_frac', 0)) >= float(cfg['guard_at_min'])
               for h in saved_data.get('site_holdoff') or [])


def _m(mat, r, c) -> Optional[float]:
    try:
        v = float(mat[r][c])
    except (IndexError, TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def update_group(state: Dict, system: str, combo_name: str, saved_data: Dict,
                 labels: Sequence[Tuple[int, int]], applied: Dict, cfg: Dict,
                 epoch: int, run: str) -> Dict:
    """Record one group run: step the flagged sites' pairs, or guard back a
    corrected site that overshot. ``applied``: the run's own x/s matrices
    (``read_applied_biases``), the base for pairs seen for the first time.

    Returns {'flagged': flagged_sites output, 'stepped': any ratio lowered,
    'capped': a flagged pair already at the floor, 'guarded': sites guarded,
    'edges': {(i, j): (base_x, base_s, r)} for every edge at a flagged or
    guarded site (the values the group now uses)}."""
    flagged = flagged_sites(saved_data, labels, cfg)
    entry = state.setdefault(GROUP_KEY, {}).setdefault(group_id(system, combo_name),
                                                       {'pairs': {}, 'history': []})
    pairs = entry['pairs']
    xm, sm = applied.get('x') or [], applied.get('s') or []
    step, floor = float(cfg['step']), float(cfg['floor'])
    out = {'flagged': flagged, 'stepped': False, 'capped': False, 'guarded': [], 'edges': {}}
    flagged_site_ids = {f['site'] for f in flagged}

    def _pair(i, j):
        eid = edge_id(system, labels[i], labels[j])
        p = pairs.get(eid)
        if p is None:
            bx, bs = _m(xm, i, j), _m(sm, i, j)
            if bx is None or bs is None:
                return eid, None
            p = pairs[eid] = {'base_x': bx, 'base_s': bs, 'r': 1.0, 'last': 0.0}
        return eid, p

    for f in flagged:
        for (i, j), w in f['edges'].items():
            eid, p = _pair(i, j)
            if p is None:
                continue
            new = max(floor, p['r'] - step * w)
            if new < p['r'] - 1e-9:
                p['last'], p['r'] = p['r'] - new, new
                out['stepped'] = True
            else:
                out['capped'] = True
        for i in f['nodes']:
            for j in f['nodes']:
                if i != j:
                    p = pairs.get(edge_id(system, labels[i], labels[j]))
                    if p is not None:
                        out['edges'][(i, j)] = (p['base_x'], p['base_s'], p['r'])
    for idx, (site, nodes) in enumerate(run_sites(labels)):
        if site in flagged_site_ids:
            continue
        ids = [(i, j, edge_id(system, labels[i], labels[j])) for i in nodes for j in nodes if i != j]
        corrected = [(i, j, e) for i, j, e in ids if e in pairs and pairs[e]['r'] < 1.0]
        if not corrected or not _overcorrected(saved_data, nodes, idx, cfg):
            continue
        for i, j, e in corrected:
            p = pairs[e]
            back = p['last'] / 2.0
            p['r'], p['last'] = min(1.0, p['r'] + back), back
            out['edges'][(i, j)] = (p['base_x'], p['base_s'], p['r'])
        out['guarded'].append(site)
    if flagged or out['guarded']:
        entry['history'].append({
            'epoch': int(epoch), 'run': str(run), 'stepped': out['stepped'], 'capped': out['capped'],
            'flagged': [{'kind': f['kind'], 'site': f['site'],
                         **({'dom': f"s{labels[f['dom']][0]}:{labels[f['dom']][1]}"} if 'dom' in f else {})}
                        for f in flagged],
            'guarded': out['guarded'],
            'ratio': {str(f['site']): round(min((pairs[edge_id(system, labels[i], labels[j])]['r']
                                                  for (i, j) in f['edges']
                                                  if edge_id(system, labels[i], labels[j]) in pairs), default=1.0), 3)
                      for f in flagged}})
        del entry['history'][:-int(cfg['history'])]
    return out


def apply_group_values(actions: torch.Tensor, state: Optional[Dict], system: str, combo_name: str,
                       labels: Optional[Sequence[Tuple[int, int]]],
                       edge_index: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Set each corrected pair's x / s (columns skew / end) to base x r, on a
    copy. Groups without corrections (and pair combos) are unchanged."""
    entry = ((state or {}).get(GROUP_KEY) or {}).get(group_id(system, combo_name))
    if not entry or not labels:
        return actions, 0
    pairs = entry.get('pairs') or {}
    out, n = actions.clone(), 0
    for k in range(edge_index.size(1)):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i >= len(labels) or j >= len(labels):
            continue
        p = pairs.get(edge_id(system, labels[i], labels[j]))
        if p is None or p['r'] >= 1.0:
            continue
        out[k, END_COLS[0]] = p['base_x'] * p['r']
        out[k, END_COLS[1]] = p['base_s'] * p['r']
        n += 1
    return out, n


def correction_targets(edge_index: torch.Tensor, result: Dict, cfg: Dict
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For ``update_group``'s result: ([E, 4] targets, [E, 4] var, [E] mask of
    edges at a flagged or guarded site). Masked edges get their pair's
    corrected x / s as targets (sd max(target_sd_frac x |value|,
    target_sd_min)); their b and c (var inf) and other edges are untouched by
    the caller."""
    E = edge_index.size(1)
    t = torch.zeros(E, 4, dtype=torch.float64)
    v = torch.full((E, 4), math.inf, dtype=torch.float64)
    mask = torch.zeros(E, dtype=torch.bool)
    nodes = {n for f in result['flagged'] for n in f['nodes']}
    edges = result['edges']
    frac, floor = float(cfg['target_sd_frac']), float(cfg['target_sd_min'])
    for k in range(E):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if (i, j) in edges:
            bx, bs, r = edges[(i, j)]
            for col, val in zip(END_COLS, (bx * r, bs * r)):
                t[k, col] = val
                v[k, col] = max(frac * abs(val), floor) ** 2
            mask[k] = True
        elif i in nodes and j in nodes:
            mask[k] = True
    return t, v, mask


def run_record(state: Optional[Dict], system: str, combo_name: str, run: str) -> Optional[Dict]:
    """The history record ``update_group`` wrote for this run, if any."""
    entry = ((state or {}).get(GROUP_KEY) or {}).get(group_id(system, combo_name)) or {}
    name = Path(str(run)).name
    return next((h for h in reversed(entry.get('history') or []) if Path(h.get('run', '')).name == name), None)
