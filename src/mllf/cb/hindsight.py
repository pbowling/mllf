"""Hindsight bias targets, per-pair offsets and per-pair exploration for online training.

A finished MSLD run measures how far its biases were from the ones that would
have worked. Each run is therefore relabelled with the corrected biases it
implies, and the policy learns from those targets. Scoring the run with a
scalar reward would say only whether the biases were good, not in which
direction or by how much they were off.

Per pair, from the run's own lambda trajectory (read_output.parse_pair_profile;
slopes from the pretraining bias-sensitivity analysis):

* linear: raising b[j] raises ln(p_j / p_i) by 1/kT, so the balancing
  difference for directed edge (i -> j), whose action approximates
  ``b[j] - b[i]``, is ``db* = db - kT ln(p_j / p_i)``; a substituent never
  sampled gives a capped step toward it;
* quadratic: the midpoint barrier falls by ``BARRIER_C_SLOPE`` per unit c, so
  ``c* = c + (barrier - target) / |slope|``;
* skew: the skew asymmetry moves -/+``SKEW_X_SLOPE`` per unit x[lo][hi] /
  x[hi][lo]; the step removing it is split between the two directions;
* end: the end asymmetry (K_hi - K_lo) moves -``END_S_SLOPE_HI`` per unit
  s[lo][hi] and +``END_S_SLOPE_LO`` per unit s[hi][lo]; split likewise;
* an endpoint the trajectory crowds up to but does not hold (many frames just
  short of it, almost none at it) means the end term there holds lambda off
  it: s[hi][lo] at the lo end, s[lo][hi] at the hi end. That s is moved part of
  the way toward ``END_REFERENCE_S``, so a large error shrinks geometrically
  over a few runs. This takes precedence over the end asymmetry for that s.

Each target carries its own variance (measurement noise plus a term for
linearization) and each step is capped (a trust region), since the linearized
responses only hold near sensible biases.

Iterative flattening schemes correct one system's biases from its own
simulations and start every new system from scratch. Here each corrected
target instead trains a shared model conditioned on the substituents' Uni-Mol
embeddings, so what is learned on one pair improves the first guess for pairs
and systems that have never been simulated:

* every target enters the NeuralLinear posterior as an observation weighted by
  its own variance, rather than being regularized with fixed constants;
* per-pair offsets combine a pair's own measurements with the model prediction
  as a Bayesian prior, so a pair the model predicts well needs little
  correction and one it predicts poorly is fixed from its own data without
  distorting the shared model;
* proposals sample each pair's offset rather than taking its mean, and a pair
  whose run fails without giving a direction for any coefficient widens its
  own exploration, so failure modes that no signal diagnoses are still
  escaped; the shared posterior's uncertainty and the pairs' own state decide
  which pairs to simulate next.

Offsets and exploration factors live in the bandit state under ``OFFSET_KEY``,
keyed by the ORIGINAL site/sub numbers from the combo's mapping.json, so they
follow a pair into any combo. Per pair, slots: ``linear`` (lower sub -> higher
sub), ``quadratic``, and ``skew_fwd``/``skew_rev``, ``end_fwd``/``end_rev`` (the
x and s entries from the lower to the higher sub and back, which are
independent parameters).
"""
import json
import math
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

KB_KCAL = 0.0019872041
DEFAULT_TEMPERATURE = 298.15

# Responses of the lambda-profile signals, from the pretraining bias-sensitivity
# analysis (see read_output.parse_pair_profile).
BARRIER_C_SLOPE = -0.233
SKEW_X_SLOPE = 0.19      # skew: -0.19 per unit x[lo][hi], +0.19 per unit x[hi][lo]
END_S_SLOPE_HI = 0.44    # end (K_hi - K_lo): -0.44 per unit s[lo][hi]
END_S_SLOPE_LO = 0.46    #                     +0.46 per unit s[hi][lo]

# Trust-region caps on one correction (bandit.hindsight_*_step_cap). The b and
# c values follow the step limits of ALF's flattening (1.5 x its cutoffs); x and
# s use the same rule.
LINEAR_STEP_CAP = 3.0
QUADRATIC_STEP_CAP = 12.0
# A pair stepping b toward an unsampled substituent in the same direction run
# after run is far from its working b: each consecutive step doubles, up to
# this (bandit.linear_step_max); reset once both substituents are sampled.
LINEAR_STEP_MAX = 12.0
# Barrier trap: a pair that still never crosses once its b bracket has closed
# to within BARRIER_TRAP_WIDTH (or overlaps: stuck on either side over the same
# b) is held by the barrier, which no run measures (that needs a crossing). Its
# c steps up by the quadratic step cap, doubling run after run up to
# QUADRATIC_STEP_MAX (bandit.barrier_trap_width / quadratic_step_max);
# converged ALF c reaches 100-300 for some systems.
BARRIER_TRAP_WIDTH = 2.0
QUADRATIC_STEP_MAX = 48.0
# A run whose lambda kept >= PINNED_FRACTION of the pair's frames on its
# starting substituent (and never sampled the other) is pinned: the barrier,
# not b, held it, so it gives no b direction and steps c up. A run sampling
# neither endpoint is stranded mid-way: c flattens too much, so it steps c down.
PINNED_FRACTION = 0.99
SKEW_STEP_CAP = 3.0
END_STEP_CAP = 1.5

# Noise. b: pair b* scatter around its consensus (cycle closure in the
# pretraining data). Barrier, skew and end: assumed run-to-run scatter of the
# signal at fixed biases.
LINEAR_SD_FLOOR = 0.35
BARRIER_SD = 0.2
SKEW_SD = 0.3
END_SD = 0.25
# Extra relative uncertainty of a linearized step (the slopes are measured
# near sensible values; large corrections overshoot).
LINEARIZATION_FRAC = 0.25

# Endpoint held off: >= NEAR_END_MIN_FRACTION of the pair's on-edge frames are
# just short of the endpoint and < HELD_OFF_RATIO of the frames near or at it
# are at it. Its s then moves END_REFERENCE_FRACTION of the way to
# END_REFERENCE_S (bandit.end_reference_s), a typical converged end bias.
NEAR_END_MIN_FRACTION = 0.20
HELD_OFF_RATIO = 0.05
END_REFERENCE_S = 1.5
END_REFERENCE_FRACTION = 0.5

# Pair success, for exploration (minority fraction, pair transitions).
SUCCESS_MIN_FRACTION = 0.2
SUCCESS_MIN_TRANSITIONS = 10

# Revisits (bandit.revisit): a pair in progress is rerun every epoch until its
# exploration is satisfactory (REVISIT_SUCCESSES successful runs) or it has had
# REVISIT_MAX_VISITS visits; per-pair corrections (b bracket, c steps) need
# several consecutive visits to converge.
REVISIT_SUCCESSES = 2
REVISIT_MAX_VISITS = 8
REVISIT_COMBOS_KEPT = 4

OFFSET_KEY = '_pair_offsets'
# Recent runs sit closest to the current biases, where the linearization holds.
OFFSET_HISTORY = 5
# Prior sd of a pair's offset when combining its measurements with the model:
# how far the model can plausibly be off for one pair (end biases have been
# off by >10 for substituents with poor pretraining data), so a data-backed
# target, even a directional one, dominates.
OFFSET_PRIOR_SD = {'linear': 3.0, 'quadratic': 8.0, 'skew': 4.0, 'end': 8.0}
# Exploration: a measured offset is sampled with sd base_sd x factor, capped by
# its posterior sd; an unmeasured slot adds noise beyond the model's own
# Thompson sample only once its factor > 1. The factor doubles when the pair
# fails and the run gives no direction for any of its coefficients (an
# unexplained failure), or when it has failed PERSISTENT_FAILURES runs in a row
# although its targets gave directions (bandit.explore_after_failures; the
# targets are not fixing it); it halves on success.
EXPLORE_BASE_SD = {'linear': 0.5, 'quadratic': 2.0, 'skew': 0.5, 'end': 0.5}
EXPLORE_FACTOR_MAX = 8.0
PERSISTENT_FAILURES = 3

BIAS_DIMS = ('linear', 'quadratic', 'skew', 'end')   # policy output columns
OFFSET_DIMS = {name: d for d, name in enumerate(BIAS_DIMS)}


def read_applied_biases(run_dir) -> Optional[Dict[str, List]]:
    """The b/c/x/s matrices a run actually used, from its ``variables.py``."""
    import yaml
    path = Path(run_dir) / 'variables.py'
    if not path.exists():
        return None
    m = re.search(r"bias_string\s*=\s*('''|\"\"\")(.*?)\1", path.read_text(), re.S)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(2))
    except Exception:
        return None
    b = data.get('b')
    if b and isinstance(b[0], list):
        b = b[0]
    return {'b': b, 'c': data.get('c'), 'x': data.get('x'), 's': data.get('s')}


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _capped(delta: float, cap: float, base_var: float) -> Tuple[float, float]:
    """Clip a step to the trust region; variance adds linearization and clipping."""
    clipped = max(-cap, min(cap, delta))
    var = base_var + (LINEARIZATION_FRAC * clipped) ** 2
    if clipped != delta:
        var += (delta - clipped) ** 2  # capped: the true value is further out
    return clipped, var


def linear_hindsight(db: float, p_i: float, p_j: float, transitions: Optional[int],
                     kT: float, step_cap: float = LINEAR_STEP_CAP,
                     node_transitions: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
    """Target and variance for db = b[j] - b[i] from the pair's populations.

    Both sampled and crossing: the balancing value, with variance from the
    number of crossings (each side visited ~T/2 times). One side unsampled: a
    capped step toward it (a direction, not a value) with sd step_cap / 2, so
    against the offset prior (sd 3) it moves ~80% of the step, and a later
    two-sided measurement outweighs it ~6x.
    Neither sampled: no information (variance inf).

    ``node_transitions`` (c_i, c_j): transitions into or out of each
    substituent, over all its partners on the site (``node_transition_counts``).
    On a site with more than two substituents lambda can reach j from i only
    through others, so the pair's own count can be 0 while both are sampled
    and exchanging; their populations then still measure db. Given, the
    variance is kT^2 (2/c_i + 2/c_j) (each visited ~c/2 times; equals the
    pair formula for a two-substituent site), and the target is directional
    only when a side is unsampled.
    """
    if p_i <= 0 and p_j <= 0:
        return db, math.inf
    if node_transitions is not None and p_i > 0 and p_j > 0 and min(node_transitions) > 0:
        c_i, c_j = node_transitions
        delta = -kT * math.log(p_j / p_i)
        clipped, var = _capped(delta, step_cap,
                               (kT ** 2) * (2.0 / c_i + 2.0 / c_j) + LINEAR_SD_FLOOR ** 2)
        return db + clipped, var
    if p_i <= 0 or p_j <= 0 or not transitions:
        # Trapped: move toward the unsampled (or less-populated) side.
        step = step_cap if p_j <= p_i else -step_cap
        return db + step, (step_cap / 2.0) ** 2
    delta = -kT * math.log(p_j / p_i)
    clipped, var = _capped(delta, step_cap,
                           (kT ** 2) * 4.0 / max(transitions, 1) + LINEAR_SD_FLOOR ** 2)
    return db + clipped, var


def quadratic_hindsight(c: float, barrier: Optional[float], barrier_target: float,
                        slope: float = BARRIER_C_SLOPE,
                        step_cap: float = QUADRATIC_STEP_CAP) -> Tuple[float, float]:
    """Target and variance for c from the pair's midpoint barrier."""
    if not _finite(barrier):
        return c, math.inf
    clipped, var = _capped((barrier - barrier_target) / abs(slope), step_cap,
                           (BARRIER_SD / slope) ** 2)
    return c + clipped, var


def skew_hindsight(x_lh: float, x_hl: float, skew: Optional[float],
                   step_cap: float = SKEW_STEP_CAP):
    """Targets and variances for (x[lo][hi], x[hi][lo]) from the skew asymmetry.

    Removing the asymmetry needs -0.19 dx_lh + 0.19 dx_hl = -skew; the step is
    split equally: dx_lh = skew / (2 * 0.19), dx_hl = -dx_lh.
    """
    if not _finite(skew):
        return (x_lh, math.inf), (x_hl, math.inf)
    step, var = _capped(skew / (2.0 * SKEW_X_SLOPE), step_cap,
                        (SKEW_SD / (2.0 * SKEW_X_SLOPE)) ** 2)
    return (x_lh + step, var), (x_hl - step, var)


def end_hindsight(s_lh: float, s_hl: float, end: Optional[float],
                  step_cap: float = END_STEP_CAP):
    """Targets and variances for (s[lo][hi], s[hi][lo]) from the end asymmetry.

    Removing it needs -0.44 ds_lh + 0.46 ds_hl = -end; split equally:
    ds_lh = end / (2 * 0.44), ds_hl = -end / (2 * 0.46).
    """
    if not _finite(end):
        return (s_lh, math.inf), (s_hl, math.inf)
    d_lh, v_lh = _capped(end / (2.0 * END_S_SLOPE_HI), step_cap,
                         (END_SD / (2.0 * END_S_SLOPE_HI)) ** 2)
    d_hl, v_hl = _capped(-end / (2.0 * END_S_SLOPE_LO), step_cap,
                         (END_SD / (2.0 * END_S_SLOPE_LO)) ** 2)
    return (s_lh + d_lh, v_lh), (s_hl + d_hl, v_hl)


def endpoint_held_off(endpoints: Optional[Dict], side: str) -> bool:
    """Whether the trajectory crowds up to an endpoint ('lo'/'hi') without holding it."""
    if not endpoints or not endpoints.get('edge'):
        return False
    at, near = endpoints.get(f'at_{side}', 0), endpoints.get(f'near_{side}', 0)
    if near < NEAR_END_MIN_FRACTION * endpoints['edge']:
        return False
    return at < HELD_OFF_RATIO * (at + near)


def held_off_end_hindsight(s: float, reference: float = END_REFERENCE_S,
                           fraction: float = END_REFERENCE_FRACTION) -> Tuple[float, float]:
    """Target and variance for an end bias that holds lambda off its endpoint.

    Moves ``fraction`` of the way to ``reference``; a direction more than a
    value, so the variance is (step / 2)^2. No information when s is already
    at or below the reference (something else is holding the endpoint off).
    """
    if s <= reference:
        return s, math.inf
    step = fraction * (s - reference)
    return s - step, max((step / 2.0) ** 2, 1e-4)


def _pair_transitions(saved_data: Dict, key: str) -> Optional[int]:
    pair_trans = saved_data.get('transition_pairs') or {}
    if key in pair_trans:
        return pair_trans[key]
    # Pair runs have one site: its total is the pair's count when the
    # per-pair breakdown is missing (older runs).
    site_trans = saved_data.get('transitions') or []
    if isinstance(site_trans, dict):
        site_trans = list(site_trans.values())
    return site_trans[0] if len(site_trans) == 1 else None


def site_nodes(edge_index: torch.Tensor) -> Dict[int, List[int]]:
    """Node -> sorted nodes of its site. Online graphs have edges only within
    a site, so the sites are the connected components of ``edge_index``."""
    parent: Dict[int, int] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k in range(edge_index.size(1)):
        a, b = int(edge_index[0, k]), int(edge_index[1, k])
        parent[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for n in list(parent):
        groups.setdefault(find(n), []).append(n)
    return {n: sorted(g) for g in groups.values() for n in g}


def node_transition_counts(saved_data: Dict, nodes: Sequence[int],
                           block_offset: int = 2) -> Optional[Dict[int, int]]:
    """Transitions into or out of each node over all its partners on the site
    (from ``transition_pairs``); None if the per-pair breakdown is missing."""
    pair_trans = saved_data.get('transition_pairs') or {}
    if not pair_trans:
        return None
    counts = {n: 0 for n in nodes}
    for a in nodes:
        for b in nodes:
            if a < b:
                t = pair_trans.get(f"{a + block_offset}_{b + block_offset}") or 0
                counts[a] += t
                counts[b] += t
    return counts


def _site_transitions(saved_data: Dict, nodes: Sequence[int], block_offset: int = 2) -> Optional[int]:
    """Total transitions between the site's substituents (None if unknown)."""
    counts = node_transition_counts(saved_data, nodes, block_offset)
    return None if counts is None else sum(counts.values()) // 2


def pair_succeeded(saved_data: Dict, i: int, j: int, block_offset: int = 2,
                   min_fraction: float = SUCCESS_MIN_FRACTION,
                   min_transitions: int = SUCCESS_MIN_TRANSITIONS,
                   site: Optional[Sequence[int]] = None) -> bool:
    """Balanced (minority fraction) and crossing (pair transitions) in this run.

    ``site``: the nodes on the pair's site. With more than two, lambda moves
    between i and j through the others, so crossing means each of the two
    entered or left at least ``min_transitions`` times (any partner) rather
    than ``min_transitions`` direct i <-> j transitions; for a pair the two
    are the same.
    """
    pops = saved_data.get('populations') or []
    p_i = pops[i] if i < len(pops) else 0
    p_j = pops[j] if j < len(pops) else 0
    if p_i <= 0 or p_j <= 0 or min(p_i, p_j) / (p_i + p_j) < min_fraction:
        return False
    if site is not None and len(site) > 2:
        counts = node_transition_counts(saved_data, site, block_offset)
        if counts is not None:
            return min(counts[i], counts[j]) >= min_transitions
    key = f"{min(i, j) + block_offset}_{max(i, j) + block_offset}"
    T = _pair_transitions(saved_data, key)
    return T is not None and T >= min_transitions


def hindsight_targets(edge_index: torch.Tensor, applied: Dict[str, List], saved_data: Dict,
                      barrier_target: float, temperature: float = DEFAULT_TEMPERATURE,
                      block_offset: int = 2,
                      linear_step_cap: float = LINEAR_STEP_CAP,
                      quadratic_step_cap: float = QUADRATIC_STEP_CAP,
                      skew_step_cap: float = SKEW_STEP_CAP,
                      end_step_cap: float = END_STEP_CAP,
                      end_reference: float = END_REFERENCE_S,
                      return_directional: bool = False):
    """Per-edge hindsight targets for all four heads.

    Args:
        edge_index: [2, E] directed edges (node indices).
        applied: ``read_applied_biases`` output for the run.
        saved_data: the run's epoch_results.pt dict (populations,
            transition_pairs / transitions, barrier_pairs, skew_pairs,
            end_pairs, endpoint_pairs).
        barrier_target: the reward's barrier target (kcal/mol).

    Returns:
        (targets [E, 4], var [E, 4]) in physical units, columns
        [linear, quadratic, skew, end]: for edge (i -> j) the directed linear
        difference b[j] - b[i], c, x[i][j] and s[i][j]. ``var`` is inf where
        the run says nothing about that coefficient. Use
        ``held_off_counts`` to see which s targets came from a held-off
        endpoint. With ``return_directional``, also a [E, 4] bool mask of
        targets that are a direction rather than a measured value (b toward a
        substituent never sampled, s at a held-off endpoint); see
        ``record_pair_targets``.
    """
    kT = KB_KCAL * temperature
    E = edge_index.size(1)
    targets = torch.zeros(E, 4, dtype=torch.float64)
    var = torch.full((E, 4), math.inf, dtype=torch.float64)
    directional = torch.zeros(E, 4, dtype=torch.bool)
    b = applied.get('b') or []
    cmat = applied.get('c') or []
    xmat = applied.get('x') or []
    smat = applied.get('s') or []
    pops = saved_data.get('populations') or []
    barriers = saved_data.get('barrier_pairs') or {}
    skews = saved_data.get('skew_pairs') or {}
    ends = saved_data.get('end_pairs') or {}
    endpoints = saved_data.get('endpoint_pairs') or {}
    sites = site_nodes(edge_index)
    node_trans: Dict[Tuple[int, ...], Optional[Dict[int, int]]] = {}

    def _m(mat, r, c):
        try:
            return float(mat[r][c])
        except (IndexError, TypeError, ValueError):
            return None

    for k in range(E):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        lo, hi = min(i, j), max(i, j)
        key = f"{lo + block_offset}_{hi + block_offset}"
        if i < len(b) and j < len(b):
            p_i = pops[i] if i < len(pops) else 0
            p_j = pops[j] if j < len(pops) else 0
            T = _pair_transitions(saved_data, key)
            site = tuple(sites.get(i, (i, j)))
            nt = None
            if len(site) > 2:
                if site not in node_trans:
                    node_trans[site] = node_transition_counts(saved_data, site, block_offset)
                counts = node_trans[site]
                nt = (counts[i], counts[j]) if counts is not None else None
            t, v = linear_hindsight(float(b[j]) - float(b[i]), p_i, p_j, T, kT,
                                    step_cap=linear_step_cap, node_transitions=nt)
            targets[k, 0], var[k, 0] = t, v
            measured = (p_i > 0 and p_j > 0
                        and (bool(T) or (nt is not None and min(nt) > 0)))
            directional[k, 0] = math.isfinite(v) and not measured
        c = _m(cmat, lo, hi)
        if c is not None:
            targets[k, 1], var[k, 1] = quadratic_hindsight(c, barriers.get(key), barrier_target,
                                                           step_cap=quadratic_step_cap)
        forward = i == lo   # this edge carries x[lo][hi] / s[lo][hi]
        x_lh, x_hl = _m(xmat, lo, hi), _m(xmat, hi, lo)
        if x_lh is not None and x_hl is not None:
            fw, bw = skew_hindsight(x_lh, x_hl, skews.get(key), step_cap=skew_step_cap)
            targets[k, 2], var[k, 2] = fw if forward else bw
        s_lh, s_hl = _m(smat, lo, hi), _m(smat, hi, lo)
        if s_lh is not None and s_hl is not None:
            fw, bw = end_hindsight(s_lh, s_hl, ends.get(key), step_cap=end_step_cap)
            ep = endpoints.get(key)
            # s[lo][hi] shapes the hi end, s[hi][lo] the lo end.
            held = {'hi': endpoint_held_off(ep, 'hi'), 'lo': endpoint_held_off(ep, 'lo')}
            if held['hi']:
                fw = held_off_end_hindsight(s_lh, end_reference)
            if held['lo']:
                bw = held_off_end_hindsight(s_hl, end_reference)
            targets[k, 3], var[k, 3] = fw if forward else bw
            directional[k, 3] = math.isfinite(var[k, 3]) and held['hi' if forward else 'lo']
    if return_directional:
        return targets, var, directional
    return targets, var


def grow_directional_b_steps(state: Dict, system: str, labels: List[Tuple[int, int]],
                             edge_index: torch.Tensor, targets: torch.Tensor,
                             directional: torch.Tensor, applied: Dict[str, List],
                             base_step: float = LINEAR_STEP_CAP,
                             max_step: float = LINEAR_STEP_MAX,
                             var: Optional[torch.Tensor] = None,
                             skip: Optional[torch.Tensor] = None,
                             counts: Optional[Dict[str, int]] = None) -> Tuple[int, int]:
    """Directional b targets for trapped pairs: grow the step, then bracket (in place).

    ``targets``/``directional`` are ``hindsight_targets`` outputs, whose
    directional b targets are steps of ``base_step``. Per pair the bandit state
    keeps, in the pair's lower -> higher original-sub orientation:

    * ``b_streak`` = {'dir': +1/-1, 'n': consecutive steps that way}: a step in
      the same direction as the last becomes base_step x 2^n (capped at
      ``max_step``); a reversal restarts at base_step. If ``var`` is given, a
      grown step's variance is scaled by (base_step / size)^2, so the pair's
      offset takes nearly the whole step rather than ~80% of it;
    * ``b_bracket`` = {'lo', 'hi'}: a run stuck needing a higher b sets a lower
      bound (the largest such b), one needing a lower b an upper bound (the
      smallest). Once both exist the working b lies between them, so the
      target is the midpoint (bisection) instead of another step, and, if
      ``var`` is given, its variance shrinks to (bracket width / 4)^2 (not
      above the base step's).
    * Inverted bounds (lo > hi: stuck on either side over the same b range)
      mean the barrier, not b, traps lambda, and the endpoint a run ends on
      is chance. Bisecting them would lock b at sd 0.25 until a crossing
      that the barrier prevents, so they are unlocked instead: bracket and
      streak are cleared, the target is their midpoint (the best b guess)
      with the base step's variance, and ``barrier_trap`` is set on the
      pair so ``barrier_trapped_c_targets`` steps c up.

    Any non-directional run (both substituents sampled) clears both. Pairs
    flagged in ``skip`` ([E] bool; runs pinned at their start or stranded
    mid-way, see ``lambda_states``) say nothing about b and leave both
    untouched. Returns (n pairs whose step grew, n pairs targeted by bracket
    midpoint); ``counts``, if given, gets 'b_unlocked' incremented per unlock.
    """
    store = state.setdefault(OFFSET_KEY, {})
    b = applied.get('b') or []
    grown = bracketed = 0
    by_pair: Dict[str, List[Tuple[int, int, int, bool]]] = {}
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        by_pair.setdefault(key, []).append((k, i, j, forward))
    for key, edges in by_pair.items():
        entry = store.setdefault(key, {})
        entry.pop('barrier_trap', None)                   # set per run, below
        k, i, j, forward = edges[0]
        if i >= len(b) or j >= len(b):
            continue
        if skip is not None and bool(skip[k]):
            continue
        if not bool(directional[k, 0]):
            entry.pop('b_streak', None)
            entry.pop('b_bracket', None)
            continue
        sgn = 1.0 if forward else -1.0
        db = float(b[j]) - float(b[i])
        step = float(targets[k, 0]) - db
        if step == 0.0:
            continue
        direction = (1 if step > 0 else -1) * (1 if forward else -1)
        db_c = sgn * db                                   # canonical orientation
        bracket = entry.setdefault('b_bracket', {'lo': None, 'hi': None})
        if direction > 0:
            bracket['lo'] = db_c if bracket['lo'] is None else max(bracket['lo'], db_c)
        else:
            bracket['hi'] = db_c if bracket['hi'] is None else min(bracket['hi'], db_c)

        if bracket['lo'] is not None and bracket['hi'] is not None and bracket['lo'] > bracket['hi']:
            # inverted: unlock (see docstring)
            target_c = 0.5 * (bracket['lo'] + bracket['hi'])
            entry.pop('b_bracket', None)
            entry.pop('b_streak', None)
            entry['barrier_trap'] = True
            entry['b_unlocks'] = int(entry.get('b_unlocks', 0)) + 1
            if counts is not None:
                counts['b_unlocked'] = counts.get('b_unlocked', 0) + 1
            for k2, i2, j2, fwd2 in edges:
                targets[k2, 0] = (1.0 if fwd2 else -1.0) * target_c
                if var is not None:
                    var[k2, 0] = (base_step / 2.0) ** 2
            continue

        if bracket['lo'] is not None and bracket['hi'] is not None:
            target_c = 0.5 * (bracket['lo'] + bracket['hi'])
            width = abs(bracket['hi'] - bracket['lo'])
            entry['b_streak'] = {'dir': 0, 'n': 0}
            bracketed += 1
            for k2, i2, j2, fwd2 in edges:
                s2 = 1.0 if fwd2 else -1.0
                targets[k2, 0] = s2 * target_c
                if var is not None:
                    var[k2, 0] = min(float(var[k2, 0]), max((width / 4.0) ** 2, 0.25 ** 2))
            continue

        streak = entry.get('b_streak') or {'dir': 0, 'n': 0}
        n = streak['n'] if streak.get('dir') == direction else 0
        size = min(base_step * (2 ** n), max_step)
        entry['b_streak'] = {'dir': direction, 'n': n + 1}
        if size > abs(step):
            grown += 1
        for k2, i2, j2, fwd2 in edges:
            db2 = float(b[j2]) - float(b[i2])
            targets[k2, 0] = db2 + direction * size * (1 if fwd2 else -1)
            if var is not None and size > base_step:
                # With the base step's variance, the pair's offset (blended
                # with the model prediction, prior sd 3) would take only ~80%
                # of a grown step each run, losing the doubling; scale the
                # variance by (base / size)^2 so it takes nearly all of it.
                var[k2, 0] = min(float(var[k2, 0]),
                                 (base_step / 2.0) ** 2 * (base_step / size) ** 2)
    return grown, bracketed


def barrier_trapped_c_targets(state: Dict, system: str, labels: List[Tuple[int, int]],
                              edge_index: torch.Tensor, targets: torch.Tensor,
                              var: torch.Tensor, directional: torch.Tensor,
                              applied: Dict[str, List],
                              base_step: float = QUADRATIC_STEP_CAP,
                              max_step: float = QUADRATIC_STEP_MAX,
                              trap_width: float = BARRIER_TRAP_WIDTH,
                              grow: bool = True,
                              pinned: Optional[torch.Tensor] = None,
                              stranded: Optional[torch.Tensor] = None,
                              counts: Optional[Dict[str, int]] = None) -> Tuple[int, int]:
    """Directional c targets where the run's lambda shows c is off (in place).

    c too low (step up): the run was ``pinned`` at its starting substituent
    (``lambda_states``), or it did not cross (``directional`` b) although its
    ``b_bracket`` (see ``grow_directional_b_steps``, run first) has closed to
    within ``trap_width``: b is near its working value, so the barrier holds
    lambda; or ``grow_directional_b_steps`` unlocked an inverted b bracket on
    this run (``barrier_trap``). c too high (step down): the run was
    ``stranded`` between the endpoints, unless it measured a barrier (that
    target is kept).

    The c target becomes c +/- step with the base step's variance
    (base_step / 2)^2, and ``directional[:, 1]`` is set, so the step replaces
    the pair's earlier directional c targets. (A variance growing with the step
    would cancel the growth: against the offset prior, steps of 12, 24, 48 would
    each move the offset only ~5-8.) A step down stops at 0 unless c is already
    below it.

    grow=True (per-pair offsets): consecutive steps the same way double
    (``c_streak`` = {'dir', 'n'}) up to ``max_step``; a reversal restarts at
    ``base_step``. ``c_bracket`` = {'lo': highest c that was too low, 'hi':
    lowest c that was too high}; once both exist the target is their midpoint
    (bisection) with variance (width / 4)^2, not above the base step's (floor
    1). Inverted bounds (lo > hi: told both ways over the same c range) are
    unlocked: bracket and streak are cleared and this run's step is taken
    from the base step, so c is not held at the midpoint by contradicting
    runs. A run that crosses clears both. grow=False (shared model): always
    ``base_step``, state untouched.
    Returns (n pairs stepped up, n pairs stepped down).
    """
    store = state.setdefault(OFFSET_KEY, {}) if grow else state.get(OFFSET_KEY, {})
    cmat = applied.get('c') or []
    base_var = (base_step / 2.0) ** 2
    by_pair: Dict[str, List[Tuple[int, int, int, bool]]] = {}
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        by_pair.setdefault(key, []).append((k, i, j, forward))
    n_up = n_down = 0
    for key, edges in by_pair.items():
        entry = store.setdefault(key, {}) if grow else (store.get(key) or {})
        k = edges[0][0]
        is_pinned = pinned is not None and bool(pinned[k])
        is_stranded = stranded is not None and bool(stranded[k])
        trapped = bool(entry.get('barrier_trap')) and bool(directional[k, 0])
        if bool(directional[k, 0]) and not is_pinned and not trapped:
            bracket = entry.get('b_bracket') or {}
            lo, hi = bracket.get('lo'), bracket.get('hi')
            trapped = lo is not None and hi is not None and hi - lo <= trap_width
        up = is_pinned or trapped
        down = is_stranded and not math.isfinite(float(var[k, 1]))
        if not (up or down):
            if grow and not (is_pinned or is_stranded or bool(directional[k, 0])):
                entry.pop('c_streak', None)     # crossed: c is workable
                entry.pop('c_bracket', None)
            continue
        lo2, hi2 = min(edges[0][1], edges[0][2]), max(edges[0][1], edges[0][2])
        try:
            c = float(cmat[lo2][hi2])
        except (IndexError, TypeError, ValueError):
            continue
        direction = 1 if up else -1
        target, v = c + direction * base_step, base_var
        if grow:
            cb = entry.setdefault('c_bracket', {'lo': None, 'hi': None})
            if up:
                cb['lo'] = c if cb['lo'] is None else max(cb['lo'], c)
            else:
                cb['hi'] = c if cb['hi'] is None else min(cb['hi'], c)
            if cb['lo'] is not None and cb['hi'] is not None and cb['lo'] > cb['hi']:
                # inverted: unlock, and take this run's step from the base
                entry.pop('c_bracket', None)
                entry['c_streak'] = {'dir': direction, 'n': 1}
                entry['c_unlocks'] = int(entry.get('c_unlocks', 0)) + 1
                if counts is not None:
                    counts['c_unlocked'] = counts.get('c_unlocked', 0) + 1
            elif cb['lo'] is not None and cb['hi'] is not None:
                width = abs(cb['hi'] - cb['lo'])
                target = 0.5 * (cb['lo'] + cb['hi'])
                v = min(base_var, max((width / 4.0) ** 2, 1.0))
                entry['c_streak'] = {'dir': 0, 'n': 0}
            else:
                streak = entry.get('c_streak') or {}
                n = int(streak.get('n', 0)) if streak.get('dir', 1) == direction else 0
                target = c + direction * min(base_step * (2 ** n), max_step)
                entry['c_streak'] = {'dir': direction, 'n': n + 1}
        if direction < 0 and c >= 0:
            target = max(target, 0.0)
        for k2, _, _, _ in edges:
            targets[k2, 1] = target
            var[k2, 1] = v
            directional[k2, 1] = True
        if up:
            n_up += 1
        else:
            n_down += 1
    return n_up, n_down


def lambda_states(edge_index: torch.Tensor, saved_data: Dict, block_offset: int = 2,
                  pinned_fraction: float = PINNED_FRACTION) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per edge: was the pair's run pinned at its start, or stranded mid-way?

    pinned: lambda never left the substituent it starts on (the site's first;
    the pair's lower node in a pair combo): >= ``pinned_fraction`` of the
    pair's frames at that endpoint and no other substituent on the site
    sampled. Such a run shows the barrier holds lambda, not which side b
    favors. Only pairs that include the starting substituent can be pinned.
    stranded: no substituent on the site sampled (lambda kept between them),
    the sign of a quadratic bias that flattens too much. On a site with more
    than two substituents, a pair neither of whose substituents was sampled
    while others were is neither: lambda just never reached it.
    Returns two [E] bool tensors.
    """
    E = edge_index.size(1)
    pinned = torch.zeros(E, dtype=torch.bool)
    stranded = torch.zeros(E, dtype=torch.bool)
    pops = saved_data.get('populations') or []
    endpoints = saved_data.get('endpoint_pairs') or {}
    sites = site_nodes(edge_index)

    def _p(n):
        return pops[n] if n < len(pops) else 0
    for k in range(E):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        lo, hi = min(i, j), max(i, j)
        site = sites.get(i, [lo, hi])
        others_sampled = any(_p(n) > 0 for n in site if n not in (lo, hi))
        if _p(lo) <= 0 and _p(hi) <= 0:
            stranded[k] = not others_sampled
            continue
        if lo != site[0] or others_sampled:
            continue
        ep = endpoints.get(f"{lo + block_offset}_{hi + block_offset}") or {}
        edge = ep.get('edge') or 0
        if edge and _p(hi) <= 0 and ep.get('at_lo', 0) >= pinned_fraction * edge:
            pinned[k] = True
    return pinned, stranded


def uncrossed_pairs(edge_index: torch.Tensor, saved_data: Dict,
                    block_offset: int = 2) -> torch.Tensor:
    """[E] bool: the run sampled at least one endpoint but never crossed the pair
    (0 pair transitions). Lambda staying on one side then reflects the barrier
    as much as b (a pair can sit on its favored side behind a barrier, and a
    b step "toward the unsampled side" would then push b the wrong way), so
    such a run gives no b direction; it says c is too low. Pairs whose
    transition count is unknown are not flagged.

    On a site with more than two substituents the pair's own count is
    usually 0 (lambda moves between them through the others), so a pair
    there is flagged only when the whole site never crossed; otherwise a
    substituent lambda moved freely around but never reached says its b is
    off, which ``linear_hindsight`` already gives as a direction."""
    E = edge_index.size(1)
    out = torch.zeros(E, dtype=torch.bool)
    pops = saved_data.get('populations') or []
    sites = site_nodes(edge_index)
    for k in range(E):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        lo, hi = min(i, j), max(i, j)
        p_lo = pops[lo] if lo < len(pops) else 0
        p_hi = pops[hi] if hi < len(pops) else 0
        if p_lo <= 0 and p_hi <= 0:
            continue                      # stranded: handled by lambda_states
        site = sites.get(i, [lo, hi])
        if len(site) > 2:
            T = _site_transitions(saved_data, site, block_offset)
        else:
            T = _pair_transitions(saved_data, f"{lo + block_offset}_{hi + block_offset}")
        out[k] = T is not None and T == 0
    return out


def held_off_counts(edge_index: torch.Tensor, saved_data: Dict, block_offset: int = 2) -> int:
    """Number of (pair, endpoint) held off in this run (for logging)."""
    endpoints = saved_data.get('endpoint_pairs') or {}
    n = 0
    for k in range(edge_index.size(1)):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i < j:
            ep = endpoints.get(f"{i + block_offset}_{j + block_offset}")
            n += endpoint_held_off(ep, 'lo') + endpoint_held_off(ep, 'hi')
    return n


def preact_weights(policy, targets: torch.Tensor, var: torch.Tensor, dims: Sequence[int],
                   scale: float = 1.0) -> torch.Tensor:
    """Posterior-update weights that give each target its own noise.

    ``BayesianLinearHead.update`` treats a weight w as noise_var / w, and the
    posterior lives in pre-activation space (x = s / (1 - |s|), s = y / S), so
    a physical variance v maps to v * (dx/dy)^2 there and the matching weight
    is noise_var / (v * (dx/dy)^2). Zero where var is inf. ``targets`` /
    ``var`` columns correspond to ``dims``.
    """
    w = torch.zeros(targets.shape[0], len(dims), dtype=torch.float32)
    for col, d in enumerate(dims):
        S = float(policy.scale_factors[d])
        s = (targets[:, col] / S).clamp(-1 + 1e-3, 1 - 1e-3)
        dxdy = 1.0 / (S * (1.0 - s.abs()) ** 2)
        v_pre = var[:, col] * dxdy ** 2
        noise = float(policy.edge_mlp.mlps[d].bayesian_head.noise_var)
        w_col = torch.where(torch.isfinite(v_pre), noise / v_pre.clamp(min=1e-12),
                            torch.zeros_like(v_pre))
        w[:, col] = (scale * w_col).float()
    return w


# ── Per-pair offsets and exploration ─────────────────────────────────────────

def combo_node_labels(combo_path) -> Optional[List[Tuple[int, int]]]:
    """(original site, original sub) for each node, in node order."""
    path = Path(combo_path) / 'mapping.json'
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text())
    except Exception:
        return None
    nodes = {}
    for e in entries:
        nodes[(e['new_site'], e['new_sub'])] = (int(e['original_site']), int(e['original_sub']))
    return [nodes[k] for k in sorted(nodes)]


def pair_key(system: str, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[str, bool]:
    """Canonical pair key and whether (a -> b) is the stored direction."""
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{system}|s{lo[0]}|{lo[1]}|{hi[1]}", a <= b


def edge_slot(dim: str, forward: bool) -> Tuple[str, float]:
    """Offset slot and sign for one directed edge's coefficient.

    ``forward``: the edge runs from the pair's lower to its higher original sub.
    Linear is antisymmetric (stored lower -> higher, negated for the reverse
    edge), quadratic is shared, skew/end are separate per direction.
    """
    if dim == 'linear':
        return 'linear', (1.0 if forward else -1.0)
    if dim == 'quadratic':
        return 'quadratic', 1.0
    return f"{dim}_{'fwd' if forward else 'rev'}", 1.0


def _slot_dim(slot: str) -> str:
    return slot.split('_')[0]


def _edges(edge_index, labels, system):
    """(k, i, j, key, forward) for every edge with labelled nodes."""
    for k in range(edge_index.size(1)):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i < len(labels) and j < len(labels):
            key, forward = pair_key(system, labels[i], labels[j])
            yield k, i, j, key, forward


def record_pair_targets(state: Dict, system: str, labels: List[Tuple[int, int]],
                        edge_index: torch.Tensor, targets: torch.Tensor, var: torch.Tensor,
                        history: int = OFFSET_HISTORY, dims: Sequence[str] = BIAS_DIMS,
                        directional: Optional[torch.Tensor] = None) -> int:
    """Append each pair's finite hindsight targets to the bandit state.

    ``targets``/``var`` are [E, 4] (columns BIAS_DIMS). Linear and quadratic
    are recorded once per pair (from the src < dst edge); skew and end once per
    direction. Observations are stored as [target, variance, directional].

    A directional target (``directional`` mask from ``hindsight_targets``) is a
    step from the biases that run used, so it replaces the slot's earlier
    directional targets instead of being averaged with them: averaged, each
    new step would be diluted by targets computed from older, staler biases,
    and a pair stuck far from its working biases would creep toward them.
    Measured targets keep averaging. Returns the number of observations
    recorded.
    """
    store = state.setdefault(OFFSET_KEY, {})
    n = 0
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        entry = store.setdefault(key, {})
        for name in dims:
            if name in ('linear', 'quadratic') and i > j:
                continue
            col = OFFSET_DIMS[name]
            v = float(var[k, col])
            if not math.isfinite(v):
                continue
            slot, sign = edge_slot(name, forward)
            is_dir = bool(directional[k, col]) if directional is not None else False
            obs = entry.setdefault(slot, [])
            if is_dir:
                obs[:] = [o for o in obs if not (len(o) > 2 and o[2])]
            obs.append([sign * float(targets[k, col]), v, is_dir])
            del obs[:-history]
            n += 1
    return n


def pair_offset(mean: float, observations: List, prior_sd: float) -> float:
    """Shift from the model mean to its precision-weighted combination with the pair's targets."""
    return pair_offset_posterior(mean, observations, prior_sd)[0]


def pair_offset_posterior(mean: float, observations: List, prior_sd: float) -> Tuple[float, float]:
    """(offset mean, offset variance): the model mean as a prior of sd ``prior_sd``
    combined with the pair's observations. No observations: (0, prior_sd^2)."""
    prec = 1.0 / prior_sd ** 2
    acc = 0.0
    for o in observations or []:
        t, v = o[0], o[1]
        prec += 1.0 / v
        acc += (t - mean) / v
    return acc / prec, 1.0 / prec


def explore_factor(entry: Optional[Dict], slot: str) -> float:
    return float(((entry or {}).get('explore') or {}).get(slot, 1.0))


def apply_pair_offsets(actions: torch.Tensor, mean: torch.Tensor, state: Optional[Dict],
                       system: str, labels: Optional[List[Tuple[int, int]]],
                       edge_index: torch.Tensor,
                       prior_sd: Optional[Dict[str, float]] = None,
                       sample: bool = False,
                       base_sd: Optional[Dict[str, float]] = None,
                       dims: Sequence[str] = BIAS_DIMS,
                       rng: Optional[random.Random] = None) -> Tuple[torch.Tensor, int]:
    """Shift each edge's actions by its pair's offset (on a copy).

    ``mean`` is the model's deterministic prediction (same shape as actions);
    offsets are computed against it, so the model's own Thompson noise is kept.

    sample=False: the offset posterior mean, only for slots with observations
    (no state -> actions unchanged). sample=True: one draw per (pair, slot),
    shared by the pair's two edges, from N(offset mean, sd^2) with
    sd = base_sd x explore factor, capped by the offset posterior sd once the
    slot has observations. Slots without observations are left to the model's
    own Thompson sample until the pair's exploration has widened (factor > 1,
    after an unexplained failure); then a draw around 0 is added to it. Slots WITH observations replace
    it: proposal = model mean + offset draw, so once a pair has measured a
    coefficient its own posterior, not the shared model's noise, sets how far
    the next proposal explores.
    Returns (actions, number of edges shifted).
    """
    if not labels:
        return actions, 0
    store = (state or {}).get(OFFSET_KEY, {})
    if not sample and not store:
        return actions, 0
    prior_sd = {**OFFSET_PRIOR_SD, **(prior_sd or {})}
    base_sd = {**EXPLORE_BASE_SD, **(base_sd or {})}
    rng = rng or random

    # One canonical model mean per (pair, slot): average over the pair's edges.
    acc: Dict[Tuple[str, str], List[float]] = {}
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        for name in dims:
            slot, sign = edge_slot(name, forward)
            acc.setdefault((key, slot), []).append(sign * float(mean[k, OFFSET_DIMS[name]]))

    offsets: Dict[Tuple[str, str], float] = {}
    measured = set()
    for (key, slot), ms in acc.items():
        entry = store.get(key)
        obs = (entry or {}).get(slot)
        if not obs and not sample:
            continue
        if obs:
            measured.add((key, slot))
        dim = _slot_dim(slot)
        m = sum(ms) / len(ms)
        off_mean, off_var = pair_offset_posterior(m, obs, prior_sd[dim])
        if not sample:
            offsets[(key, slot)] = off_mean
            continue
        factor = explore_factor(entry, slot)
        if not obs and factor <= 1.0:
            continue   # the model's own Thompson sample is the exploration
        sd = base_sd[dim] * factor
        if obs:
            sd = min(sd, math.sqrt(off_var))
        offsets[(key, slot)] = (off_mean if obs else 0.0) + rng.gauss(0.0, sd)

    out = actions.clone()
    shifted = set()
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        for name in dims:
            slot, sign = edge_slot(name, forward)
            off = offsets.get((key, slot))
            if off is None:
                continue
            col = OFFSET_DIMS[name]
            base = mean[k, col] if (sample and (key, slot) in measured) else out[k, col]
            out[k, col] = base + sign * off
            shifted.add(k)
    return out, len(shifted)


def linear_edge_variance(state: Optional[Dict], system: str,
                         labels: Optional[List[Tuple[int, int]]], edge_index: torch.Tensor,
                         prior_sd: Optional[Dict[str, float]] = None) -> Optional[torch.Tensor]:
    """[E] variance of each edge's b difference after its pair's offset: the
    offset posterior variance where the pair has linear observations, else the
    prior variance (the model alone). For weighting the per-site b fit in
    ``write_variables_from_actions``; None without labels."""
    if not labels:
        return None
    store = (state or {}).get(OFFSET_KEY, {})
    sd = {**OFFSET_PRIOR_SD, **(prior_sd or {})}['linear']
    out = torch.full((edge_index.size(1),), sd ** 2, dtype=torch.float64)
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        obs = (store.get(key) or {}).get('linear')
        if obs:
            out[k] = pair_offset_posterior(0.0, obs, sd)[1]
    return out


def update_exploration(state: Dict, system: str, labels: List[Tuple[int, int]],
                       edge_index: torch.Tensor, var: torch.Tensor, saved_data: Dict,
                       factor_max: float = EXPLORE_FACTOR_MAX,
                       dims: Sequence[str] = BIAS_DIMS,
                       persistent_failures: int = PERSISTENT_FAILURES,
                       min_transitions: int = SUCCESS_MIN_TRANSITIONS,
                       min_fraction: float = SUCCESS_MIN_FRACTION) -> Tuple[int, int]:
    """Widen or narrow each pair's per-coefficient exploration after a run.

    Per pair: it failed (pair_succeeded) and the run gave no target for ANY of
    its coefficients (an unexplained failure) -> each slot's factor x2 up to
    ``factor_max``, except slots that already have observations (their own
    data sets their spread); it succeeded -> each factor / 2, down to 1. A
    single failure that gives a direction for some coefficient leaves the
    factors alone: the targets address it, and widening the others would only
    add noise to the next run.

    A pair that has failed ``persistent_failures`` runs in a row (its
    ``fail_streak``, kept by ``record_pair_outcomes``, which runs first) is
    not being fixed by its targets, so every slot widens, measured ones
    included; a measured slot's spread stays capped by its offset posterior sd
    (``apply_pair_offsets``). 0 disables this. Since b gets a target whenever
    either substituent is sampled, nearly every failure counts as explained,
    and this is what lets a stuck pair widen its search at all.
    ``min_transitions`` / ``min_fraction``: the success test (the short
    exploratory test for a short run, see ``record_pair_outcomes``).
    Returns (n widened, n narrowed).
    """
    store = state.setdefault(OFFSET_KEY, {})
    seen = set()
    widened = narrowed = 0
    # Pairs whose run gave a direction for at least one coefficient (either edge).
    explained = set()
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        if any(math.isfinite(float(var[k, OFFSET_DIMS[name]])) for name in dims):
            explained.add(key)
    sites = site_nodes(edge_index)
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        ok = pair_succeeded(saved_data, i, j, site=sites.get(i), min_transitions=min_transitions,
                            min_fraction=min_fraction)
        for name in dims:
            slot, _ = edge_slot(name, forward)
            if (key, slot) in seen:
                continue
            seen.add((key, slot))
            entry = store.setdefault(key, {})
            factors = entry.setdefault('explore', {})
            f = float(factors.get(slot, 1.0))
            persistent = (persistent_failures > 0
                          and int(entry.get('fail_streak', 0)) >= persistent_failures)
            if ok and f > 1.0:
                factors[slot] = max(1.0, f / 2.0)
                narrowed += 1
            elif not ok and f < factor_max and (
                    persistent or (key not in explained and not entry.get(slot))):
                factors[slot] = min(factor_max, f * 2.0)
                widened += 1
    return widened, narrowed


def pair_attention(state: Optional[Dict], system: str, labels: Optional[List[Tuple[int, int]]],
                   edge_index: torch.Tensor, last_failed_bonus: float = 1.0) -> float:
    """Selection bonus for a combo from its pairs' own state.

    Mean over the combo's pairs of: (largest exploration factor - 1) / (max - 1)
    (a pair widening its search) plus ``last_failed_bonus`` if the pair's most
    recent run failed. 0 for pairs never seen.
    """
    if not state or not labels:
        return 0.0
    store = state.get(OFFSET_KEY, {})
    scores = {}
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        if key in scores:
            continue
        entry = store.get(key) or {}
        f = max((entry.get('explore') or {}).values(), default=1.0)
        score = (f - 1.0) / max(EXPLORE_FACTOR_MAX - 1.0, 1e-9)
        if entry.get('last_failed'):
            score += last_failed_bonus
        scores[key] = score
    return sum(scores.values()) / len(scores) if scores else 0.0


def record_pair_outcomes(state: Dict, system: str, labels: List[Tuple[int, int]],
                         edge_index: torch.Tensor, saved_data: Dict,
                         combo: Optional[str] = None, stage: Optional[str] = None,
                         short: bool = False, short_min_transitions: int = 2,
                         short_min_fraction: float = SUCCESS_MIN_FRACTION,
                         confirm: bool = True, epoch: Optional[int] = None) -> None:
    """Store whether each pair's latest run failed (for pair_attention) and its
    consecutive failures, ``fail_streak`` (for update_exploration); count its
    training ``visits`` and ``successes`` and remember the ``combos`` (paths,
    most recent last) it ran in, for ``select_revisits``. With ``stage`` (the
    system's curriculum stage), also counts them per stage under
    ``stages[stage]``, for ``pair_resolved``.

    ``short``: the run was a short exploratory one (a stage with
    ``run_steps``). It cannot count as a success (too few frames and
    crossings for the full test); instead a pair with both substituents
    sampled, at least ``short_min_transitions`` crossings and a minority
    fraction of at least ``short_min_fraction`` (0: no balance requirement)
    is ``awaiting_confirm``: its
    next run is full length, and only that run can count as a success. A
    full-length run clears the flag either way. ``confirm=False``: no
    full-length confirmation; a short run passing the short test counts as a
    success itself.

    ``epoch``: the stage entry also counts the distinct epochs the pair ran
    in (``epochs``, ``last_epoch``), for retirement (two runs of a pair in
    one epoch, e.g. both orientations, count once). A run in which the pair
    never crossed (``uncrossed_pairs``: stuck at an endpoint) or sampled
    neither endpoint (stranded) sets ``run_both``: from then on the pair
    also runs in the reverse orientation (lambda starting on the other
    substituent), so which endpoint a barrier holds it at is not chance."""
    store = state.setdefault(OFFSET_KEY, {})
    sites = site_nodes(edge_index)
    _, stranded = lambda_states(edge_index, saved_data)
    stuck = uncrossed_pairs(edge_index, saved_data) | stranded
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        if i < j:
            entry = store.setdefault(key, {})
            if short:
                failed = not pair_succeeded(saved_data, i, j, site=sites.get(i),
                                            min_transitions=short_min_transitions,
                                            min_fraction=short_min_fraction)
                entry['awaiting_confirm'] = (not failed) and confirm
                success = (not failed) and not confirm
            else:
                failed = not pair_succeeded(saved_data, i, j, site=sites.get(i))
                entry['awaiting_confirm'] = False
                success = not failed
            entry['last_failed'] = failed
            entry['fail_streak'] = int(entry.get('fail_streak', 0)) + 1 if failed else 0
            entry['visits'] = int(entry.get('visits', 0)) + 1
            entry['successes'] = int(entry.get('successes', 0)) + (1 if success else 0)
            if bool(stuck[k]):
                entry['run_both'] = True
            if stage:
                st = entry.setdefault('stages', {}).setdefault(stage, {'visits': 0, 'successes': 0})
                st['visits'] += 1
                st['successes'] += 1 if success else 0
                if epoch is not None and st.get('last_epoch') != int(epoch):
                    st['last_epoch'] = int(epoch)
                    st['epochs'] = int(st.get('epochs', 0)) + 1
                if short:
                    st['short_promising'] = int(st.get('short_promising', 0)) + (0 if failed else 1)
            if combo:
                combos = [c for c in entry.get('combos', []) if c != combo] + [combo]
                entry['combos'] = combos[-REVISIT_COMBOS_KEPT:]


def select_revisits(state: Optional[Dict], quota: Dict[str, int],
                    allowed: Callable[[str], bool],
                    successes_needed: int = REVISIT_SUCCESSES,
                    max_visits: int = REVISIT_MAX_VISITS) -> Tuple[List[str], Dict[str, int]]:
    """Combos to rerun this epoch so pairs already in progress keep being visited.

    A pair stays in the revisit pool from its first training visit until its
    exploration is satisfactory: ``successes_needed`` successful runs
    (``retired`` = 'satisfied'), or ``max_visits`` visits without that
    ('exhausted', so a pair that cannot be freed stops holding a slot). Per
    system, up to ``quota[system]`` pairs are taken, most visited first (the
    ones closest to resolving), each with its most recent combo that
    ``allowed`` accepts. Retirements are written to the state.

    Returns (combos, counts) with counts = {'active', 'satisfied', 'exhausted'}
    over the pool after this epoch's retirements.
    """
    store = (state or {}).get(OFFSET_KEY, {})
    counts = {'active': 0, 'satisfied': 0, 'exhausted': 0}
    by_system: Dict[str, List[Tuple[int, str]]] = {}
    for key, entry in store.items():
        visits = int(entry.get('visits', 0))
        if not visits or not entry.get('combos'):
            continue
        if not entry.get('retired'):
            if int(entry.get('successes', 0)) >= successes_needed:
                entry['retired'] = 'satisfied'
            elif visits >= max_visits:
                entry['retired'] = 'exhausted'
        if entry.get('retired'):
            counts[entry['retired']] = counts.get(entry['retired'], 0) + 1
            continue
        counts['active'] += 1
        system = key.split('|', 1)[0]
        combo = next((c for c in reversed(entry['combos']) if allowed(c)), None)
        if combo is not None:
            by_system.setdefault(system, []).append((visits, combo))
    chosen: List[str] = []
    for system, items in sorted(by_system.items()):
        items.sort(key=lambda t: (-t[0], t[1]))
        chosen += [c for _, c in items[:max(0, int(quota.get(system, 0)))]]
    return chosen, counts


def merge_revisits(revisits: List[str], selected: List[str], system_of: Callable[[str], str],
                   n_total: int, min_per_system: int = 0) -> List[str]:
    """This epoch's combos: the revisits, then each system topped up to
    ``min_per_system`` from ``selected`` (in its rank order), then the rest of
    ``selected`` up to ``n_total``. Combos are matched by (system, directory
    name), since a revisit's path (the run's out_dir) can differ from the
    manifest path the selection returns for the same combo."""
    def ident(c):
        return system_of(c), Path(c).name
    out, seen = [], set()
    per_system: Dict[str, int] = {}

    def add(c):
        k = ident(c)
        if k in seen:
            return
        seen.add(k)
        out.append(c)
        per_system[k[0]] = per_system.get(k[0], 0) + 1

    for c in revisits:
        add(c)
    for c in selected:
        if len(out) >= n_total:
            break
        if per_system.get(system_of(c), 0) < min_per_system:
            add(c)
    for c in selected:
        if len(out) >= n_total:
            break
        add(c)
    return out
