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
from typing import Dict, List, Optional, Sequence, Tuple

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
                     kT: float, step_cap: float = LINEAR_STEP_CAP) -> Tuple[float, float]:
    """Target and variance for db = b[j] - b[i] from the pair's populations.

    Both sampled and crossing: the balancing value, with variance from the
    number of crossings (each side visited ~T/2 times). One side unsampled: a
    capped step toward it (a direction, not a value) with sd step_cap / 2, so
    against the offset prior (sd 3) it moves ~80% of the step, and a later
    two-sided measurement outweighs it ~6x.
    Neither sampled: no information (variance inf).
    """
    if p_i <= 0 and p_j <= 0:
        return db, math.inf
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


def pair_succeeded(saved_data: Dict, i: int, j: int, block_offset: int = 2,
                   min_fraction: float = SUCCESS_MIN_FRACTION,
                   min_transitions: int = SUCCESS_MIN_TRANSITIONS) -> bool:
    """Balanced (minority fraction) and crossing (pair transitions) in this run."""
    pops = saved_data.get('populations') or []
    p_i = pops[i] if i < len(pops) else 0
    p_j = pops[j] if j < len(pops) else 0
    if p_i <= 0 or p_j <= 0 or min(p_i, p_j) / (p_i + p_j) < min_fraction:
        return False
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
            t, v = linear_hindsight(float(b[j]) - float(b[i]), p_i, p_j, T, kT,
                                    step_cap=linear_step_cap)
            targets[k, 0], var[k, 0] = t, v
            directional[k, 0] = math.isfinite(v) and (p_i <= 0 or p_j <= 0 or not T)
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
                             var: Optional[torch.Tensor] = None) -> Tuple[int, int]:
    """Directional b targets for trapped pairs: grow the step, then bracket (in place).

    ``targets``/``directional`` are ``hindsight_targets`` outputs, whose
    directional b targets are steps of ``base_step``. Per pair the bandit state
    keeps, in the pair's lower -> higher original-sub orientation:

    * ``b_streak`` = {'dir': +1/-1, 'n': consecutive steps that way}: a step in
      the same direction as the last becomes base_step x 2^n (capped at
      ``max_step``); a reversal restarts at base_step;
    * ``b_bracket`` = {'lo', 'hi'}: a run stuck needing a higher b sets a lower
      bound (the largest such b), one needing a lower b an upper bound (the
      smallest). Once both exist the working b lies between them, so the
      target is the midpoint (bisection) instead of another step, and, if
      ``var`` is given, its variance shrinks to (bracket width / 4)^2 (not
      above the base step's). Overlapping bounds (stuck on either side over
      the same b range) mean the barrier, not b, traps lambda; the midpoint is
      still the best b guess.

    Any non-directional run (both substituents sampled) clears both. Returns
    (n pairs whose step grew, n pairs targeted by bracket midpoint).
    """
    store = state.setdefault(OFFSET_KEY, {})
    b = applied.get('b') or []
    grown = bracketed = 0
    by_pair: Dict[str, List[Tuple[int, int, int, bool]]] = {}
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        by_pair.setdefault(key, []).append((k, i, j, forward))
    for key, edges in by_pair.items():
        entry = store.setdefault(key, {})
        k, i, j, forward = edges[0]
        if i >= len(b) or j >= len(b):
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
    return grown, bracketed


def barrier_trapped_c_targets(state: Dict, system: str, labels: List[Tuple[int, int]],
                              edge_index: torch.Tensor, targets: torch.Tensor,
                              var: torch.Tensor, directional: torch.Tensor,
                              applied: Dict[str, List],
                              base_step: float = QUADRATIC_STEP_CAP,
                              max_step: float = QUADRATIC_STEP_MAX,
                              trap_width: float = BARRIER_TRAP_WIDTH,
                              grow: bool = True) -> int:
    """Directional c targets for pairs trapped by their barrier (in place).

    A pair is barrier-trapped when this run did not cross it (``directional``
    b) and its ``b_bracket`` (see ``grow_directional_b_steps``, run first) has
    both bounds within ``trap_width``: b is already near its working value,
    so b steps cannot free it. Its c target becomes c + step with variance
    (step / 2)^2, and ``directional[:, 1]`` is set, so the step replaces the
    pair's earlier directional c targets.

    grow=True (per-pair offsets): the step doubles over consecutive trapped
    runs (``c_streak``) up to ``max_step``; a run that crosses clears the
    streak. grow=False (shared model): always ``base_step``, state untouched.
    Returns the number of pairs targeted.
    """
    store = state.setdefault(OFFSET_KEY, {}) if grow else state.get(OFFSET_KEY, {})
    cmat = applied.get('c') or []
    by_pair: Dict[str, List[Tuple[int, int, int, bool]]] = {}
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        by_pair.setdefault(key, []).append((k, i, j, forward))
    n = 0
    for key, edges in by_pair.items():
        entry = store.get(key) or {}
        k = edges[0][0]
        if not bool(directional[k, 0]):
            if grow:
                entry.pop('c_streak', None)
            continue
        bracket = entry.get('b_bracket') or {}
        lo, hi = bracket.get('lo'), bracket.get('hi')
        if lo is None or hi is None or hi - lo > trap_width:
            continue
        size = base_step
        if grow:
            streak = int((entry.get('c_streak') or {}).get('n', 0))
            size = min(base_step * (2 ** streak), max_step)
            entry['c_streak'] = {'n': streak + 1}
        targeted = False
        for k2, i2, j2, _ in edges:
            lo2, hi2 = min(i2, j2), max(i2, j2)
            try:
                c = float(cmat[lo2][hi2])
            except (IndexError, TypeError, ValueError):
                continue
            targets[k2, 1] = c + size
            var[k2, 1] = (size / 2.0) ** 2
            directional[k2, 1] = True
            targeted = True
        n += targeted
    return n


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


def update_exploration(state: Dict, system: str, labels: List[Tuple[int, int]],
                       edge_index: torch.Tensor, var: torch.Tensor, saved_data: Dict,
                       factor_max: float = EXPLORE_FACTOR_MAX,
                       dims: Sequence[str] = BIAS_DIMS,
                       persistent_failures: int = PERSISTENT_FAILURES) -> Tuple[int, int]:
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
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        ok = pair_succeeded(saved_data, i, j)
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
                         edge_index: torch.Tensor, saved_data: Dict) -> None:
    """Store whether each pair's latest run failed (for pair_attention) and its
    consecutive failures, ``fail_streak`` (for update_exploration)."""
    store = state.setdefault(OFFSET_KEY, {})
    for k, i, j, key, forward in _edges(edge_index, labels, system):
        if i < j:
            entry = store.setdefault(key, {})
            failed = not pair_succeeded(saved_data, i, j)
            entry['last_failed'] = failed
            entry['fail_streak'] = int(entry.get('fail_streak', 0)) + 1 if failed else 0
