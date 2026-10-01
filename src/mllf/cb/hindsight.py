"""Hindsight bias targets and per-pair offsets for online training.

A finished MSLD run measures how far its biases were from the ones that would
have worked. Each run is therefore relabelled with the corrected biases it
implies, and the policy learns from those targets. Scoring the run with a
scalar reward would say only whether the biases were good, not in which
direction or by how much they were off.

* linear: raising b[j] raises ln(p_j / p_i) by 1/kT, so the balancing
  difference for directed edge (i -> j), whose action approximates
  ``b[j] - b[i]``, is ``db* = db - kT ln(p_j / p_i)``;
* quadratic: the pair's midpoint barrier falls by ``BARRIER_C_SLOPE`` per unit
  c (see read_output.parse_pair_profile), so
  ``c* = c + (barrier - target) / |slope|``.

Iterative flattening schemes correct one system's biases from its own
simulations and start every new system from scratch. Here each corrected
target instead trains a shared model conditioned on the substituents' Uni-Mol
embeddings, so what is learned on one pair improves the first guess for pairs
and systems that have never been simulated:

* every target carries its own variance (measurement noise from the run's
  crossings or barrier, plus a term for linearization) and enters the
  NeuralLinear posterior as a properly weighted observation, rather than being
  regularized with fixed per-coefficient constants;
* the same posterior gives the epistemic uncertainty that decides which pairs
  to simulate next;
* per-pair offsets (below) combine a pair's own measurements with the model
  prediction as a Bayesian prior, so a pair the model already predicts well
  needs little correction, and one it predicts poorly is fixed from its own
  data without distorting the shared model.

Each step is also capped (a trust region), since the linearized slopes only
hold near sensible values. Skew and end are not relabelled here; their
responses are small and noisy.

Per-pair offsets hold what the shared model cannot fit for a specific pair: the
last few hindsight targets per pair are kept in the bandit state and the
proposal is shifted toward their precision-weighted mean (the model's mean
acts as a prior with sd ``offset_prior_sd``). Keys use the ORIGINAL site/sub
numbers from the combo's mapping.json, so a pair's offset follows it into any
combo; linear offsets are stored for the (lower sub -> higher sub) direction.
"""
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

KB_KCAL = 0.0019872041
DEFAULT_TEMPERATURE = 298.15

# Barrier change per unit c, from the pretraining bias-sensitivity analysis
# (see read_output.parse_pair_profile).
BARRIER_C_SLOPE = -0.233

# Trust-region caps on one correction; configurable via
# bandit.hindsight_linear_step_cap / hindsight_quadratic_step_cap. The values
# follow the step limits of ALF's flattening (1.5 x its b and c cutoffs).
LINEAR_STEP_CAP = 3.0
QUADRATIC_STEP_CAP = 12.0

# Noise floors. b: pair b* scatter around its consensus (cycle closure in the
# pretraining data). Barrier: assumed run-to-run scatter at fixed biases.
LINEAR_SD_FLOOR = 0.35
BARRIER_SD = 0.2
# Extra relative uncertainty of a linearized step (the slopes are measured
# near sensible values; large corrections overshoot).
LINEARIZATION_FRAC = 0.25

OFFSET_KEY = '_pair_offsets'
# Recent runs sit closest to the current biases, where the linearization holds.
OFFSET_HISTORY = 5
OFFSET_PRIOR_SD = {'linear': 3.0, 'quadratic': 8.0}
OFFSET_DIMS = {'linear': 0, 'quadratic': 1}


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
    clipped = max(-step_cap, min(step_cap, delta))
    var = (kT ** 2) * 4.0 / max(transitions, 1) + LINEAR_SD_FLOOR ** 2
    var += (LINEARIZATION_FRAC * clipped) ** 2
    if clipped != delta:
        var += (delta - clipped) ** 2  # capped: the true value is further out
    return db + clipped, var


def quadratic_hindsight(c: float, barrier: Optional[float], barrier_target: float,
                        slope: float = BARRIER_C_SLOPE,
                        step_cap: float = QUADRATIC_STEP_CAP) -> Tuple[float, float]:
    """Target and variance for c from the pair's midpoint barrier."""
    if not _finite(barrier):
        return c, math.inf
    delta = (barrier - barrier_target) / abs(slope)
    clipped = max(-step_cap, min(step_cap, delta))
    var = (BARRIER_SD / slope) ** 2 + (LINEARIZATION_FRAC * clipped) ** 2
    if clipped != delta:
        var += (delta - clipped) ** 2
    return c + clipped, var


def hindsight_targets(edge_index: torch.Tensor, applied: Dict[str, List], saved_data: Dict,
                      barrier_target: float, temperature: float = DEFAULT_TEMPERATURE,
                      block_offset: int = 2,
                      linear_step_cap: float = LINEAR_STEP_CAP,
                      quadratic_step_cap: float = QUADRATIC_STEP_CAP):
    """Per-edge hindsight targets for the linear and quadratic heads.

    Args:
        edge_index: [2, E] directed edges (node indices).
        applied: ``read_applied_biases`` output for the run.
        saved_data: the run's epoch_results.pt dict (populations,
            transition_pairs / transitions, barrier_pairs).
        barrier_target: the reward's barrier target (kcal/mol).

    Returns:
        (targets [E, 2], var [E, 2]) in physical units; column 0 is the
        directed linear difference b[dst] - b[src], column 1 is c. ``var`` is
        inf where the run says nothing about that pair.
    """
    kT = KB_KCAL * temperature
    E = edge_index.size(1)
    targets = torch.zeros(E, 2, dtype=torch.float64)
    var = torch.full((E, 2), math.inf, dtype=torch.float64)
    b = applied.get('b') or []
    cmat = applied.get('c') or []
    pops = saved_data.get('populations') or []
    barriers = saved_data.get('barrier_pairs') or {}
    pair_trans = saved_data.get('transition_pairs') or {}
    site_trans = saved_data.get('transitions') or []
    if isinstance(site_trans, dict):
        site_trans = list(site_trans.values())
    # Pair runs have one site: its total is the pair's count when the
    # per-pair breakdown is missing (older runs).
    fallback_trans = site_trans[0] if len(site_trans) == 1 else None

    for k in range(E):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        lo, hi = min(i, j), max(i, j)
        key = f"{lo + block_offset}_{hi + block_offset}"
        if i < len(b) and j < len(b):
            p_i = pops[i] if i < len(pops) else 0
            p_j = pops[j] if j < len(pops) else 0
            T = pair_trans.get(key, fallback_trans)
            t, v = linear_hindsight(float(b[j]) - float(b[i]), p_i, p_j, T, kT,
                                    step_cap=linear_step_cap)
            targets[k, 0], var[k, 0] = t, v
        if lo < len(cmat) and hi < len(cmat[lo]):
            t, v = quadratic_hindsight(float(cmat[lo][hi]), barriers.get(key), barrier_target,
                                       step_cap=quadratic_step_cap)
            targets[k, 1], var[k, 1] = t, v
    return targets, var


def preact_weights(policy, targets: torch.Tensor, var: torch.Tensor, dims: Sequence[int],
                   scale: float = 1.0) -> torch.Tensor:
    """Posterior-update weights that give each target its own noise.

    ``BayesianLinearHead.update`` treats a weight w as noise_var / w, and the
    posterior lives in pre-activation space (x = s / (1 - |s|), s = y / S), so
    a physical variance v maps to v * (dx/dy)^2 there and the matching weight
    is noise_var / (v * (dx/dy)^2). Zero where var is inf.
    """
    w = torch.zeros_like(targets, dtype=torch.float32)
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


# ── Per-pair offsets ─────────────────────────────────────────────────────────

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


def record_pair_targets(state: Dict, system: str, labels: List[Tuple[int, int]],
                        edge_index: torch.Tensor, targets: torch.Tensor, var: torch.Tensor,
                        history: int = OFFSET_HISTORY) -> int:
    """Append each pair's finite hindsight targets to the bandit state.

    One entry per undirected pair (the src < dst edge). Returns the number of
    (pair, dim) observations recorded.
    """
    store = state.setdefault(OFFSET_KEY, {})
    n = 0
    for k in range(edge_index.size(1)):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i >= j or i >= len(labels) or j >= len(labels):
            continue
        key, forward = pair_key(system, labels[i], labels[j])
        entry = store.setdefault(key, {})
        for name, col in OFFSET_DIMS.items():
            v = float(var[k, col])
            if not math.isfinite(v):
                continue
            t = float(targets[k, col])
            if name == 'linear' and not forward:
                t = -t
            obs = entry.setdefault(name, [])
            obs.append([t, v])
            del obs[:-history]
            n += 1
    return n


def pair_offset(mean: float, observations: List, prior_sd: float) -> float:
    """Shift from the model mean to its precision-weighted combination with the pair's targets."""
    if not observations:
        return 0.0
    prec = 1.0 / prior_sd ** 2
    acc = 0.0
    for t, v in observations:
        prec += 1.0 / v
        acc += (t - mean) / v
    return acc / prec


def apply_pair_offsets(actions: torch.Tensor, mean: torch.Tensor, state: Optional[Dict],
                       system: str, labels: Optional[List[Tuple[int, int]]],
                       edge_index: torch.Tensor,
                       prior_sd: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, int]:
    """Shift linear/quadratic actions by each pair's offset (in place on a copy).

    ``mean`` is the model's deterministic prediction (same shape as actions);
    the offset is computed against it so exploration noise is kept. Returns
    (actions, number of edges shifted).
    """
    if not state or not labels or OFFSET_KEY not in state:
        return actions, 0
    prior_sd = {**OFFSET_PRIOR_SD, **(prior_sd or {})}
    store = state[OFFSET_KEY]
    out = actions.clone()
    n = 0
    for k in range(edge_index.size(1)):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i >= len(labels) or j >= len(labels):
            continue
        key, forward = pair_key(system, labels[i], labels[j])
        entry = store.get(key)
        if not entry:
            continue
        shifted = False
        for name, col in OFFSET_DIMS.items():
            obs = entry.get(name)
            if not obs:
                continue
            sign = 1.0 if (name != 'linear' or forward) else -1.0
            m = sign * float(mean[k, col])
            off = pair_offset(m, obs, prior_sd[name])
            out[k, col] = out[k, col] + sign * off
            shifted = True
        n += shifted
    return out, n
