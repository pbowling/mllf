"""What a simulation measured for each pair of substituents.

Per-edge confidence weights (population balance where the pair's DDG is
finite) and per-bias-type "measured" masks, shared by pretraining
(``pretrain_policy``), the hindsight regression (``hindsight_regression``)
and the online REINFORCE / behavioral-cloning updates (``reinforce_trainer``).
"""
import math
from typing import Dict, List, Optional

import numpy as np


def is_finite(value) -> bool:
    """True for a number (or numeric string) that is not NaN or infinite;
    False for None and anything that is not a number."""
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def compute_pairwise_confidence_weights(
    sim_results: Dict,
    nsubs_per_site: List[int],
    pairs: List[tuple],
) -> Optional[List[float]]:
    """Compute per-edge confidence weights from pairwise population balance and DDG data.

    For each directed edge (i, j), extracts populations and DDG values from simulation results
    to compute a confidence weight. This weight reflects how well-sampled and reliable the
    pairwise interaction is.

    **Confidence Computation:**
    - If either population is 0 or missing: weight = 0 (no data)
    - If DDG value is NaN or infinite: weight = 0 (unreliable)
    - Otherwise: weight = min(pop_i, pop_j) / max(pop_i, pop_j)
      (population balance, ranges 0-1, perfect when populations are equal)

    **Node to Block Mapping:**
    MSLD block numbering starts at 1 (reference), with block 2 being the first real block.
    In the graph, nodes are 0-indexed (node 0 = block 2, node 1 = block 3, etc.).

    Args:
        sim_results: Dict with 'populations' and 'ddg_pairs' from simulation_results.json
        nsubs_per_site: List of number of substituents per site (for validation)
        pairs: List of (i, j) directed edge tuples from build_directed_pairs()

    Returns:
        List of per-edge confidence weights [0.0, 1.0], or None if cannot extract populations
    """
    populations = sim_results.get("populations", {})
    ddg_pairs = sim_results.get("ddg_pairs", {})

    if not populations:
        return None

    # Extract population counts (use only HIGHEST lambda value, matching reward logic)
    pop_dict = {}  # node_idx (0-based) -> population count
    for block_id_str in populations.keys():
        try:
            block_id = int(block_id_str)
        except ValueError:
            continue

        # Convert block ID to node index: node_idx = block_id - 2
        # (block 2 = node 0, block 3 = node 1, etc.)
        node_idx = block_id - 2
        if node_idx < 0:
            continue

        block_data = populations[block_id_str]
        counts = block_data.get("counts", {})
        if counts:
            # Use only the highest lambda value (same as reward computation)
            max_lambda = max(counts.keys(), key=lambda x: float(x))
            pop_dict[node_idx] = counts[max_lambda]

    if not pop_dict:
        return None

    # Compute confidence weight for each edge
    weights = []
    for (i, j) in pairs:
        pop_i = pop_dict.get(i, 0)
        pop_j = pop_dict.get(j, 0)

        # Zero weight if either population is missing or zero
        if pop_i <= 0 or pop_j <= 0:
            weights.append(0.0)
            continue

        # Check DDG value for this pair
        # DDG key format: "{i+2}_{j+2}" (convert node indices back to block IDs)
        # For reverse pairs: ΔG(j→i) = -ΔG(i→j), so if forward key missing,
        # look for reverse key and negate its value
        ddg_key_fwd = f"{i+2}_{j+2}"
        ddg_key_rev = f"{j+2}_{i+2}"

        ddg_value = None
        if ddg_key_fwd in ddg_pairs:
            ddg_value = ddg_pairs[ddg_key_fwd]
        elif ddg_key_rev in ddg_pairs:
            # Use negation of reverse pair (antisymmetric property)
            reverse_val = ddg_pairs[ddg_key_rev]
            try:
                ddg_value = -float(reverse_val)
            except (ValueError, TypeError):
                ddg_value = None

        # Zero weight if DDG is missing, NaN, or infinite
        if ddg_value is None:
            weights.append(0.0)
            continue

        try:
            ddg_val = float(ddg_value)
            if not (np.isfinite(ddg_val)):  # catches NaN and inf
                weights.append(0.0)
                continue
        except (ValueError, TypeError):
            weights.append(0.0)
            continue

        # Population balance weight: min / max (symmetric around 0.5 for equal pops)
        balance = min(pop_i, pop_j) / max(pop_i, pop_j)
        weights.append(float(balance))

    return weights


# Signal that has to have been measured for each bias type's target to count
# (order matches the policy's [linear, quadratic, skew, end] outputs; None =
# the linear rule in compute_pairwise_confidence_weights).
MEASURED_SIGNAL = (None, 'barrier_pairs', 'skew_pairs', 'end_pairs')


def compute_pairwise_measured_mask(
    sim_results: Dict,
    pairs: List[tuple],
    linear_weights: Optional[List[float]],
) -> List[List[float]]:
    """Per-edge, per-bias-type mask: 1 where the run measured what sets that target.

    A run's biases are only evidence for a coefficient when the simulation
    actually sampled the part of lambda space that coefficient shapes; a
    trapped or diverging run leaves the others wherever the optimizer that
    produced them happened to stop (e.g. end biases of 10-18 in flattening
    runs that never reached their endpoints). So each target counts only if:

    * linear: both substituents were sampled and the pair's DDG is finite
      (``linear_weights`` > 0, from ``compute_pairwise_confidence_weights``);
    * quadratic: the pair's midpoint barrier was measured;
    * skew / end: the pair's skew / end asymmetry was measured (these need
      frames at both endpoint cores; see read_output.parse_pair_profile).

    Runs without the profile fields (collected before they were parsed) get 0
    for quadratic/skew/end: there is no evidence those targets were sampled.

    Returns:
        [E][4] list of 0.0 / 1.0.
    """
    mask = []
    for k, (i, j) in enumerate(pairs):
        lo, hi = min(i, j) + 2, max(i, j) + 2
        row = [1.0 if (linear_weights is not None and linear_weights[k] > 0) else 0.0]
        for field in MEASURED_SIGNAL[1:]:
            values = sim_results.get(field) or {}
            v = values.get(f"{lo}_{hi}", values.get(f"{hi}_{lo}"))
            row.append(1.0 if is_finite(v) else 0.0)
        mask.append(row)
    return mask
