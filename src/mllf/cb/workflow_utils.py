"""Workflow utility functions for training with SLURM job management.

This module contains reusable functions for:
- Fixing system-specific simulation scripts
- Managing simulation success checking and metric parsing
- SLURM job submission and monitoring utilities

Note: For manifest loading, use mllf.cli.workflow.load_manifest()
"""
import math
from pathlib import Path
from typing import List, Dict, Optional
import json
import re
import torch

# Re-export load_manifest from cli.workflow for convenience
from mllf.cli.workflow import load_manifest


def fix_msld_flat_for_single_site(combo_path: Path, 
                                   site_atoms: Dict[int, str] = None) -> bool:
    """Modify msld_flat.py to delete only atoms that overlap with present sites.
    
    For multi-site λ-dynamics simulations, certain atoms in the base structure
    may overlap with specific substituents. When only a subset of sites are
    present in a combination, we should only delete atoms that would overlap
    with the present sites.
    
    This function reads mapping.json to determine which original sites are
    present, then modifies the atom deletion command in msld_flat.py accordingly.
    
    Example usage:
        # 14benz system
        site_atoms = {1: 'C4 H4', 2: 'C5 H5'}
        fix_msld_flat_for_single_site(combo_path, site_atoms)
        
        # Indole system
        site_atoms = {1: 'C2 H2', 2: 'C6 H6'}
        fix_msld_flat_for_single_site(combo_path, site_atoms)
    
    Args:
        combo_path: Path to combination directory containing msld_flat.py
            and mapping.json.
        site_atoms: Dictionary mapping site numbers to atom selection strings.
            Example: {1: 'C4 H4', 2: 'C5 H5'} for 14benz system.
            If None, defaults to 14benz atoms for backward compatibility.
    
    Returns:
        True if file was modified, False if skipped or unchanged.
    """
    # Default to 14benz atoms for backward compatibility
    if site_atoms is None:
        site_atoms = {1: 'C4 H4', 2: 'C5 H5'}
    msld_flat = combo_path / 'msld_flat.py'
    if not msld_flat.exists():
        return False
    
    # Read mapping.json to determine ORIGINAL sites present
    mapping_file = combo_path / 'mapping.json'
    if not mapping_file.exists():
        return False
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Extract unique original site numbers from entries that have site info
    original_sites = set()
    for entry in mapping:
        site = entry.get('original_site')
        if site is not None:
            original_sites.add(site)
    
    # Determine what to delete based on original sites
    # Only modify if exactly one site is present
    if len(original_sites) != 1:
        return False  # Either no sites identified or multiple sites - leave unchanged
    
    original_site = list(original_sites)[0]
    
    # Determine which atoms to delete based on ORIGINAL site number
    atoms_to_delete = site_atoms.get(original_site)
    if atoms_to_delete is None:
        # Site not in mapping - leave as is
        return False
    
    # Read and modify the msld_flat.py content
    content = msld_flat.read_text()
    
    # Replace the delete line - handle both original and already-modified versions
    # Pattern matches any combination of atoms in the selection
    old_pattern = r"select\.store_selection\('todelete',pycharmm\.SelectAtoms\(\)\.by_res_and_type\(ligseg,resnum,'[^']+'\)\)"
    new_line = f"select.store_selection('todelete',pycharmm.SelectAtoms().by_res_and_type(ligseg,resnum,'{atoms_to_delete}'))"
    
    new_content = re.sub(old_pattern, new_line, content)
    
    # Only write if something changed
    if new_content != content:
        msld_flat.write_text(new_content)
        return True
    
    return False


def check_simulation_success(output_file: Path) -> bool:
    """Check if a simulation completed successfully.
    
    Args:
        output_file: Path to simulation output file.
    
    Returns:
        True if simulation terminated normally, False otherwise.
    """
    from mllf.file_handling.read_output import terminated_normally
    
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
        return terminated_normally(output_text)
    except Exception:
        return False


def parse_simulation_metrics(output_file: Path) -> Dict[str, List]:
    """Parse raw populations, transitions, and per-pair DDG data from simulation output.
    
    This is a convenience wrapper that extracts aggregate metrics
    (total populations per block, total transitions per site, per-pair DDG presence)
    for use in reward computation and per-pair credit assignment.
    
    Args:
        output_file: Path to simulation output file.
    
    Returns:
        Dict with:
          'populations': list of population counts per block at highest lambda
          'transitions': list of transition counts per site at highest lambda
          'ddg_pairs': dict mapping "blk_i_blk_j" → float|None at highest lambda
                       (None = NaN = no crossings between that pair)
          'fraction_physical': float|None, fraction of trajectory spent in a
                       fully-resolved ("physical") ligand state at highest
                       lambda (None if the diagnostic isn't in the output)
    """
    from mllf.file_handling.read_output import (
        parse_single_population,
        parse_transitions_and_rates,
        parse_single_ddg,
        parse_fraction_physical_ligand,
    )
    
    raw_metrics = {'populations': [], 'transitions': [], 'ddg_pairs': {}, 'fraction_physical': None}
    
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
        
        population_data = parse_single_population(output_text)
        transitions_data, _ = parse_transitions_and_rates(output_text)
        ddg_data = parse_single_ddg(output_text)
        raw_metrics['fraction_physical'] = parse_fraction_physical_ligand(output_text)
        
        # Extract populations per block - use only HIGHEST lambda value (0.990)
        for block_id, block_info in population_data.items():
            counts_dict = block_info.get('counts', {})
            if counts_dict:
                # Use only the highest lambda value
                max_lambda = max(counts_dict.keys(), key=lambda x: float(x))
                raw_metrics['populations'].append(counts_dict[max_lambda])
            else:
                raw_metrics['populations'].append(0)
        
        # Extract transitions per site - use only HIGHEST lambda value (0.990)
        for site_id, trans_dict in transitions_data.items():
            if trans_dict:
                # Use only the highest lambda value
                max_lambda = max(trans_dict.keys(), key=lambda x: float(x))
                raw_metrics['transitions'].append(trans_dict[max_lambda])
            else:
                raw_metrics['transitions'].append(0)

        # Store per-pair DDG with string keys ("blk_i_blk_j") for serialisation
        raw_metrics['ddg_pairs'] = {
            f"{lo}_{hi}": val for (lo, hi), val in ddg_data.items()
        }
    except Exception:
        pass
    
    return raw_metrics


def build_edge_weights(
    edge_index: torch.Tensor,
    ddg_pairs: dict,
    no_transition_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Per-edge weight tensor derived from per-pair DDG transition data.

    Edges whose substituent pair had no observed lambda-space transitions
    (NaN DDG → None in ddg_pairs) get *no_transition_weight*; edges where
    transitions were observed get 1.0.  When *ddg_pairs* is empty (data not
    available for this run) every edge gets 1.0 so the loss is unchanged.

    Block ID mapping: block_id = node_idx + 2 (block 1 = reference).

    Args:
        edge_index: [2, E] node-index tensor.
        ddg_pairs: dict from simulation_results 'ddg_pairs': {"blk_i_blk_j": float|None}.
        no_transition_weight: weight for no-transition pairs (default 0.2).
        device: target torch device.

    Returns:
        Float tensor of shape [E].
    """
    if not ddg_pairs:
        return torch.ones(edge_index.size(1), device=device)

    weights: list = []
    for k in range(edge_index.size(1)):
        src = int(edge_index[0, k].item())
        dst = int(edge_index[1, k].item())
        lo = min(src + 2, dst + 2)
        hi = max(src + 2, dst + 2)
        entry = ddg_pairs.get(f"{lo}_{hi}", "missing")
        # None      → NaN or Inf (no usable crossings) → down-weight
        # finite float → transitions observed            → full weight
        # "missing" → no DDG data at all                → full weight (don't penalise old data)
        no_crossing = entry is None or (isinstance(entry, float) and math.isinf(entry))
        weights.append(no_transition_weight if no_crossing else 1.0)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_pair_reward(
    edge_index: torch.Tensor,
    ddg_pairs: dict,
    populations: list,
    total_transitions: int = 0,
    t_baseline: float = 50.0,
    block_offset: int = 2,
    fraction_physical: Optional[float] = None,
) -> torch.Tensor:
    """Compute per-edge, per-dimension reward tensor for credit assignment.

    Returns a separate reward signal for each of the 4 MSLD bias types so that
    each MLP head receives a gradient signal matched to the physical quantity it
    controls.  Dimension assignments mirror the policy output order:

      0  linear    — population balance between substituents.
                     Signal: ``minority_frac = min(pop_i,pop_j)/(pop_i+pop_j+ε)``
                     ∈ [0, 0.5].  Higher = more equal sampling = better.
      1  quadratic — barrier height.  Uses two signals in tandem:
                     (a) Per-pair DDG existence (binary 1.0 when this specific
                         pair's DDG is finite) — precise even with 3+ subs,
                         because each pair's DDG is checked independently.
                     (b) Normalized total transitions ``min(n_trans,T_base)/T_base``
                         — a continuous quality signal shared across the combo.
                     Combined as ``(pair_visited + trans_quality) / 2`` ∈ [0.5, 1.0]
                     when the pair was visited, giving both per-pair resolution and
                     a continuous quality gradient.
      2  skew      — barrier asymmetry from soft-core introduction.
                     Signal: ``combined`` — mean of the population-balance and
                     quadratic-quality proxies (previously shared with end;
                     cannot yet be further isolated from available data).
      3  end       — entropic / surface-tension cost of creating substituent space.
                    Signal: ``min(1.0, fraction_physical * total_pairs)`` — the
                     combo-level FRACTION PHYSICAL LIGAND diagnostic (fraction
                     of the trajectory spent in a fully-resolved, single-
                     substituent-per-site state) scaled by the number of
                     possible substituent pairs in the combo. Scaling
                     counteracts the combinatorial shrinkage of "physical"
                     time as more substituents compete for occupancy, so
                     combos of different sizes are graded comparably. Falls
                     back to the ``combined`` proxy (see dim 3) when
                     ``fraction_physical`` is unavailable (e.g. a cached
                     epoch_results.pt from before this diagnostic was parsed).

    For all dimensions: ``-1.0`` when this specific pair was not visited
    (DDG is None/NaN/Inf) OR when both substituents have zero population
    (neither was ever sampled by REMD).  With 3+ substituents per site this
    correctly penalises edges whose pair was never sampled even when other
    pairs contributed many transitions, and also catches the failure case
    where the sampling biases failed to explore one or both substituents.

    Args:
        edge_index: [2, E] node-index tensor.
        ddg_pairs: dict from simulation_results 'ddg_pairs':
                   keys ``"blk_lo_blk_hi"`` → float | None.
        populations: list of raw highest-lambda population counts per block
                     (Block ID = node_idx + block_offset).
        total_transitions: total lambda-space crossings observed for this combo
                           (sum of per-site transition counts).  Used as a
                           continuous quality modifier for the quadratic signal;
                           the per-pair DDG check provides per-pair resolution.
        t_baseline: normalization constant for transition rate (default 50.0,
                    matching ``T_baseline`` in the reward config).
        block_offset: integer offset from node index to block ID (default 2).
        fraction_physical: combo-level FRACTION PHYSICAL LIGAND value at the
                    highest lambda (0.990), from ``parse_simulation_metrics``.
                    None if the diagnostic wasn't present in the output.

    Returns:
        Float tensor of shape ``[E, 4]`` with per-dimension per-edge rewards.
    """
    num_edges = edge_index.size(1)
    rewards = torch.zeros(num_edges, 4, dtype=torch.float32)

    # Combo-level transition quality: continuous, shared across all edges.
    trans_quality = float(min(total_transitions, t_baseline)) / float(t_baseline)
    # Combo-level pair count: build_directed_pairs() emits both (i,j) and
    # (j,i) per unordered pair, so undirected pairs = directed edges / 2.
    total_pairs = num_edges // 2

    for k in range(num_edges):
        src = int(edge_index[0, k].item())
        dst = int(edge_index[1, k].item())
        lo = min(src + block_offset, dst + block_offset)
        hi = max(src + block_offset, dst + block_offset)
        entry = ddg_pairs.get(f"{lo}_{hi}")

        # Fetch populations for this edge
        pop_i = populations[src] if src < len(populations) else 0
        pop_j = populations[dst] if dst < len(populations) else 0

        # Per-pair DDG existence: True only when THIS pair was never visited.
        # With 3+ subs, total_transitions may be non-zero while this specific
        # pair has no DDG — the per-pair check catches that correctly.
        # Also check for zero populations: if both subs were never sampled,
        # this pair couldn't possibly have been visited (DDG=None).
        no_crossing = (
            entry is None
            or (isinstance(entry, float) and (math.isinf(entry) or math.isnan(entry)))
            or (pop_i == 0 and pop_j == 0)  # Both subs never sampled = failure
        )
        if no_crossing:
            rewards[k, :] = -1.0
        else:
            minority_frac = min(pop_i, pop_j) / (pop_i + pop_j + 1e-8)

            # Quadratic: tandem of per-pair visited binary (1.0, since we are
            # inside the else branch) and combo-level quality → ∈ [0.5, 1.0].
            pair_visited = 1.0
            quad_signal = (pair_visited + trans_quality) / 2.0

            # Combined proxy for skew: population balance (normalised to
            # [0,1]) and quad_signal averaged.
            combined = (minority_frac * 2.0 + quad_signal) / 2.0

            # End: distinct signal from the FRACTION PHYSICAL LIGAND
            # diagnostic, scaled by the combo's total possible substituent
            # pairs (clamped to 1.0 since the scaling can overshoot for
            # combos with many pairs). Falls back to `combined` when the
            # diagnostic wasn't parsed (older cached runs).
            if fraction_physical is not None:
                end_signal = min(1.0, float(fraction_physical) * total_pairs)
            else:
                end_signal = combined

            rewards[k, 0] = minority_frac   # linear:    population balance ∈ [0, 0.5]
            rewards[k, 1] = quad_signal     # quadratic: per-pair visited + quality ∈ [0.5, 1]
            rewards[k, 2] = combined        # skew:      combined proxy
            rewards[k, 3] = end_signal      # end:       fraction-physical-based proxy

    return rewards


def load_bandit_state(path: Path) -> Dict[str, Dict]:
    """Load the per-combo NeuralLinear+TS bookkeeping state (retries/deferrals).

    The state tracks, per combo, how many times it has failed for
    infrastructure/software reasons (not simulator-sampling reasons — those are
    already filtered out before this function's caller sees them; see
    ``record_combo_failure``) so uncertainty-driven combo selection can stop
    repeatedly re-selecting a combo that simply cannot be simulated.

    Args:
        path: Path to the JSON state file (created on first save if missing).

    Returns:
        Dict mapping combo key (``str(combo_path)``) -> {"software_failures":
        int, "deferred": bool}. Empty dict if the file doesn't exist yet.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_bandit_state(path: Path, state: Dict[str, Dict]) -> None:
    """Persist the per-combo NeuralLinear+TS bookkeeping state to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


def record_combo_failure(state: Dict[str, Dict], combo_key: str, cap: int = 3) -> None:
    """Record one infrastructure/software failure for a combo, deferring it past ``cap``.

    Mutates *state* in place. Once a combo has failed ``cap`` or more times it
    is marked ``deferred`` — excluded from uncertainty-driven selection (see
    ``select_combos_by_uncertainty``) without being discarded outright, so the
    active-learning loop doesn't spend its whole budget retrying one broken
    combo (guardrail from the NeuralLinear+TS design note: a software failure
    is a missing observation, not evidence the combo is a bad arm).

    Args:
        state: Bandit state dict (as returned by ``load_bandit_state``).
        combo_key: Combo identifier (``str(combo_path)``).
        cap: Number of failures after which the combo is deferred (default 3).
    """
    entry = state.setdefault(combo_key, {"software_failures": 0, "deferred": False})
    entry["software_failures"] = entry.get("software_failures", 0) + 1
    if entry["software_failures"] >= cap:
        entry["deferred"] = True


def select_combos_by_uncertainty(
    candidates: List[str],
    policy,
    combo_system_map: Dict[str, Dict],
    config: Dict,
    n_select: int,
    bandit_state: Dict[str, Dict],
    device: str,
    build_graph_fn,
) -> List[str]:
    """Rank candidate combos by predictive uncertainty and return the top ``n_select``.

    Implements the "uncertainty-driven selection" half of NeuralLinear + Thompson
    Sampling: train next on the combos the model is least sure about. Callers are
    expected to have already applied the simulator-tractability gate (the
    existing curriculum/transition filters) to *candidates* — this function only
    ranks within whatever pool it's given, plus the deferred-backfill fallback
    below.

    Args:
        candidates: Combo path strings to rank (already curriculum/transition
            filtered — the "simulator-tractability gate").
        policy: A ``UnimolPolicy`` with ``use_bayesian_heads=True``.
        combo_system_map: combo path str -> per-system config dict.
        config: Full workflow config (used to build each candidate's graph).
        n_select: Number of combos to return.
        bandit_state: Per-combo failure/defer tracker (see ``load_bandit_state``);
            deferred combos are skipped unless needed as a last-resort backfill.
        device: torch device string for graph building / inference.
        build_graph_fn: Callable ``(combo_dir_str, sys_cfg, config, device) ->
            (data, extras)`` — injected rather than imported directly to avoid a
            circular import with ``training/workflow.py`` (typically
            ``build_graph_and_data``).

    Returns:
        List of up to ``n_select`` combo path strings, highest predictive
        variance first. If fewer than ``n_select`` non-deferred candidates are
        eligible, backfills from the deferred pool (oldest-first by failure
        count) and logs how many were backfilled rather than silently
        truncating the epoch's combo pool.
    """
    eligible = [c for c in candidates if not bandit_state.get(c, {}).get('deferred', False)]
    deferred = [c for c in candidates if bandit_state.get(c, {}).get('deferred', False)]

    scored: List[tuple] = []
    n_skipped = 0
    with torch.no_grad():
        for combo_dir in eligible:
            sys_cfg = combo_system_map.get(combo_dir, {})
            try:
                data, _ = build_graph_fn(combo_dir, sys_cfg, config, device)
                _, var = policy.predict_uncertainty(data.x, data.edge_index, data.edge_type)
                score = float(var.mean().item())
            except Exception:
                n_skipped += 1
                continue
            scored.append((score, combo_dir))

    if n_skipped:
        print(f"  select_combos_by_uncertainty: {n_skipped}/{len(eligible)} candidates "
              f"could not be scored (graph build failed) — excluded from ranking")

    scored.sort(key=lambda t: t[0], reverse=True)
    selected = [c for _, c in scored[:n_select]]

    if len(selected) < n_select and deferred:
        n_backfill = min(n_select - len(selected), len(deferred))
        backfilled = sorted(
            deferred,
            key=lambda c: bandit_state.get(c, {}).get('software_failures', 0),
        )[:n_backfill]
        print(f"  select_combos_by_uncertainty: only {len(selected)} eligible combos "
              f"scored; backfilling {len(backfilled)} deferred combo(s) to reach "
              f"n_select={n_select}: {[Path(c).name for c in backfilled]}")
        selected.extend(backfilled)

    return selected


def compute_failure_reward(
    edge_index: torch.Tensor,
    populations: list,
    failure_advantage: float = -2.0,
    block_offset: int = 2,
) -> torch.Tensor:
    """Compute per-edge, per-dimension advantages for below-floor (failed) combos.

    When a combo produces fewer transitions than the floor threshold, we cannot
    compute a meaningful DDG-based reward.  However, the REMD *population* data
    is still valid: it reflects how much time the system spent in each state
    regardless of crossing events.

    This function constructs a physically grounded per-dimension signal that
    avoids penalising dimensions whose actions may have been correct:

      - Dim 0 (linear): calibrated by population balance.  If ``minority_frac``
        is near 0.5 (populations balanced), the linear bias was probably fine and
        receives a near-zero advantage.  If ``minority_frac`` is near 0 (system
        pinned in one state), it receives the full ``failure_advantage``.
        Formula: ``failure_advantage * (1 - 2 * minority_frac)``  ∈
        [failure_advantage, 0].

      - Dim 1 (quadratic): fixed ``failure_advantage``.  No transitions always
        means the quadratic barrier was too high, regardless of populations.

      - Dims 2, 3 (skew, end): partial penalty combining linear calibration
        (half-weight) with the full quadratic failure.  Cannot be isolated from
        available data, but should not receive the full penalty when the
        population is already balanced.
        Formula: ``(failure_advantage * (1 - 2*minority_frac) + failure_advantage) / 2``
                  ``= failure_advantage * (1 - minority_frac)``  ∈
        [failure_advantage, failure_advantage/2].

    Args:
        edge_index: [2, E] node-index tensor.
        populations: list of raw highest-lambda population counts per block.
        failure_advantage: magnitude of penalty (negative; default -2.0).
        block_offset: integer offset from node index to block ID (default 2).

    Returns:
        Float tensor of shape ``[E, 4]`` with per-dimension pre-normalised
        advantages, ready to be used directly in the policy gradient without
        batch normalisation.
    """
    num_edges = edge_index.size(1)
    rewards = torch.zeros(num_edges, 4, dtype=torch.float32)

    for k in range(num_edges):
        src = int(edge_index[0, k].item())
        dst = int(edge_index[1, k].item())
        pop_i = populations[src] if src < len(populations) else 0
        pop_j = populations[dst] if dst < len(populations) else 0
        minority_frac = min(pop_i, pop_j) / (pop_i + pop_j + 1e-8)

        # Dim 0 (linear): penalty scales with population imbalance.
        # minority_frac ≈ 0.5 → near-zero penalty (linear was fine).
        # minority_frac ≈ 0.0 → full failure_advantage (linear pinned the system).
        rewards[k, 0] = failure_advantage * (1.0 - 2.0 * minority_frac)

        # Dim 1 (quadratic): always full penalty — no transitions = barrier too high.
        rewards[k, 1] = failure_advantage

        # Dims 2, 3 (skew/end): average of linear calibration and quadratic penalty.
        rewards[k, 2] = failure_advantage * (1.0 - minority_frac)
        rewards[k, 3] = failure_advantage * (1.0 - minority_frac)

    return rewards
