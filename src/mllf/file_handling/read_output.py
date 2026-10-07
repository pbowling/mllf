import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union


def terminated_normally(text: str) -> bool:
    """Return True if CHARMM ran dynamics and terminated normally.

    Both a full run and a premature charmm_script('stop') call produce the
    line 'NORMAL TERMINATION BY NORMAL STOP'.  We distinguish them by
    requiring actual dynamics output (DYNA> lines) to be present.
    """
    has_termination = bool(re.search(r"(?<!AB)NORMAL TERMINATION", text))
    has_dynamics = bool(re.search(r"^DYNA>", text, re.MULTILINE))
    return has_termination and has_dynamics


def _find_header_lambdas(lines: List[str], start_idx: int, search_forward: bool = False) -> List[float]:
    """Search for a header line containing lambda values.

    Looks for lines containing 'BLOCK' or 'SITE' and extracts all floats in that line.
    Returns a list of lambda floats in the order they appear.
    
    Args:
        lines: List of text lines
        start_idx: Index to start searching from
        search_forward: If True, search forward; if False, search backward
    """
    header_re = re.compile(r"\b(?:BLOCK|SITE)\b.*>")
    float_re = re.compile(r"\d+\.\d+")
    
    if search_forward:
        # Search forward up to 10 lines
        for i in range(start_idx, min(len(lines), start_idx + 10)):
            line = lines[i]
            if header_re.search(line):
                found = float_re.findall(line)
                if found:
                    return [float(x) for x in found]
    else:
        # Search backward up to 100 lines
        for i in range(start_idx, max(-1, start_idx - 100), -1):
            if i < 0 or i >= len(lines):
                continue
            line = lines[i]
            if header_re.search(line):
                found = float_re.findall(line)
                if found:
                    return [float(x) for x in found]
    return []


def parse_single_population(text: str) -> Dict[int, Dict[float, int]]:
    """Parse SINGLE POPULATION table blocks.

    Returns a mapping block -> {"counts": {lambda_value: count, ...}, "site": site_index or None}
    Dynamically determines lambda columns by inspecting the header line above
    the population table. The site association is discovered by scanning
    nearby 'Site:' sections and the 'SINGLE DDG>' block listings.
    """
    results: Dict[int, Dict[str, object]] = {}
    lines = text.splitlines()
    # Find the index of the 'Total Population Count' block (if present)
    header_idx = None
    for idx, ln in enumerate(lines):
        if 'Total Population Count' in ln:
            header_idx = idx
            break

    # If not found, use the first SINGLE POPULATION occurrence as anchor
    pop_lines_idx = []
    pop_re = re.compile(r"SINGLE POPULATION>\s*(\d+)(.*)$")
    for idx, ln in enumerate(lines):
        if pop_re.search(ln):
            pop_lines_idx.append(idx)

    if not pop_lines_idx:
        return results

    # If we have a "Total Population Count" header, search forward for the BLOCK line
    # Otherwise search backward from the first SINGLE POPULATION line
    if header_idx is not None:
        lambdas = _find_header_lambdas(lines, header_idx, search_forward=True)
    else:
        lambdas = _find_header_lambdas(lines, pop_lines_idx[0], search_forward=False)
    
    # If header detection fails, fall back to default two lambdas [0.95, 0.99]
    if not lambdas:
        lambdas = [0.95, 0.99]

    # Build block->site mapping by scanning SINGLE DDG entries grouped under Site: sections
    block_to_site: Dict[int, int] = {}
    site_re = re.compile(r"^Site:\s*(\d+)")
    ddg_re = re.compile(r"SINGLE DDG>\s*(\d+)\s+(\d+)" )
    current_site = None
    for ln in lines:
        m = site_re.search(ln)
        if m:
            current_site = int(m.group(1))
            continue
        m2 = ddg_re.search(ln)
        if m2 and current_site is not None:
            b1 = int(m2.group(1))
            b2 = int(m2.group(2))
            block_to_site[b1] = current_site
            block_to_site[b2] = current_site

    # Now parse the population lines and map numeric columns to lambdas
    for idx in pop_lines_idx:
        m = pop_re.search(lines[idx])
        if not m:
            continue
        block = int(m.group(1))
        rest = m.group(2).strip()
        # Extract all integer columns following the block index
        nums = [int(x) for x in re.findall(r"\d+", rest)]
        # Map last len(lambdas) numbers to lambdas (allow extra columns)
        if len(nums) >= len(lambdas):
            # take the last len(lambdas) entries as the lambda counts
            vals = nums[-len(lambdas):]
        else:
            # Not enough numeric columns; pad with zeros
            vals = ([0] * (len(lambdas) - len(nums))) + nums
        results[block] = {"counts": {l: v for l, v in zip(lambdas, vals)}, "site": block_to_site.get(block)}
    return results


def parse_transitions_and_rates(text: str) -> Tuple[Dict[int, Dict[float, int]], Dict[int, Dict[float, float]]]:
    """Parse SINGLE TRANSITIONS and SINGLE TRANS RATES tables.

    Returns (transitions, rates) where each is mapping site -> {lambda: value}
    and the lambda columns are determined by scanning the header above the tables.
    """
    transitions: Dict[int, Dict[float, int]] = {}
    rates: Dict[int, Dict[float, float]] = {}
    lines = text.splitlines()

    # collect indices for transitions and rates lines
    trans_idx = []
    rates_idx = []
    trans_re = re.compile(r"SINGLE TRANSITIONS>\s*(\d+)(.*)$")
    rates_re = re.compile(r"SINGLE TRANS RATES>\s*(\d+)(.*)$")
    for idx, ln in enumerate(lines):
        if trans_re.search(ln):
            trans_idx.append(idx)
        if rates_re.search(ln):
            rates_idx.append(idx)

    if not trans_idx and not rates_idx:
        return transitions, rates

    # determine lambdas using the first transitions index as anchor, else use rates
    anchor = trans_idx[0] if trans_idx else (rates_idx[0] if rates_idx else 0)
    lambdas = _find_header_lambdas(lines, anchor)
    if not lambdas:
        lambdas = [0.95, 0.99]

    # parse transitions lines
    for idx in trans_idx:
        m = trans_re.search(lines[idx])
        if not m:
            continue
        site = int(m.group(1))
        rest = m.group(2).strip()
        nums = [int(x) for x in re.findall(r"\d+", rest)]
        if len(nums) >= len(lambdas):
            vals = nums[-len(lambdas):]
        else:
            vals = ([0] * (len(lambdas) - len(nums))) + nums
        transitions[site] = {l: v for l, v in zip(lambdas, vals)}

    # parse rates lines
    for idx in rates_idx:
        m = rates_re.search(lines[idx])
        if not m:
            continue
        site = int(m.group(1))
        rest = m.group(2).strip()
        # floats in the rest
        nums = [float(x) for x in re.findall(r"[0-9]+\.[0-9eE+-]+|[0-9]+\.[0-9]+|[0-9]+", rest)]
        # try to take last len(lambdas) floats
        if len(nums) >= len(lambdas):
            vals = nums[-len(lambdas):]
        else:
            vals = ([0.0] * (len(lambdas) - len(nums))) + nums
        rates[site] = {l: v for l, v in zip(lambdas, vals)}

    return transitions, rates


def parse_fraction_physical_ligand(text: str) -> Optional[float]:
    """Parse the FRACTION PHYSICAL LIGAND diagnostic and return the
    highest-lambda (rightmost / 0.990) value.

    This metric estimates the fraction of the trajectory spent in a fully
    resolved ("physical") ligand state -- exactly one substituent dominant
    at each site -- as opposed to alchemical intermediate/superposition
    states. Single-site combos print one unprefixed line:

        FRACTION PHYSICAL LIGAND>    0.03891    0.01248

    Multi-site combos print an earlier combo-level line:

        MULTI-SITE FRACTION PHYSICAL LIGAND>    0.99445    0.90200

    followed later by a second, near-duplicate unprefixed line computed at
    the hybrid-ligand-pair level. This returns the value from whichever
    line appears FIRST in the output -- for multi-site combos that's the
    'MULTI-SITE' line, which sits immediately after the per-site SINGLE DDG
    blocks that our reward system's ddg_pairs actually key off of; for
    single-site combos it's the only line present.

    Each line has two numeric columns (0.950, 0.990); the last (rightmost)
    column is always the higher/stricter lambda cutoff, matching the
    convention used by parse_single_population/parse_transitions_and_rates/
    parse_single_ddg elsewhere in this module.

    Returns:
        The 0.990-lambda fraction as a float, or None if the line isn't
        present in this output (e.g. older CHARMM builds without this
        diagnostic).
    """
    pattern = re.compile(r"(?:MULTI-SITE\s+)?FRACTION PHYSICAL LIGAND>\s*([^\n]+)")
    m = pattern.search(text)
    if not m:
        return None
    nums = re.findall(r"\d*\.\d+|\d+", m.group(1))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def parse_single_ddg(text: str) -> Dict[Tuple[int, int], Optional[float]]:
    """Parse SINGLE DDG table to discover which substituent pairs had transitions.

    For each unordered pair (blk_i, blk_j) with blk_i < blk_j, returns the biased
    free-energy difference at the highest lambda cutoff (rightmost bias column), or
    None where the output contained 'NaN' (no crossings occurred at that bias level).

    This is more granular than SINGLE TRANSITIONS, which only counts total crossings
    per site: a site with 5 transitions may have 4 between subs 2↔4 and 1 between
    2↔3, while 3↔4 never crossed at all.  The DDG NaN pattern captures that directly.

    Block IDs start at 2 in CHARMM MSLD output (block 1 = reference).
    For a sequential 0-based node index i, block_id = i + 2.

    Args:
        text: Full text of a CHARMM MSLD output file.

    Returns:
        Dict mapping (blk_i, blk_j) → float|None where blk_i < blk_j.
        float : biased DDG at highest lambda (pair had multiple crossings).
        None  : NaN or Inf in output (no usable crossings — either zero, or
                only one transition so no round-trip free energy estimate).
        Empty dict if no SINGLE DDG section is found.
    """
    result: Dict[Tuple[int, int], Optional[float]] = {}
    lines = text.splitlines()

    # Detect number of lambda columns from the DDG header line.
    # Format: "BLK(I)..BLK(J).....> 0.950 ....> 0.990 .....> 0.950 ....> 0.990"
    # Columns alternate: [nobias_λ1, nobias_λ2, bias_λ1, bias_λ2]
    # We want only the bias half — last n_lambda_cols values after the 2 block IDs.
    n_lambda_cols = 2  # default: assume 0.95 and 0.99
    blk_hdr_re = re.compile(r"BLK\(I\).*BLK\(J\)")
    lambda_re = re.compile(r">\s*(\d+\.\d+)")
    for ln in lines:
        if blk_hdr_re.search(ln):
            found = lambda_re.findall(ln)
            if found:
                n_lambda_cols = len(found) // 2  # half nobias, half bias
            break

    ddg_re = re.compile(r"SINGLE DDG>\s+(\d+)\s+(\d+)\s+(.*)")
    for ln in lines:
        m = ddg_re.search(ln)
        if not m:
            continue
        blk_i = int(m.group(1))
        blk_j = int(m.group(2))
        raw = m.group(3).strip().split()

        # Parse values — convert 'NaN' and Inf to None.
        # Inf means only one crossing (no round-trip), so DDG is undefined;
        # treat it the same as NaN (no usable crossing data).
        vals: List[Optional[float]] = []
        for v in raw:
            if v.lower() == 'nan':
                vals.append(None)
            else:
                try:
                    fval = float(v)
                    vals.append(None if math.isinf(fval) else fval)
                except ValueError:
                    vals.append(None)

        # Bias columns are the last n_lambda_cols values
        bias_vals = vals[-n_lambda_cols:] if len(vals) >= n_lambda_cols else vals
        # Use the highest-lambda bias column (last element = 0.990 by default)
        highest_bias = bias_vals[-1] if bias_vals else None

        # Store with blk_i < blk_j (upper triangle, matching output ordering)
        lo, hi = (blk_i, blk_j) if blk_i < blk_j else (blk_j, blk_i)
        result[(lo, hi)] = highest_bias

    return result


OUTPUT_FILE_NAMES = ("output.out", "output")


def resolve_output_file(run_dir: Union[str, Path]) -> Optional[Path]:
    """Locate the CHARMM output for a run without requiring a local copy.

    Checks ``run_dir`` itself first (``output.out``, then ``output``). Collected
    pretraining runs do not always carry their output file; for those, the
    run's ``metadata.json`` names the original simulation in
    ``source_run_dir``, which is checked next. Returns None if neither exists.
    """
    run_dir = Path(run_dir)
    candidates = [run_dir / name for name in OUTPUT_FILE_NAMES]
    metadata = run_dir / "metadata.json"
    if metadata.exists():
        try:
            source = json.loads(metadata.read_text()).get("source_run_dir")
        except (OSError, ValueError):
            source = None
        if source:
            candidates += [Path(source) / name for name in OUTPUT_FILE_NAMES]
    return next((c for c in candidates if c.is_file()), None)


def parse_lambda_trajectory(text: str) -> Tuple[Optional[int], List[List[float]]]:
    """Parse the per-frame lambda trajectory printed by ``traj lamb print``.

    CHARMM prints a header naming the block range, followed by one line per
    saved frame (time, then one lambda per block, in block order)::

        LAMBDA.......TIME(ps).. .BLOCKS   2   THRU BLOCK    5
        LAMBDA>       0.0200  0.0001  0.0002  0.7249  0.2748

    Returns:
        (first_block, frames) -- ``first_block`` is the block ID of the first
        lambda column (2 in practice; None if no header was found, in which
        case callers should assume 2), and ``frames`` is a list of per-frame
        lambda lists with the time column dropped. Empty if no trajectory
        was printed.
    """
    first_block: Optional[int] = None
    hdr = re.search(r"LAMBDA\.+TIME.*?BLOCKS\s+(\d+)\s+THRU\s+BLOCK\s+(\d+)", text)
    if hdr:
        first_block = int(hdr.group(1))
    frames: List[List[float]] = []
    for m in re.finditer(r"^\s*LAMBDA>\s+([^\n]+)", text, re.MULTILINE):
        try:
            vals = [float(v) for v in m.group(1).split()]
        except ValueError:
            continue
        if len(vals) >= 2:
            frames.append(vals[1:])
    return first_block, frames


# Pair lambda-profile estimator settings (see parse_pair_profile).
BARRIER_MID_WINDOW = (0.45, 0.55)   # pair coordinate r treated as "midpoint"
BARRIER_END_WIDTH = 0.10            # r < 0.10 / r >= 0.90 treated as the two endpoints (barrier)
BARRIER_MIN_PAIR_SUM = 0.90         # frame must have lambda_i + lambda_j >= this to be on the pair's edge
BARRIER_PSEUDOCOUNT = 0.5           # keeps estimates finite when a bin is rarely visited
BARRIER_MIN_FRAMES = 100            # below this many on-edge frames the estimates are too noisy to use
# Finer bins for the skew / end asymmetries: endpoint cores (r < 0.01, r >= 0.99),
# near-end shoulders ([0.01, 0.05), [0.95, 0.99)) and the two flanks ([0.25, 0.35), [0.65, 0.75)).
PROFILE_BINS = {"E0": (0.0, 0.01), "K0": (0.01, 0.05), "M0": (0.25, 0.35),
                "M1": (0.65, 0.75), "K1": (0.95, 0.99), "E1": (0.99, 1.0)}
PROFILE_MIN_END_FRAMES = 5          # each endpoint core needs this many frames for skew/end
# Skew signal = flank asymmetry minus this fraction of the end asymmetry, which removes
# the end terms' reach into the flanks (pooled within-pair fit on 26 systems: with this
# value the skew signal moves -0.19/+0.19 per unit x_lo->hi / x_hi->lo and ~0.01 per unit s).
SKEW_END_CORRECTION = 0.687
# Endpoint occupancy (see parse_pair_profile 'endpoints'): r < END_CORE / r >= 1 - END_CORE
# is the endpoint itself (the physical state for a two-substituent site), and
# [END_CORE, NEAR_END_WIDTH) / [1 - NEAR_END_WIDTH, 1 - END_CORE) is just short of it.
END_CORE = 0.01
NEAR_END_WIDTH = 0.20


def _pair_counts(frames, ci: int, cj: int) -> Dict[str, int]:
    """Bin one pair's on-edge frames along r = lambda_j / (lambda_i + lambda_j)."""
    mid_lo, mid_hi = BARRIER_MID_WINDOW
    n = {"edge": 0, "end0": 0, "end1": 0, "mid": 0,
         "at_lo": 0, "near_lo": 0, "near_hi": 0, "at_hi": 0, **{k: 0 for k in PROFILE_BINS}}
    for f in frames:
        s = f[ci] + f[cj]
        if s < BARRIER_MIN_PAIR_SUM:
            continue
        n["edge"] += 1
        r = f[cj] / s
        if r < BARRIER_END_WIDTH:
            n["end0"] += 1
        elif r >= 1.0 - BARRIER_END_WIDTH:
            n["end1"] += 1
        elif mid_lo <= r < mid_hi:
            n["mid"] += 1
        if r < END_CORE:
            n["at_lo"] += 1
        elif r < NEAR_END_WIDTH:
            n["near_lo"] += 1
        elif r >= 1.0 - END_CORE:
            n["at_hi"] += 1
        elif r >= 1.0 - NEAR_END_WIDTH:
            n["near_hi"] += 1
        for k, (lo, hi) in PROFILE_BINS.items():
            if lo <= r < hi or (hi == 1.0 and r >= lo):
                n[k] += 1
                break
    return n


PHYSICAL_LAMBDA = 0.99   # a site is in a physical state when one substituent has λ ≥ this


def _pair_transition_counts(frames, pairs, first_block: int, n_cols: int) -> Dict[Tuple[int, int], int]:
    """Per-pair transition counts at the λ ≥ PHYSICAL_LAMBDA cutoff.

    CHARMM's SINGLE TRANSITIONS definition, resolved by pair: at each site,
    frames where one substituent has λ ≥ 0.99 form a sequence of physical
    states; every change of state is one transition, credited to the pair it
    moves between (lo → other → hi counts lo/other and other/hi, not lo/hi).
    Sites are the connected components of ``pairs`` (same-site pairs from
    SINGLE DDG).

    CHARMM's site total also counts the first entry into a physical state, so
    per site: sum of pair counts + 1 (if the site was ever physical) = CHARMM's
    total. Checked on 35 outputs (2-sub, 3-8-sub, two-site): exact except for a
    rare ±1 when a printed λ is exactly 0.9900 (the trajectory is printed to 4
    decimals; CHARMM compares unrounded values).
    """
    # group blocks into sites via the pair graph
    parent: Dict[int, int] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    sites: Dict[int, List[int]] = {}
    for blk in list(parent):
        sites.setdefault(find(blk), []).append(blk)

    counts: Dict[Tuple[int, int], int] = {}
    for blocks in sites.values():
        cols = [(blk, blk - first_block) for blk in sorted(blocks) if 0 <= blk - first_block < n_cols]
        prev = None
        for f in frames:
            state = next((blk for blk, c in cols if f[c] >= PHYSICAL_LAMBDA), None)
            if state is None or state == prev:
                continue
            if prev is not None:
                key = (min(prev, state), max(prev, state))
                counts[key] = counts.get(key, 0) + 1
            prev = state
    return counts


def parse_pair_profile(
    text: str,
    pairs: Optional[List[Tuple[int, int]]] = None,
    temperature: Optional[float] = None,
) -> Dict[Tuple[int, int], Dict[str, Optional[float]]]:
    """Barrier, skew and end signals (kcal/mol) from each pair's biased lambda profile.

    For a pair of blocks (lo, hi) on the same site, frames on the pair's edge
    (lambda_lo + lambda_hi >= BARRIER_MIN_PAIR_SUM) are projected onto
    r = lambda_hi / (lambda_lo + lambda_hi), and G(bin) = -kT ln(count / width).
    For a two-substituent site every frame is on the edge and r is lambda_hi.
    All three signals describe the BIASED profile the simulation sampled, i.e.
    what is left after the biases have (or have not) cancelled the physical
    free-energy surface; the reward targets say what that residual should be.

    * ``barrier`` = G(r~0.5) - [G(r~0) + G(r~1)]/2 (0.1-wide bins). The quadratic
      term -c*lambda_lo*lambda_hi is zero at both ends and -c/4 at the midpoint,
      so this moves by -c/4 per unit c, while the linear bias cancels. Measured
      within-pair slope vs c: -0.233 (theory -0.25); vs b: -0.04.
    * ``transitions`` = transitions between this pair's two substituents at
      the λ ≥ 0.99 cutoff (CHARMM's SINGLE TRANSITIONS, resolved by pair; see
      ``_pair_transition_counts``). Per site they sum to CHARMM's total minus
      its count of the first entry into a physical state.
    * ``end`` = K_hi - K_lo, where K is each endpoint's sharpness,
      G(near-end shoulder) - G(endpoint core): [0.95, 0.99) vs >= 0.99 at the hi
      end, [0.01, 0.05) vs < 0.01 at the lo end. The end term s_ij peaks within a
      few percent of an endpoint and vanishes at it, so K_hi moves ~-0.44 per
      unit s[lo][hi] and K_lo ~-0.46 per unit s[hi][lo] (x leaks ~-0.09; b, c
      ~0). Zero means both ends hold their population equally sharply.
    * ``skew`` = A - SKEW_END_CORRECTION * end, where A = G(r~0.7) - G(r~0.3)
      minus the linear tilt (0.4/0.99 of G(r>=0.99) - G(r<0.01)). The skew term
      x[lo][hi] is a broad hump peaking near r~0.7 (x[hi][lo] near r~0.3), so A
      tracks x[lo][hi] - x[hi][lo]; end terms also reach the flanks and are
      removed via the end signal. Measured: -0.19 / +0.19 per unit
      x[lo][hi] / x[hi][lo], ~0.01 per unit s, ~0 for b and c.

    * ``endpoints`` = on-edge frame counts {'edge', 'at_lo', 'near_lo', 'near_hi',
      'at_hi'}: at an endpoint (r < 0.01 / r >= 0.99, the physical state for a
      two-substituent site) or just short of it ([0.01, 0.2) / [0.8, 0.99)).
      Many frames near an endpoint that is never reached means the end term at
      that endpoint holds lambda off it (s[hi][lo] at the lo end, s[lo][hi] at
      the hi end); available even when the barrier/skew/end are not.

    Signs of ``skew`` and ``end`` follow the (lo, hi) block orientation; rewards
    use their magnitude.

    Args:
        text: Full text of a CHARMM MSLD output file.
        pairs: Block-ID pairs (lo, hi) to evaluate. Defaults to the pairs
            listed in the SINGLE DDG table (i.e. every same-site pair).
        temperature: Kelvin; defaults to the last nonzero 'Temperature is:'
            line, else 298.15.

    Returns:
        Dict (lo, hi) -> {'barrier', 'skew', 'end', 'transitions', 'endpoints'};
        the first three are floats or None, ``transitions`` an int,
        ``endpoints`` a dict of ints (None if the pair's blocks are not in the
        trajectory).
        ``barrier`` is None when the pair has fewer than BARRIER_MIN_FRAMES
        frames on its edge or an endpoint (r < 0.1 / r >= 0.9) was never
        visited; ``skew``/``end`` are None when either endpoint core
        (r < 0.01 / r >= 0.99) has fewer than PROFILE_MIN_END_FRAMES frames.
        Empty dict if no lambda trajectory was printed.
    """
    first_block, frames = parse_lambda_trajectory(text)
    if not frames:
        return {}
    if first_block is None:
        first_block = 2
    if pairs is None:
        pairs = sorted(parse_single_ddg(text).keys())
    if temperature is None:
        # Earlier setup sections can print 'Temperature is: 0.000'; the lambda
        # analysis prints the real value last.
        temps = [float(t) for t in re.findall(r"Temperature is:\s*([\d.]+)", text) if float(t) > 0]
        temperature = temps[-1] if temps else 298.15
    kT = 0.0019872041 * temperature
    pc = BARRIER_PSEUDOCOUNT

    n_cols = min(len(f) for f in frames)
    norm_pairs = [(min(a, b), max(a, b)) for a, b in pairs]
    transitions = _pair_transition_counts(frames, norm_pairs, first_block, n_cols)
    result: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}
    for blk_a, blk_b in pairs:
        lo, hi = (blk_a, blk_b) if blk_a < blk_b else (blk_b, blk_a)
        ci, cj = lo - first_block, hi - first_block
        out: Dict[str, Optional[float]] = {"barrier": None, "skew": None, "end": None,
                                           "transitions": transitions.get((lo, hi), 0),
                                           "endpoints": None}
        result[(lo, hi)] = out
        if ci < 0 or cj >= n_cols:
            continue
        n = _pair_counts(frames, ci, cj)
        out["endpoints"] = {k: n[k] for k in ("edge", "at_lo", "near_lo", "near_hi", "at_hi")}
        if n["edge"] < BARRIER_MIN_FRAMES:
            continue
        if n["end0"] > 0 and n["end1"] > 0:
            out["barrier"] = -kT * (math.log(n["mid"] + pc)
                                    - 0.5 * (math.log(n["end0"] + pc) + math.log(n["end1"] + pc)))
        if n["E0"] >= PROFILE_MIN_END_FRAMES and n["E1"] >= PROFILE_MIN_END_FRAMES:
            G = {k: -kT * math.log((n[k] + pc) / (b - a)) for k, (a, b) in PROFILE_BINS.items()}
            tilt = G["E1"] - G["E0"]
            end = (G["K1"] - G["E1"]) - (G["K0"] - G["E0"])
            flank = (G["M1"] - G["M0"]) - (0.4 / 0.99) * tilt
            out["end"] = end
            out["skew"] = flank - SKEW_END_CORRECTION * end
    return result


def parse_pair_barriers(
    text: str,
    pairs: Optional[List[Tuple[int, int]]] = None,
    temperature: Optional[float] = None,
) -> Dict[Tuple[int, int], Optional[float]]:
    """Biased midpoint free-energy barrier (kcal/mol) for each substituent pair.

    Convenience wrapper returning only the ``barrier`` entry of
    ``parse_pair_profile`` (see there for the definition and validation).

    Returns:
        Dict (lo, hi) -> barrier in kcal/mol, or None when the pair has fewer
        than BARRIER_MIN_FRAMES frames on its edge or one endpoint was never
        visited. Empty dict if no lambda trajectory was printed.
    """
    return {k: v["barrier"] for k, v in parse_pair_profile(text, pairs, temperature).items()}


__all__ = [
    "terminated_normally",
    "resolve_output_file",
    "parse_single_population",
    "parse_transitions_and_rates",
    "parse_single_ddg",
    "parse_fraction_physical_ligand",
    "parse_lambda_trajectory",
    "parse_pair_profile",
    "parse_pair_barriers",
]
