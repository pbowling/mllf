import math
import re
from typing import Dict, Optional, Tuple, List


def terminated_normally(text: str) -> bool:
    """Return True if the output text indicates a normal termination."""
    return bool(re.search(r"NORMAL TERMINATION", text))


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


__all__ = [
    "terminated_normally",
    "parse_single_population",
    "parse_transitions_and_rates",
    "parse_single_ddg",
]
