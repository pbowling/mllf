import re
from typing import Dict, Tuple, List


def terminated_normally(text: str) -> bool:
    """Return True if the output text indicates a normal termination."""
    return bool(re.search(r"NORMAL TERMINATION", text))


def _find_header_lambdas(lines: List[str], start_idx: int) -> List[float]:
    """Search backwards from start_idx for a header line containing lambda values.

    Looks for lines containing 'BLOCK' or 'SITE' and extracts all floats in that line.
    Returns a list of lambda floats in the order they appear.
    """
    header_re = re.compile(r"\b(?:BLOCK|SITE)\b.*>")
    float_re = re.compile(r"\d+\.\d+")
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

    anchor_idx = header_idx if header_idx is not None else pop_lines_idx[0]
    lambdas = _find_header_lambdas(lines, anchor_idx)
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


__all__ = [
    "terminated_normally",
    "parse_single_population",
    "parse_transitions_and_rates",
]
import re
from typing import Dict, Tuple, List


def terminated_normally(text: str) -> bool:
    """Return True if the output text indicates a normal termination."""
    return bool(re.search(r"NORMAL TERMINATION", text))


def _find_header_lambdas(lines: List[str], start_idx: int) -> List[float]:
    """Search backwards from start_idx for a header line containing lambda values.

    Looks for lines containing 'BLOCK' or 'SITE' and extracts all floats in that line.
    Returns a list of lambda floats in the order they appear.
    """
    header_re = re.compile(r"\b(?:BLOCK|SITE)\b.*>")
    float_re = re.compile(r"\d+\.\d+")
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

    anchor_idx = header_idx if header_idx is not None else pop_lines_idx[0]
    lambdas = _find_header_lambdas(lines, anchor_idx)
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


__all__ = [
    "terminated_normally",
    "parse_single_population",
    "parse_transitions_and_rates",
]
