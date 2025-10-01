import re

def parse_old(filename):
    """Parses a variables#.inp file and returns a dictionary of parameter types and values."""
    data = {"lams": {}, "cs": {}, "xs": {}, "ss": {}}
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r'set\s+(\w+)\s*=\s*(-?\d+\.\d+)', line)
            if match:
                param, value = match.groups()
                value = float(value)
                if param.startswith("lams"):
                    data["lams"][param] = value
                elif param.startswith("cs"):
                    data["cs"][param] = value
                elif param.startswith("xs"):
                    data["xs"][param] = value
                elif param.startswith("ss"):
                    data["ss"][param] = value
    return data


def parse_new(filename):
    """Parse the newer variables#.py style file.

    - Reads a top-level `b:` nested list and maps it to lams<i>s<j> keys
      (e.g. lams1s1, lams1s2, ...).
    - Reads scalar lines of the form `name: value` and maps keys
      starting with cs/xs/ss/lams into the corresponding dicts.
    """
    data = {"lams": {}, "cs": {}, "xs": {}, "ss": {}}

    with open(filename, 'r') as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)
    # simple state machine to capture the nested list after 'b:'
    while i < n:
        line = lines[i]

        # detect start of b: nested list
        if re.match(r'^\s*b\s*:\s*$', line):
            # collect all numbers from the nested lists following b:
            nums_all = []
            i += 1
            while i < n and not re.match(r'^\w+\s*:', lines[i]):
                l = lines[i]
                nums = re.findall(r'-?\d+\.\d+|-?\d+', l)
                if nums:
                    nums_all.extend([float(x) for x in nums])
                i += 1

            # Assign flattened numbers into lams keys. The file encodes blocks
            # for multiple i values concatenated; each block begins with a 0.0
            # value which should be assigned as the first j for that i.
            i_idx = 1
            j_idx = 1
            for val in nums_all:
                # if we hit a zero and we're not at the start of a block,
                # that zero actually starts the next i block
                if val == 0.0 and j_idx != 1:
                    i_idx += 1
                    j_idx = 1

                key = f"lams{i_idx}s{j_idx}"
                data['lams'][key] = float(val)

                # advance j unless the next zero will signal a new block
                j_idx += 1

            continue

        # match scalar entries like 'cs1s1s1s2: 2.3'
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(-?\d+\.\d+|-?\d+)\s*$", line)
        if m:
            param, value = m.groups()
            value = float(value)
            if param.startswith("cs"):
                data["cs"][param] = value
            elif param.startswith("xs"):
                data["xs"][param] = value
            elif param.startswith("ss"):
                data["ss"][param] = value

        i += 1

    return data


def read_bias_coeff(filename):
    """Reads a bias coefficient file and returns a dictionary of parameter types and values."""
    if ".inp" in filename:
        return parse_old(filename)
    else:
        return parse_new(filename)