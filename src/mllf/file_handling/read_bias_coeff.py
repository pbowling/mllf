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

    - Reads scalar lines of the form `name: value` and maps keys
      starting with cs/xs/ss/lams into the corresponding dicts.
    """
    data = {"lams": {}, "cs": {}, "xs": {}, "ss": {}}

    with open(filename, 'r') as fh:
        lines = fh.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # match scalar entries like 'cs1s1s1s2: 2.3' or 'lams1s2: -11.4'
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(-?\d+\.\d+|-?\d+)\s*$", line)
        if m:
            param, value = m.groups()
            value = float(value)
            if param.startswith("lams"):
                data["lams"][param] = value
            elif param.startswith("cs"):
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