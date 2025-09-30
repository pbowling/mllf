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
    pass


def read_bias_coeff(filename):
    """Reads a bias coefficient file and returns a dictionary of parameter types and values."""
    if ".inp" in filename:
        return parse_old(filename)
    else:
        return parse_new(filename)