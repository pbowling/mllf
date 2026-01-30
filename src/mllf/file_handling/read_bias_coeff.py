import re
import yaml

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

    The file contains a bias_string with YAML-formatted data including:
    - b: nested list (single row vector)
    - c, x, s: NxN matrices
    - scalar entries: lams*, cs*, xs*, ss* as YAML scalars
    
    Returns a dict with 'lams', 'cs', 'xs', 'ss' subdicts and optionally 'b', 'c', 'x', 's' matrices.
    """
    data = {"lams": {}, "cs": {}, "xs": {}, "ss": {}}

    with open(filename, 'r') as fh:
        content = fh.read()

    # Method 1: Try to extract and parse bias_string directly
    # Look for bias_string = """ ... """ or bias_string = ''' ... '''
    bias_string_match = re.search(r'bias_string\s*=\s*"""(.*?)"""', content, re.DOTALL)
    if not bias_string_match:
        bias_string_match = re.search(r"bias_string\s*=\s*'''(.*?)'''", content, re.DOTALL)
    
    if bias_string_match:
        bias_string = bias_string_match.group(1)
        try:
            bias_data = yaml.safe_load(bias_string)
            
            # Extract matrices if present
            if isinstance(bias_data, dict):
                # Store matrices separately
                for key in ['b', 'c', 'x', 's']:
                    if key in bias_data:
                        data[key] = bias_data[key]
                
                # Extract scalar entries
                for param, value in bias_data.items():
                    if isinstance(param, str):
                        if param.startswith("lams"):
                            data["lams"][param] = float(value)
                        elif param.startswith("cs"):
                            data["cs"][param] = float(value)
                        elif param.startswith("xs"):
                            data["xs"][param] = float(value)
                        elif param.startswith("ss"):
                            data["ss"][param] = float(value)
            
            return data
        except yaml.YAMLError as e:
            # Fall through to line-by-line parsing
            pass
    
    # Method 2: Fallback to line-by-line parsing (for files without proper bias_string)
    lines = content.split('\n')
    for line in lines:
        # match scalar entries like 'cs1s1s1s2: 2.3' or 'lams1s2: -11.4'
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*:\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*$", line)
        if m:
            param, value = m.groups()
            try:
                value = float(value)
                if param.startswith("lams"):
                    data["lams"][param] = value
                elif param.startswith("cs"):
                    data["cs"][param] = value
                elif param.startswith("xs"):
                    data["xs"][param] = value
                elif param.startswith("ss"):
                    data["ss"][param] = value
            except ValueError:
                continue

    return data


def read_bias_coeff(filename):
    """Reads a bias coefficient file and returns a dictionary of parameter types and values."""
    if ".inp" in filename:
        return parse_old(filename)
    else:
        return parse_new(filename)