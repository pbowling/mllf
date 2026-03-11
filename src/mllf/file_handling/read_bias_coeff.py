import re
from pathlib import Path
from typing import List, Optional
import yaml


def _extract_nsubs(content: str, prep_dir: Optional[Path] = None) -> Optional[List[int]]:
    """Return nsubs_per_site as a list of ints from the first source that works.

    Tries in order:
      1. ``alf_info_string`` YAML block embedded in *content*.
      2. ``prep_dir/alf_info.py`` — ``alf_info['nsubs'] = [...]`` assignment.
      3. Module-level ``nsubs = [...]`` assignment in *content*.
    """
    # 1. alf_info_string YAML block
    m = re.search(r'alf_info_string\s*=\s*"""(.*?)"""', content, re.DOTALL)
    if not m:
        m = re.search(r"alf_info_string\s*=\s*'''(.*?)'''", content, re.DOTALL)
    if m:
        try:
            alf = yaml.safe_load(m.group(1))
            if isinstance(alf, dict) and 'nsubs' in alf:
                nsubs = list(alf['nsubs'])
                if all(isinstance(n, int) for n in nsubs):
                    return nsubs
        except Exception:
            pass

    # 2. prep_dir/alf_info.py
    if prep_dir is not None:
        alf_py = Path(prep_dir) / 'alf_info.py'
        if alf_py.exists():
            alf_content = alf_py.read_text()
            m2 = re.search(r"alf_info\['nsubs'\]\s*=\s*(\[[^\]]+\])", alf_content)
            if m2:
                try:
                    return [int(x) for x in re.findall(r'\d+', m2.group(1))]
                except Exception:
                    pass

    # 3. Module-level nsubs assignment
    m3 = re.search(r'^nsubs\s*=\s*(\[[^\]]+\])', content, re.MULTILINE)
    if m3:
        try:
            return [int(x) for x in re.findall(r'\d+', m3.group(1))]
        except Exception:
            pass

    return None


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


def parse_new(filename, prep_dir=None):
    """Parse the newer variables#.py style file.

    The file contains a bias_string with YAML-formatted data including:
    - b: nested list (single row vector, or 2-D ``[[b1, b2, …]]`` — normalised to 1-D)
    - c, x, s: NxN matrices
    - scalar entries: lams*, cs*, xs*, ss* as YAML scalars

    Also extracts ``_nsubs_per_site`` from the ``alf_info_string`` YAML block
    embedded in the same file, or from ``prep_dir/alf_info.py`` when provided.

    Args:
        filename: Path to variables*.py file.
        prep_dir: Optional directory containing ``alf_info.py`` (fallback for
                  systems where ``alf_info_string`` is absent).

    Returns:
        Dict with ``'lams'``, ``'cs'``, ``'xs'``, ``'ss'`` subdicts,
        ``'b'``, ``'c'``, ``'x'``, ``'s'`` matrix keys (where present),
        and ``'_nsubs_per_site'`` when it can be determined.
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

        except yaml.YAMLError:
            # Fall through to line-by-line parsing
            pass
        else:
            # Normalise b: "variablesflat" format stores b as [[b1, b2, …]] (2-D).
            b = data.get('b', [])
            if b and isinstance(b[0], list):
                data['b'] = b[0]

            # Inject _nsubs_per_site when determinable
            nsubs = _extract_nsubs(content, prep_dir)
            if nsubs is not None:
                data['_nsubs_per_site'] = nsubs

            return data

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

    nsubs = _extract_nsubs(content, prep_dir)
    if nsubs is not None:
        data['_nsubs_per_site'] = nsubs

    return data


def read_bias_coeff(filename, prep_dir=None):
    """Reads a bias coefficient file and returns a dictionary of parameter types and values."""
    if ".inp" in filename:
        return parse_old(filename)
    else:
        return parse_new(filename, prep_dir=prep_dir)