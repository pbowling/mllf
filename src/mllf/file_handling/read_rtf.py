"""RTF file parser

Provides helpers to parse RTF files (example in examples/training_files)
and extract ATOM lines. Each ATOM line is expected to have the format:

	ATOM <name> <type> <charge>

We collect the atom types (third column) and sum the charges (last column).

Functions:
- parse_rtf_file(path) -> dict with keys: site, sub, atom_types (list), total_charge (float)
- parse_rtf_dir(directory) -> dict mapping "site{n}_sub{m}" -> parsed dict

"""

from __future__ import annotations

import os
import re
from typing import Dict, List


ATOM_RE = re.compile(r'^\s*ATOM\s+\S+\s+(\S+)\s+(-?\d+\.?\d*)')
FNAME_RE = re.compile(r'site(\d+)_sub(\d+)', re.IGNORECASE)


def parse_rtf_file(path: str) -> Dict[str, object]:
	"""Parse a single RTF/PRES fragment file and return extracted info.

	Returns a dict: {"site": int, "sub": int, "atom_types": [str], "total_charge": float}
	If the filename doesn't contain site/sub, site/sub will be None.
	"""
	atom_types: List[str] = []
	total_charge = 0.0

	# try to extract site/sub from filename
	basename = os.path.basename(path)
	m = FNAME_RE.search(basename)
	site = int(m.group(1)) if m else None
	sub = int(m.group(2)) if m else None

	with open(path, 'r') as fh:
		for line in fh:
			mo = ATOM_RE.match(line)
			if not mo:
				continue
			atom_type, charge_str = mo.groups()
			atom_types.append(atom_type)
			try:
				total_charge += float(charge_str)
			except ValueError:
				# ignore unparsable charges
				continue

	return {
		"site": site,
		"sub": sub,
		"atom_types": atom_types,
		"total_charge": total_charge,
		"filename": basename,
	}


def parse_rtf_dir(directory: str) -> Dict[str, Dict[str, object]]:
	"""Parse all .rtf/.rft files in a directory and return a mapping keyed by site_sub.

	The key used is `site{site}_sub{sub}` when site/sub are found in the filename.
	Files without matching names are keyed by the filename.
	"""
	results: Dict[str, Dict[str, object]] = {}
	for entry in sorted(os.listdir(directory)):
		if not entry.lower().endswith('.rtf') and not entry.lower().endswith('.rft'):
			continue
		path = os.path.join(directory, entry)
		parsed = parse_rtf_file(path)
		if parsed['site'] is not None and parsed['sub'] is not None:
			key = f"site{parsed['site']}_sub{parsed['sub']}"
		else:
			key = os.path.splitext(entry)[0]
		results[key] = parsed
	return results


if __name__ == '__main__':
	# quick smoke test when run directly
	import json
	here = os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'training_files', '14benz_vac')
	here = os.path.abspath(here)
	out = parse_rtf_dir(here)
	print(json.dumps(out, indent=2))

