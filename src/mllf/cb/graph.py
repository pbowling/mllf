"""Graph structure for contextual bandit.

Placed here so CB code can evolve independently while keeping the same
interface (Graph, EdgeCoeffs) used by the tests and helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import numpy as np


@dataclass
class EdgeCoeffs:
    linear: float = 0.0
    quadratic: float = 0.0
    skew: float = 0.0
    end: float = 0.0


class Graph:
    """Simple undirected graph with coefficients on edges.

    Edges are keyed by (i,j) with i < j.
    """

    def __init__(self, num_nodes: int):
        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        self.num_nodes = num_nodes
        self.edges: Dict[Tuple[int, int], EdgeCoeffs] = {}
        # per-node metadata (e.g. substituent info parsed from RTF files)
        # keyed by node index -> dict
        self.nodes: Dict[int, Dict] = {}
        # initialize fully-connected graph with zero coefficients
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                self.edges[(i, j)] = EdgeCoeffs()

        # per-edge masks indicating whether a particular coeff type is active
        # e.g. self.edge_mask[(i,j)] = {'linear': True, 'quadratic': True, 'skew': True, 'end': True}
        self.edge_mask: Dict[Tuple[int, int], Dict[str, bool]] = {}
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                self.edge_mask[(i, j)] = {'linear': True, 'quadratic': True, 'skew': True, 'end': True}

        # initialize empty node metadata
        for i in range(num_nodes):
            self.nodes[i] = {}

    def set_edge(self, i: int, j: int, coeffs: EdgeCoeffs | List[float] | Tuple[float, float, float, float]):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in self.edges:
            raise KeyError(f"Edge ({a},{b}) not present for {self.num_nodes} nodes")
        if isinstance(coeffs, EdgeCoeffs):
            self.edges[(a, b)] = coeffs
        else:
            self.edges[(a, b)] = EdgeCoeffs(*coeffs)

    def get_edge(self, i: int, j: int) -> EdgeCoeffs:
        a, b = (i, j) if i < j else (j, i)
        return self.edges[(a, b)]

    def as_vector(self) -> np.ndarray:
        """Return a flat vector of all edge coefficients in a consistent ordering.

        Ordering: for i in 0..n-2, for j in i+1..n-1, append [linear, quadratic, skew, end]
        """
        out = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                e = self.edges[(i, j)]
                out.extend([e.linear, e.quadratic, e.skew, e.end])
        return np.array(out, dtype=float)

    def from_vector(self, vec: List[float] | np.ndarray):
        """Populate edges from a flat vector following the same ordering as as_vector()."""
        arr = np.asarray(vec, dtype=float)
        expected = (self.num_nodes * (self.num_nodes - 1) // 2) * 4
        if arr.size != expected:
            raise ValueError(f"Vector length {arr.size} does not match expected {expected} for {self.num_nodes} nodes")
        idx = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[(i, j)] = EdgeCoeffs(float(arr[idx]), float(arr[idx+1]), float(arr[idx+2]), float(arr[idx+3]))
                idx += 4

    # Node metadata helpers
    def set_node_info(self, node: int, info: Dict):
        """Attach arbitrary metadata dict to a node index."""
        if node < 0 or node >= self.num_nodes:
            raise IndexError("node out of range")
        self.nodes[node] = dict(info)

    def get_node_info(self, node: int) -> Dict:
        """Return stored node metadata dict (may be empty)."""
        if node < 0 or node >= self.num_nodes:
            raise IndexError("node out of range")
        return self.nodes.get(node, {})

    def setup_from_rtf_results(self, rtf_results: Dict[str, Dict]):
        """Populate node metadata from parse_rtf_dir output.

        The rtf_results keys are expected to be either 'site{n}_sub{m}' or a
        filename. For keys with site information, we will collect per-site lists
        of substituents and attach them to the node metadata under 'subs' and
        also store the raw parsed entries under 'rtf'. For each substituent we
        populate metadata: total_charge, atom_types, unique_atom_types (atoms
        that differ from other subs at the same site) and solvent state (a
        simple heuristic based on filename).
        """
        # collect subs per site
        sites = {}
        for key, parsed in rtf_results.items():
            site = parsed.get('site')
            sub = parsed.get('sub')
            if site is None or sub is None:
                # skip entries without explicit site/sub info
                continue
            sites.setdefault(site, {})
            sites[site].setdefault('rtf', {})
            sites[site]['rtf'][sub] = parsed

        def detect_solvent_state(filename: str) -> str:
            fn = (filename or "").lower()
            if 'vac' in fn or 'vacuum' in fn:
                return 'vacuum'
            if 'solv' in fn or 'water' in fn or 'aq' in fn or 'sol' in fn:
                return 'solvated'
            return 'unknown'

        # attach to nodes
        for site_idx, data in sites.items():
            if site_idx - 1 < 0 or site_idx - 1 >= self.num_nodes:
                # ignore out-of-range site indices
                continue
            # compute atom type sets per sub for uniqueness computation
            per_sub = data['rtf']
            atom_sets = {s: set((per_sub[s].get('atom_types') or [])) for s in per_sub}
            # compute unique atom types per sub (difference from other subs at site)
            unique = {}
            for s, aset in atom_sets.items():
                others = set().union(*(atom_sets[o] for o in atom_sets if o != s)) if len(atom_sets) > 1 else set()
                unique[s] = sorted(list(aset - others))

            subs_meta = {}
            for s, parsed in per_sub.items():
                fname = parsed.get('filename')
                subs_meta[s] = {
                    'site': site_idx,
                    'sub': s,
                    'total_charge': float(parsed.get('total_charge', 0.0) or 0.0),
                    'atom_types': list(parsed.get('atom_types') or []),
                    'unique_atom_types': unique.get(s, []),
                    'solvent': detect_solvent_state(fname),
                    'rtf': parsed,
                }

            info = {'site': site_idx, 'subs': sorted(list(per_sub.keys())), 'rtf': per_sub, 'subs_meta': subs_meta}
            self.set_node_info(site_idx - 1, info)

        # after populating node metadata, apply default connectivity rules
        # so that inter-site edges are disabled by default and intra-site
        # connectivity can be controlled by specific rules (see apply_site_connectivity_rules)
        try:
            self.apply_site_connectivity_rules()
        except Exception:
            # be defensive: do not fail setup if masking cannot be applied
            pass

    @classmethod
    def from_rtf_results(cls, rtf_results: Dict[str, Dict], solvent_override: Optional[str] = None) -> "Graph":
        """Create a Graph with one node per substituent (sub), populating node metadata.

        rtf_results is expected to contain parsed entries with keys including
        'site' and 'sub'. The returned Graph will have nodes enumerated in
        deterministic order (sorted by site then sub) and node metadata containing
        'site', 'sub', 'total_charge', 'atom_types', 'unique_atom_types', 'solvent', and 'rtf'.
        
        Args:
            rtf_results: Dictionary mapping keys to parsed RTF data
            solvent_override: Optional environment type override. Allowed values:
                - 'gas' or 'vacuum': Gas phase / vacuum environment
                - 'solv' or 'solvent': Solvent / water environment
                - 'protein': Protein environment
                If not provided, environment is auto-detected from filenames.
        """
        # collect entries that have site and sub
        subs = []
        for key, parsed in rtf_results.items():
            site = parsed.get('site')
            sub = parsed.get('sub')
            if site is None or sub is None:
                continue
            subs.append((int(site), int(sub), key, parsed))

        if not subs:
            # fallback: single node graph
            g = cls(1)
            return g

        # sort by site then sub for deterministic ordering
        subs.sort(key=lambda x: (x[0], x[1]))

        # compute atom type sets per site/sub to determine unique atom types
        per_site = {}
        for site, sub, key, parsed in subs:
            per_site.setdefault(site, {})
            per_site[site][sub] = parsed

        # compute unique atom types per sub within each site (difference from other subs at same site)
        # and compute the intersection across all subs at a site (atoms common to every sub)
        unique_map = {}
        site_intersection = {}
        for site, subdict in per_site.items():
            atom_sets = {s: set((subdict[s].get('atom_types') or [])) for s in subdict}
            if atom_sets:
                intersection_all = set.intersection(*atom_sets.values()) if len(atom_sets) > 0 else set()
            else:
                intersection_all = set()
            site_intersection[site] = intersection_all
            for s, aset in atom_sets.items():
                others = set().union(*(atom_sets[o] for o in atom_sets if o != s)) if len(atom_sets) > 1 else set()
                unique_map[(site, s)] = sorted(list(aset - others))

        # Also compute globally unique atom types (appear only in a single sub across all sites)
        all_atom_lists = [set(parsed.get('atom_types') or []) for (_, _, _, parsed) in subs]
        global_counts = {}
        for aset in all_atom_lists:
            for a in aset:
                global_counts[a] = global_counts.get(a, 0) + 1
        globally_unique = {a for a, c in global_counts.items() if c == 1}

        # helper to detect solvent and normalize to one of: 'solv', 'gas', 'protein'
        def detect_solvent_state(filename: str) -> str:
            fn = (filename or "").lower()
            if 'prot' in fn or 'protein' in fn:
                return 'protein'
            if 'vac' in fn or 'vacuum' in fn or 'gas' in fn:
                return 'gas'
            if 'solv' in fn or 'water' in fn or 'aq' in fn or 'sol' in fn:
                return 'solv'
            # default to 'solv' when ambiguous
            return 'solv'

        # create graph with one node per substituent
        nsubs = len(subs)
        g = cls(nsubs)

        # populate node metadata per substituent
        for idx, (site, sub, key, parsed) in enumerate(subs):
            # prefer full file path for solvent detection (may include directory hints)
            fname = parsed.get('filepath') or parsed.get('filename')
            atom_types = list(parsed.get('atom_types') or [])
            total_charge = float(parsed.get('total_charge', 0.0) or 0.0)
            # distinct_atom_types: preserve duplicates and order but exclude atoms
            # that are present in every sub at this site (site_intersection)
            intersection_all = site_intersection.get(site, set())
            distinct_list = [a for a in atom_types if a not in intersection_all]

            # per-site unique atom types (present only in this sub at the site)
            per_site_unique = set(unique_map.get((site, sub), []))
            # globally unique atom types that appear in this sub
            global_unique_in_sub = sorted(list(set(atom_types).intersection(globally_unique)))
            # merged unique types (no duplicates)
            merged_unique = sorted(list(per_site_unique.union(global_unique_in_sub)))

            # Validate solvent_override if provided; warn and set to 'unknown' if invalid
            allowed_solvents = {'solv', 'gas', 'protein'}
            if solvent_override is not None:
                if solvent_override in allowed_solvents:
                    sol_state = solvent_override
                else:
                    import warnings

                    warnings.warn(
                        f"Invalid solvent override '{solvent_override}' provided; temporarily setting solvent to 'unknown'",
                        UserWarning,
                    )
                    sol_state = 'unknown'
            else:
                sol_state = detect_solvent_state(fname)

            subs_meta = {
                'site': site,
                'sub': sub,
                'total_charge': total_charge,
                'atom_types': atom_types,
                # distinct (preserve duplicates) and unique (set) representations
                'distinct_atom_types': distinct_list,
                'unique_atom_types': merged_unique,
                'solvent': sol_state,
                'rtf': parsed,
            }
            # store the metadata directly at this node index
            g.set_node_info(idx, subs_meta)

        # apply connectivity rules now that node metadata is populated
        g.apply_site_connectivity_rules()
        return g

    def apply_site_connectivity_rules(self):
        """Apply default connectivity rules per-site.

        Rules implemented:
        - Inter-site edges: all coefficient masks set to False (no coupling between sites)
        - Intra-site edges:
            * 'linear' coefficients allowed only for edges that include sub==1 (the primary sub)
              i.e., sub1 connected to each other sub; edges between non-sub1 subs have linear disabled.
            * 'quadratic', 'skew', and 'end' coefficients are allowed for all intra-site pairs

        This method relies on node metadata stored via `set_node_info` / `from_rtf_results`.
        If node metadata is missing for a node, conservative defaults (all False for inter-site) are used.
        """
        # build mapping node -> site, sub
        node_site = {}
        node_sub = {}
        for n, info in self.nodes.items():
            try:
                node_site[n] = int(info.get('site'))
            except Exception:
                node_site[n] = None
            try:
                node_sub[n] = int(info.get('sub'))
            except Exception:
                node_sub[n] = None

        for (i, j) in list(self.edges.keys()):
            si = node_site.get(i)
            sj = node_site.get(j)
            # default: disable everything
            mask = {'linear': False, 'quadratic': False, 'skew': False, 'end': False}
            if si is not None and sj is not None and si == sj:
                # intra-site pair
                # linear allowed only if one of the subs is sub==1
                subi = node_sub.get(i)
                subj = node_sub.get(j)
                is_linear = (subi == 1) or (subj == 1)
                mask['linear'] = bool(is_linear)
                # quadratic/skew/end: fully connected among subs within same site
                mask['quadratic'] = True
                mask['skew'] = True
                mask['end'] = True
            # store
            self.edge_mask[(i, j)] = mask

    def get_allowed_edges_for_bias(self, bias_name: str) -> List[Tuple[int, int]]:
        """Return list of edge (i,j) pairs where given bias_name mask is True.

        bias_name should be one of: 'linear', 'quadratic', 'skew', 'end'.
        """
        if bias_name not in ('linear', 'quadratic', 'skew', 'end'):
            raise ValueError(f'Unknown bias name: {bias_name}')
        out = []
        for (i, j), mask in self.edge_mask.items():
            if mask.get(bias_name):
                out.append((i, j))
        return out

    def padded_copy(self, new_num_nodes: int) -> "Graph":
        """Return a copy of this graph in a larger Graph of size new_num_nodes.

        Edges and node metadata from the original graph are copied into the
        returned Graph; new nodes/edges are zero-initialized.
        """
        if new_num_nodes < self.num_nodes:
            raise ValueError("new_num_nodes must be >= current num_nodes")
        newg = Graph(new_num_nodes)
        # copy edges
        for (i, j), coeffs in self.edges.items():
            newg.edges[(i, j)] = EdgeCoeffs(coeffs.linear, coeffs.quadratic, coeffs.skew, coeffs.end)
        # copy masks as well
        for (i, j), mask in self.edge_mask.items():
            newg.edge_mask[(i, j)] = dict(mask)
        # copy node metadata
        for idx, info in self.nodes.items():
            newg.nodes[idx] = dict(info)
        return newg
