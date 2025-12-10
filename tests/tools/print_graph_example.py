"""Utility to print graph details for the 14benz example.

Run this file directly for quick inspection:

    python tests/tools/print_graph_example.py

"""
import os
from pprint import pprint

from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.cb.graph import Graph


def main():
    from pathlib import Path

    def _find_repo_root(start: Path) -> Path:
        p = start.resolve()
        for parent in [p] + list(p.parents):
            if (parent / 'pyproject.toml').exists():
                return parent
        raise RuntimeError('Could not find repository root (pyproject.toml)')

    repo = _find_repo_root(Path(__file__))
    examples_dir = repo / 'examples' / 'cb' / '14benz_solv_5.5'
    if not examples_dir.is_dir():
        print('Example directory not found:', examples_dir)
        return

    rtf_results = parse_rtf_dir(str(examples_dir))
    g = Graph.from_rtf_results(rtf_results)

    print(f'Graph: nodes={g.num_nodes}, edges={len(g.edges)}')
    print('\nNode metadata:')
    for idx in range(g.num_nodes):
        info = g.get_node_info(idx)
        # print key metadata per substituent
        print(f"Node {idx}: site={info.get('site')}, sub={info.get('sub')}, total_charge={info.get('total_charge')}, distinct_atoms={info.get('distinct_atom_types')}, solvent={info.get('solvent')}")

    print('\nSample edges (first 10):')
    for i, ((a, b), coeffs) in enumerate(sorted(g.edges.items())):
        if i >= 10:
            break
        print(f'Edge ({a},{b}):', coeffs)


if __name__ == '__main__':
    main()
