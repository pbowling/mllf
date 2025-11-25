"""Example: Building node features with CHARMM atom type vocabulary.

This example demonstrates how atom types from toppar files are used to create
consistent node feature vectors across all graphs.
"""

from mllf.cb.atom_vocab import get_atom_type_vocab
from mllf.cb.graph import Graph
from mllf.cb.graph_utils import build_pyg_graph_from_mllf_graph


def main():
    # Load the atom type vocabulary from toppar files
    vocab = get_atom_type_vocab()
    print(f"Loaded vocabulary with {len(vocab)} CHARMM atom types")
    print(f"Sample atom types: {list(vocab.keys())[:10]}")
    print()
    
    # Create a simple graph with substituent metadata
    g = Graph(2)
    
    # Node 0: Benzene ring carbon atoms
    g.set_node_info(0, {
        'site': 1,
        'sub': 1,
        'total_charge': 0.0,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61', 'HGR61']  # Aromatic C and H
    })
    
    # Node 1: Different substituent
    g.set_node_info(1, {
        'site': 1,
        'sub': 2,
        'total_charge': -0.5,
        'solvent': 'solv',
        'distinct_atom_types': ['CG2R61', 'NG2R60', 'OG2D1']  # Aromatic C, N, O
    })
    
    # Convert to PyG format
    data, extras = build_pyg_graph_from_mllf_graph(g)
    
    print(f"Graph converted to PyG Data:")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Feature breakdown: 4 base + {len(vocab)} atom types = {data.x.shape[1]}")
    print()
    
    # Examine node 0 features
    node0 = data.x[0]
    print(f"Node 0 features:")
    print(f"  Charge: {node0[0].item()}")
    print(f"  is_vacuum: {node0[1].item()}")
    print(f"  is_solvent: {node0[2].item()}")
    print(f"  is_protein: {node0[3].item()}")
    
    # Check which atom types are present
    present_types = []
    for atom_type, idx in vocab.items():
        if node0[4 + idx].item() > 0:
            present_types.append(atom_type)
    print(f"  Present atom types: {present_types}")
    print()
    
    # Examine node 1 features
    node1 = data.x[1]
    print(f"Node 1 features:")
    print(f"  Charge: {node1[0].item()}")
    print(f"  is_vacuum: {node1[1].item()}")
    print(f"  is_solvent: {node1[2].item()}")
    print(f"  is_protein: {node1[3].item()}")
    
    present_types = []
    for atom_type, idx in vocab.items():
        if node1[4 + idx].item() > 0:
            present_types.append(atom_type)
    print(f"  Present atom types: {present_types}")
    print()
    
    print("Key benefits:")
    print("  ✓ Consistent feature dimensions across all graphs")
    print("  ✓ Vocabulary loaded from standard CHARMM toppar files")
    print("  ✓ Supports all 333 CHARMM atom types")
    print("  ✓ Multi-hot encoding allows multiple atom types per node")


if __name__ == '__main__':
    main()
