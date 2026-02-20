#!/usr/bin/env python3
"""Test that edge creation respects proper directionality for different bias types.

This test verifies the fix where:
- Linear: Only edges FROM sub 1 TO other subs (not reverse)
- Quadratic: Only upper triangle edges (i<j, not reverse)
- Skew/End: Both directions (unchanged)
"""

import torch
from mllf.cb.graph import Graph, EdgeCoeffs
from mllf.cb.graph_utils import build_pyg_graph_from_mllf_graph


def test_linear_directionality_from_sub1():
    """Test that linear edges only exist FROM sub 1 to other subs."""
    # Create graph with 2 sites: site 0 has 3 subs, site 1 has 2 subs
    g = Graph(5)
    
    # Set metadata
    for node_id, (site, sub) in enumerate([(0, 1), (0, 2), (0, 3), (1, 1), (1, 2)]):
        g.set_node_info(node_id, {
            'site': site,
            'sub': sub,
            'total_charge': 0.0,
            'solvent': 'solv',
            'distinct_atom_types': ['CG2R61']
        })
    
    # Set edges with linear coefficients
    # Within site 0: edges involving sub 1 (node 0)
    g.set_edge(0, 1, EdgeCoeffs(linear=1.0, quadratic=0.0, skew=0.0, end=0.0))
    g.set_edge(0, 2, EdgeCoeffs(linear=2.0, quadratic=0.0, skew=0.0, end=0.0))
    # Note: Don't set edge (1,2) - it wouldn't have linear enabled (neither is sub 1)
    
    # Within site 1: edges involving sub 1 (node 3)
    g.set_edge(3, 4, EdgeCoeffs(linear=4.0, quadratic=0.0, skew=0.0, end=0.0))
    
    # Apply connectivity rules to set edge masks
    g.apply_site_connectivity_rules()
    
    # Convert to PyG format
    data, extras = build_pyg_graph_from_mllf_graph(g)
    
    # Extract edge information
    edge_index = data.edge_index
    edge_type = data.edge_type
    rel_names = extras['relation_names']
    
    # Find linear edges
    linear_fwd_idx = rel_names.index('linear_fwd')
    linear_bwd_idx = rel_names.index('linear_bwd')
    
    linear_edges = []
    for k in range(edge_index.shape[1]):
        src = int(edge_index[0, k].item())
        dst = int(edge_index[1, k].item())
        rel = int(edge_type[k].item())
        if rel == linear_fwd_idx or rel == linear_bwd_idx:
            linear_edges.append((src, dst, rel_names[rel]))
    
    print("\nLinear edges found:")
    for src, dst, rel in sorted(linear_edges):
        print(f"  {src} → {dst} ({rel})")
    
    # Extract node metadata to check which is sub 1
    node_to_sub = {}
    for node_id in range(g.num_nodes):
        info = g.get_node_info(node_id)
        node_to_sub[node_id] = info.get('sub')
    
    # Verify: all linear edges should have sub 1 as either src or dst
    # AND only one direction should exist per pair (from sub 1 to other)
    for src, dst, rel in linear_edges:
        sub_src = node_to_sub[src]
        sub_dst = node_to_sub[dst]
        
        # At least one endpoint must be sub 1
        assert sub_src == 1 or sub_dst == 1, \
            f"Linear edge {src}→{dst} doesn't involve sub 1 (subs: {sub_src}, {sub_dst})"
        
        # If src is sub 1, direction should be forward
        if sub_src == 1:
            assert rel == 'linear_fwd', \
                f"Edge from sub 1 ({src}) should be linear_fwd, got {rel}"
        # If dst is sub 1, direction should be backward (which means we create reverse edge)
        else:
            assert rel == 'linear_bwd', \
                f"Edge to sub 1 ({dst}) should be linear_bwd, got {rel}"
    
    # Count edges between pairs - should only be ONE direction per pair
    pair_counts = {}
    for src, dst, rel in linear_edges:
        pair = tuple(sorted([src, dst]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    for pair, count in pair_counts.items():
        assert count == 1, f"Pair {pair} has {count} edges, should have exactly 1"
    
    # Expected pairs (involving sub 1 at each site)
    # Site 0: (0,1), (0,2) - node 0 is sub 1
    # Site 1: (3,4) - node 3 is sub 1
    # Pair (1,2) should NOT exist (neither is sub 1)
    expected_pairs = {(0, 1), (0, 2), (3, 4)}
    actual_pairs = set(pair_counts.keys())
    
    assert actual_pairs == expected_pairs, \
        f"Expected linear pairs: {expected_pairs}, got: {actual_pairs}"
    
    print("✅ Linear directionality test passed!")


def test_quadratic_upper_triangle_only():
    """Test that quadratic edges only exist for upper triangle (i<j)."""
    # Create simple graph with 3 nodes at same site
    g = Graph(3)
    
    for node_id in range(3):
        g.set_node_info(node_id, {
            'site': 0,
            'sub': node_id + 1,
            'total_charge': 0.0,
            'solvent': 'solv',
            'distinct_atom_types': ['CG2R61']
        })
    
    # Set edges with quadratic coefficients
    g.set_edge(0, 1, EdgeCoeffs(linear=0.0, quadratic=1.0, skew=0.0, end=0.0))
    g.set_edge(0, 2, EdgeCoeffs(linear=0.0, quadratic=2.0, skew=0.0, end=0.0))
    g.set_edge(1, 2, EdgeCoeffs(linear=0.0, quadratic=3.0, skew=0.0, end=0.0))
    
    # Apply connectivity rules to set edge masks
    g.apply_site_connectivity_rules()
    
    # Convert to PyG format
    data, extras = build_pyg_graph_from_mllf_graph(g)
    
    # Extract quadratic edges
    edge_index = data.edge_index
    edge_type = data.edge_type
    rel_names = extras['relation_names']
    
    quad_fwd_idx = rel_names.index('quadratic_fwd')
    quad_bwd_idx = rel_names.index('quadratic_bwd')
    
    quad_edges = []
    for k in range(edge_index.shape[1]):
        src = int(edge_index[0, k].item())
        dst = int(edge_index[1, k].item())
        rel = int(edge_type[k].item())
        if rel == quad_fwd_idx or rel == quad_bwd_idx:
            quad_edges.append((src, dst, rel_names[rel]))
    
    print("\nQuadratic edges found:")
    for src, dst, rel in sorted(quad_edges):
        print(f"  {src} → {dst} ({rel})")
    
    # Verify: only upper triangle (i<j) should exist
    for src, dst, rel in quad_edges:
        if src < dst:
            # Upper triangle: should be forward
            assert rel == 'quadratic_fwd', \
                f"Upper triangle edge {src}→{dst} should be quadratic_fwd, got {rel}"
        else:
            # Lower triangle: should be backward
            assert rel == 'quadratic_bwd', \
                f"Lower triangle edge {src}→{dst} should be quadratic_bwd, got {rel}"
    
    # Count edges - should be exactly ONE per pair (only one direction)
    pair_counts = {}
    for src, dst, rel in quad_edges:
        pair = tuple(sorted([src, dst]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    for pair, count in pair_counts.items():
        assert count == 1, f"Pair {pair} has {count} edges, should have exactly 1"
    
    # Should have exactly 3 pairs (all combinations of 3 nodes)
    assert len(pair_counts) == 3, f"Expected 3 quadratic pairs, got {len(pair_counts)}"
    
    print("✅ Quadratic upper triangle test passed!")


def test_skew_end_bidirectional():
    """Test that skew and end edges exist in BOTH directions."""
    # Create simple graph with 3 nodes at same site
    g = Graph(3)
    
    for node_id in range(3):
        g.set_node_info(node_id, {
            'site': 0,
            'sub': node_id + 1,
            'total_charge': 0.0,
            'solvent': 'solv',
            'distinct_atom_types': ['CG2R61']
        })
    
    # Set edges with skew and end coefficients
    g.set_edge(0, 1, EdgeCoeffs(linear=0.0, quadratic=0.0, skew=1.0, end=2.0))
    g.set_edge(0, 2, EdgeCoeffs(linear=0.0, quadratic=0.0, skew=3.0, end=4.0))
    g.set_edge(1, 2, EdgeCoeffs(linear=0.0, quadratic=0.0, skew=5.0, end=6.0))
    
    # Apply connectivity rules to set edge masks
    g.apply_site_connectivity_rules()
    
    # Convert to PyG format
    data, extras = build_pyg_graph_from_mllf_graph(g)
    
    # Extract skew and end edges
    edge_index = data.edge_index
    edge_type = data.edge_type
    rel_names = extras['relation_names']
    
    skew_fwd_idx = rel_names.index('skew_fwd')
    skew_bwd_idx = rel_names.index('skew_bwd')
    end_fwd_idx = rel_names.index('end_fwd')
    end_bwd_idx = rel_names.index('end_bwd')
    
    skew_edges = []
    end_edges = []
    for k in range(edge_index.shape[1]):
        src = int(edge_index[0, k].item())
        dst = int(edge_index[1, k].item())
        rel = int(edge_type[k].item())
        if rel == skew_fwd_idx or rel == skew_bwd_idx:
            skew_edges.append((src, dst, rel_names[rel]))
        elif rel == end_fwd_idx or rel == end_bwd_idx:
            end_edges.append((src, dst, rel_names[rel]))
    
    print("\nSkew edges found:")
    for src, dst, rel in sorted(skew_edges):
        print(f"  {src} → {dst} ({rel})")
    
    print("\nEnd edges found:")
    for src, dst, rel in sorted(end_edges):
        print(f"  {src} → {dst} ({rel})")
    
    # For each bias type, verify BOTH directions exist for each pair
    for bias_name, edges in [('skew', skew_edges), ('end', end_edges)]:
        # Count edges per pair, tracking directions
        pair_directions = {}
        for src, dst, rel in edges:
            pair = tuple(sorted([src, dst]))
            if pair not in pair_directions:
                pair_directions[pair] = {'forward': False, 'backward': False}
            
            # Determine if this is forward or backward
            if src < dst:
                pair_directions[pair]['forward'] = True
            else:
                pair_directions[pair]['backward'] = True
        
        # Verify both directions exist for each pair
        for pair, directions in pair_directions.items():
            assert directions['forward'] and directions['backward'], \
                f"{bias_name} pair {pair} missing direction: {directions}"
        
        # Should have 3 pairs (all combinations of 3 nodes)
        assert len(pair_directions) == 3, \
            f"Expected 3 {bias_name} pairs, got {len(pair_directions)}"
    
    print("✅ Skew/End bidirectional test passed!")


def test_mixed_bias_types():
    """Test that different bias types follow their respective directionality rules simultaneously."""
    # Create graph with multiple bias types
    g = Graph(4)
    
    # Site 0: nodes 0,1,2 (subs 1,2,3)
    # Site 1: nodes 3 (sub 1)
    for node_id, (site, sub) in enumerate([(0, 1), (0, 2), (0, 3), (1, 1)]):
        g.set_node_info(node_id, {
            'site': site,
            'sub': sub,
            'total_charge': 0.0,
            'solvent': 'solv',
            'distinct_atom_types': ['CG2R61']
        })
    
    # Set edges with ALL bias types
    g.set_edge(0, 1, EdgeCoeffs(linear=1.0, quadratic=2.0, skew=3.0, end=4.0))
    g.set_edge(0, 2, EdgeCoeffs(linear=5.0, quadratic=6.0, skew=7.0, end=8.0))
    # Note: Edge (1,2) will have linear disabled by edge mask (neither is sub 1)
    g.set_edge(1, 2, EdgeCoeffs(linear=9.0, quadratic=10.0, skew=11.0, end=12.0))
    
    # Apply connectivity rules to set edge masks
    g.apply_site_connectivity_rules()
    
    # Convert to PyG format
    data, extras = build_pyg_graph_from_mllf_graph(g)
    
    # Extract all edges by bias type
    edge_index = data.edge_index
    edge_type = data.edge_type
    rel_names = extras['relation_names']
    
    edges_by_type = {
        'linear': [],
        'quadratic': [],
        'skew': [],
        'end': []
    }
    
    for k in range(edge_index.shape[1]):
        src = int(edge_index[0, k].item())
        dst = int(edge_index[1, k].item())
        rel = int(edge_type[k].item())
        rel_name = rel_names[rel]
        
        for bias_type in edges_by_type.keys():
            if rel_name.startswith(bias_type):
                edges_by_type[bias_type].append((src, dst, rel_name))
    
    # Verify linear: only involving sub 1 (node 0 in site 0), one direction per pair
    # Note: Edge (1,2) has linear disabled by mask since neither is sub 1
    linear_pairs = {}
    for src, dst, rel in edges_by_type['linear']:
        pair = tuple(sorted([src, dst]))
        linear_pairs[pair] = linear_pairs.get(pair, 0) + 1
        # Must involve node 0 (sub 1 of site 0)
        assert src == 0 or dst == 0, f"Linear edge {src}→{dst} must involve sub 1 (node 0)"
    
    # Should be exactly ONE edge per pair
    for pair, count in linear_pairs.items():
        assert count == 1, f"Linear pair {pair} has {count} edges, expected 1"
    
    # Expected: (0,1) and (0,2) only, NOT (1,2)
    assert set(linear_pairs.keys()) == {(0, 1), (0, 2)}, \
        f"Linear pairs incorrect: {set(linear_pairs.keys())}"
    
    # Verify quadratic: one direction per pair
    quad_pairs = {}
    for src, dst, rel in edges_by_type['quadratic']:
        pair = tuple(sorted([src, dst]))
        quad_pairs[pair] = quad_pairs.get(pair, 0) + 1
    
    for pair, count in quad_pairs.items():
        assert count == 1, f"Quadratic pair {pair} has {count} edges, expected 1"
    
    # All 3 pairs should exist
    assert len(quad_pairs) == 3, f"Expected 3 quadratic pairs, got {len(quad_pairs)}"
    
    # Verify skew and end: both directions per pair
    for bias_type in ['skew', 'end']:
        pair_directions = {}
        for src, dst, rel in edges_by_type[bias_type]:
            pair = tuple(sorted([src, dst]))
            if pair not in pair_directions:
                pair_directions[pair] = set()
            pair_directions[pair].add((src, dst))
        
        # Each pair should have both directions
        for pair, directions in pair_directions.items():
            assert len(directions) == 2, \
                f"{bias_type} pair {pair} has {len(directions)} directions, expected 2"
    
    print("✅ Mixed bias types test passed!")


if __name__ == '__main__':
    test_linear_directionality_from_sub1()
    test_quadratic_upper_triangle_only()
    test_skew_end_bidirectional()
    test_mixed_bias_types()
    print("\n" + "="*60)
    print("All edge directionality tests passed! ✅")
    print("="*60)
