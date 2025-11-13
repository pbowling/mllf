"""Helpers to convert existing Graph objects to PyTorch Geometric Data.

This module is intentionally defensive: it will try to import the project's
Graph class and fall back to handling networkx-like graph objects.
"""
from typing import Tuple
import torch
from torch_geometric.data import Data



def _node_feature_from_meta(meta: dict):
    """Create a small numeric vector from node metadata.

    Expected keys in meta: 'total_charge', 'solvent' (bool), 'distinct_atom_types' (list or count)
    The function returns a 1-D torch.float tensor.
    """
    charge = float(meta.get('total_charge', 0.0))
    solvent = 1.0 if meta.get('solvent') else 0.0
    dat = meta.get('distinct_atom_types')
    if isinstance(dat, (list, tuple)):
        nat = float(len(dat))
    else:
        try:
            nat = float(dat)
        except Exception:
            nat = 0.0
    return torch.tensor([charge, solvent, nat], dtype=torch.get_default_dtype())


def build_pyg_graph_from_mllf_graph(g, relation_names: list = None) -> Tuple[object, dict]:
    """Convert a Graph-like object `g` into a PyG Data object and metadata.

    We expand each undirected graph edge into up to four directed relation edges,
    one per bias type. The default `relation_names` is ['linear','quadratic','skew','end'].

    Returns (pyg_data, extras) where extras contain mappings helpful for training.
    """

    if relation_names is None:
        base_relation_names = ['linear', 'quadratic', 'skew', 'end']
    else:
        base_relation_names = list(relation_names)

    # Expand base relations into directed relation types: e.g. 'linear_fwd', 'linear_bwd'
    relation_names = []
    base_relation_map = {}
    for r in base_relation_names:
        fwd = f"{r}_fwd"
        bwd = f"{r}_bwd"
        base_relation_map[r] = (fwd, bwd)
        relation_names.append(fwd)
        relation_names.append(bwd)

    rel_to_idx = {r: i for i, r in enumerate(relation_names)}

    # collect node features
    node_feats = []
    for i in range(g.num_nodes):
        meta = g.get_node_info(i) if hasattr(g, 'get_node_info') else {}
        node_feats.append(_node_feature_from_meta(meta))
    x = torch.stack(node_feats, dim=0)

    # expand edges: for each undirected (i,j) and for each bias that is allowed.
    # For each base bias we create two directed relation types so that A->B and B->A
    # are represented by distinct relation ids and can be learned separately.
    src = []
    dst = []
    edge_type_list = []
    edge_attr_list = []

    # Graph.edges is stored as dict keyed by (i,j) -> EdgeCoeffs
    for (i, j), coeffs in getattr(g, 'edges', {}).items():
        # determine which bias types are allowed from edge_mask (if present)
        mask = None
        if hasattr(g, 'edge_mask'):
            mask = g.edge_mask.get((i, j))
        for bias in base_relation_names:
            allowed = True if mask is None else bool(mask.get(bias, False))
            if not allowed:
                continue
            fwd_name, bwd_name = base_relation_map[bias]
            fwd_idx = rel_to_idx[fwd_name]
            bwd_idx = rel_to_idx[bwd_name]
            # add directed edge i->j as the forward relation for this bias
            src.append(int(i))
            dst.append(int(j))
            edge_type_list.append(fwd_idx)
            # edge_attr: only include one-hot over directed relation types
            k = len(relation_names)
            one_hot = torch.zeros((k,), dtype=torch.get_default_dtype())
            one_hot[fwd_idx] = 1.0
            edge_attr_list.append(one_hot)
            # add reverse direction j->i as the backward relation type
            src.append(int(j))
            dst.append(int(i))
            edge_type_list.append(bwd_idx)
            one_hot_r = torch.zeros((k,), dtype=torch.get_default_dtype())
            one_hot_r[bwd_idx] = 1.0
            edge_attr_list.append(one_hot_r)

    k = len(relation_names)
    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_attr = torch.zeros((0, k), dtype=torch.get_default_dtype())
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
        edge_attr = torch.stack(edge_attr_list, dim=0)

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr)
    extras = {
        'relation_names': relation_names,
        'relation_map': rel_to_idx,
        'base_relation_map': base_relation_map,
    }
    return data, extras
