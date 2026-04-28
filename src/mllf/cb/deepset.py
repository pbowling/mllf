import torch
import torch.nn as nn

class DeepSetFeatureExtractor(nn.Module):
    """DeepSet feature extractor for substituents using 3D atomic information.
    
    Implements the 4-step pipeline:
    1. Atom-Level Physical Representation: AEV + charge + atom identity
    2. Shared MLP: Compresses each atom's features independently
    3. Permutation-Invariant Pooling: Max-pool across atoms
    4. Returns fixed-size substituent embedding
    
    Args:
        aev_length: Dimension of AEV vectors (default: 2288 for ANI-2x with 11 species)
        num_atom_types: Number of distinct atom types/elements (default: 11 = 10 common + 1 unknown)
        embedding_dim: Output dimension for substituent embedding (default: 64)
        hidden_dim: Hidden layer dimension in the MLP (default: 256)
        include_charge: Whether to include atomic charge in features (default: True)
        include_atom_id: Whether to include atom type one-hot in features (default: True)
    """
    def __init__(self, aev_length=2288, num_atom_types=11, embedding_dim=64, 
                 hidden_dim=256, include_charge=True, include_atom_id=True):
        super().__init__()
        
        self.include_charge = include_charge
        self.include_atom_id = include_atom_id
        self.num_atom_types = num_atom_types
        
        # Calculate input dimension: AEV + optional charge (1D) + optional atom_id (one-hot)
        input_dim = aev_length
        if include_charge:
            input_dim += 1
        if include_atom_id:
            input_dim += num_atom_types
        
        # Shared MLP that processes each atom independently
        self.atom_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, aev_tensor, charges=None, atom_ids=None):
        """Forward pass through DeepSet encoder.
        
        Args:
            aev_tensor: [num_atoms, aev_length] - AEV vectors for each atom
            charges: [num_atoms, 1] or [num_atoms] - Partial charges (optional)
            atom_ids: [num_atoms] - Integer atom type IDs (optional)
            
        Returns:
            substituent_embedding: [embedding_dim] - Fixed-size substituent representation
        """
        features = [aev_tensor]
        
        # Concatenate charge if provided
        if self.include_charge:
            if charges is None:
                raise ValueError("charges must be provided when include_charge=True")
            if charges.dim() == 1:
                charges = charges.unsqueeze(1)
            features.append(charges)
        
        # Concatenate one-hot atom identity if provided
        if self.include_atom_id:
            if atom_ids is None:
                raise ValueError("atom_ids must be provided when include_atom_id=True")
            # Convert to one-hot encoding
            atom_one_hot = torch.nn.functional.one_hot(atom_ids, num_classes=self.num_atom_types).float()
            features.append(atom_one_hot)
        
        # Concatenate all features
        atom_features = torch.cat(features, dim=-1)
        
        # Pass through shared MLP
        atom_embeddings = self.atom_mlp(atom_features)
        
        # Max-Pool across the atoms to summarize the substituent
        # This is permutation-invariant and handles variable-sized substituents
        substituent_embedding, _ = torch.max(atom_embeddings, dim=0) 
        
        return substituent_embedding


class AtomBondGNN(nn.Module):
    """Bond-topology-aware substituent encoder using graph neural networks.

    Drop-in replacement for DeepSetFeatureExtractor that uses GINConv message
    passing over the molecular bond graph instead of independent atom processing,
    followed by GlobalAttentionPool instead of max-pool.

    Same interface attributes as DeepSetFeatureExtractor:
    - ``include_charge``, ``include_atom_id`` flags
    - ``atom_mlp[-1].out_features`` for downstream embedding-dim queries

    Args:
        aev_length: Dimension of AEV vectors (default: 2288)
        num_atom_types: Number of distinct atom types/elements (default: 11)
        embedding_dim: Output dimension for substituent embedding (default: 64)
        hidden_dim: Hidden layer dimension in GINEConv MLPs (default: 256)
        include_charge: Whether to include atomic charge in features (default: True)
        include_atom_id: Whether to include atom type one-hot in features (default: True)
    """

    def __init__(self, aev_length=2288, num_atom_types=11, embedding_dim=64,
                 hidden_dim=256, include_charge=True, include_atom_id=True,
                 bond_attr_dim=1, num_gin_layers=4):
        super().__init__()
        from torch_geometric.nn import GINEConv

        self.include_charge = include_charge
        self.include_atom_id = include_atom_id
        self.num_atom_types = num_atom_types
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bond_attr_dim = bond_attr_dim
        self.num_gin_layers = num_gin_layers

        input_dim = aev_length
        if include_charge:
            input_dim += 1
        if include_atom_id:
            input_dim += num_atom_types

        def _gin_mlp(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.ReLU(),
                nn.Linear(out_d, out_d),
            )

        # Substituent stream: processes sub atoms only through their sub-graph
        self.sub_input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.sub_gin_layers = nn.ModuleList([
            GINEConv(_gin_mlp(hidden_dim, hidden_dim), edge_dim=bond_attr_dim)
            for _ in range(num_gin_layers)
        ])

        # Core stream: processes core + ref-sub atoms through the core sub-graph
        self.core_input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.core_gin_layers = nn.ModuleList([
            GINEConv(_gin_mlp(hidden_dim, hidden_dim), edge_dim=bond_attr_dim)
            for _ in range(num_gin_layers)
        ])

        # Hybrid attentional pooling:
        #   Gate: concat(sub_node[H], core_summary[H]) -> scalar weight
        #         = "how important is this sub atom given this scaffold?"
        #   Features: sub_node[H] -> embedding_dim  (pure substituent identity)
        self.gate_nn = nn.Linear(hidden_dim * 2, 1)
        self.pool_nn = nn.Linear(hidden_dim, embedding_dim)

        # atom_mlp[-1] compatibility shim for graph_utils fallback embedding-dim query
        self.atom_mlp = nn.Sequential(nn.Linear(hidden_dim, embedding_dim))

    def forward(self, aev_tensor, charges=None, atom_ids=None,
                bond_edge_index=None, bond_edge_attr=None, sub_mask=None):
        """Forward pass through AtomBondGNN (dual-stream architecture).

        Atoms are split into two independent GINEConv streams:
          - **Sub stream**: substituent atoms processed through their own sub-graph.
          - **Core stream**: core + ref-sub atoms processed through the core sub-graph.

        The core stream produces a single mean-pooled summary vector that is
        concatenated into the *gate* of the attentional pooling step, making
        the attention weights scaffold-aware without contaminating the sub feature
        path.  The final embedding is a weighted sum of purely sub-derived features.

        When ``sub_mask`` is ``None`` (legacy mode) all atoms are processed through
        the sub stream with a zero core summary.

        Args:
            aev_tensor: [N, aev_length]
            charges: [N] or [N, 1] partial charges (required when include_charge=True)
            atom_ids: [N] integer atom-type IDs (required when include_atom_id=True)
            bond_edge_index: [2, 2E] bidirectional bond edges over full ligand (optional)
            bond_edge_attr: [2E, bond_attr_dim] bond-type weights (optional)
            sub_mask: [N] boolean tensor marking substituent atoms.  When provided
                the bond graph is partitioned into sub-only and core-only sub-graphs.

        Returns:
            substituent_embedding: [embedding_dim]
        """
        dev = aev_tensor.device

        def _build_features(aev, chg, ids):
            parts = [aev]
            if self.include_charge:
                if chg is None:
                    raise ValueError("charges must be provided when include_charge=True")
                parts.append(chg.unsqueeze(1) if chg.dim() == 1 else chg)
            if self.include_atom_id:
                if ids is None:
                    raise ValueError("atom_ids must be provided when include_atom_id=True")
                parts.append(torch.nn.functional.one_hot(ids, self.num_atom_types).float())
            return torch.cat(parts, dim=-1)

        def _prep_edges(n, ei, ea):
            """Return (edge_index, edge_attr), falling back to self-loops."""
            if ei is not None and ei.size(1) > 0:
                ea2 = ea.to(dev) if ea is not None else torch.zeros(ei.size(1), self.bond_attr_dim, device=dev)
                return ei.to(dev), (ea2.unsqueeze(1) if ea2.dim() == 1 else ea2)
            idx = torch.arange(n, device=dev)
            return torch.stack([idx, idx]), torch.zeros(n, self.bond_attr_dim, device=dev)

        def _run_stream(proj, gin_layers, aev, chg, ids, ei, ea):
            x = proj(_build_features(aev, chg, ids))
            ei2, ea2 = _prep_edges(x.size(0), ei, ea)
            for gin in gin_layers:
                x = torch.relu(gin(x, ei2, ea2) + x)
            return x

        if sub_mask is not None:
            n_sub = int(sub_mask.sum())
            core_mask = ~sub_mask

            # Partition bond graph into sub-only and core-only sub-graphs
            if bond_edge_index is not None and bond_edge_index.size(1) > 0:
                src, dst = bond_edge_index
                sub_e  = (src < n_sub) & (dst < n_sub)
                core_e = (src >= n_sub) & (dst >= n_sub)
                sub_ei  = bond_edge_index[:, sub_e]
                sub_ea  = bond_edge_attr[sub_e]  if bond_edge_attr is not None else None
                core_ei = bond_edge_index[:, core_e] - n_sub   # reindex to 0-based
                core_ea = bond_edge_attr[core_e] if bond_edge_attr is not None else None
            else:
                sub_ei = sub_ea = core_ei = core_ea = None

            # Sub stream
            x_sub = _run_stream(
                self.sub_input_proj, self.sub_gin_layers,
                aev_tensor[sub_mask],
                charges[sub_mask]  if charges  is not None else None,
                atom_ids[sub_mask] if atom_ids is not None else None,
                sub_ei, sub_ea,
            )  # [n_sub, H]

            # Core stream -> mean-pool to single scaffold summary
            n_core = int(core_mask.sum())
            if n_core > 0:
                x_core = _run_stream(
                    self.core_input_proj, self.core_gin_layers,
                    aev_tensor[core_mask],
                    charges[core_mask]  if charges  is not None else None,
                    atom_ids[core_mask] if atom_ids is not None else None,
                    core_ei, core_ea,
                )  # [n_core, H]
                core_summary = x_core.mean(0, keepdim=True).expand(n_sub, -1)  # [n_sub, H]
            else:
                core_summary = torch.zeros(n_sub, self.hidden_dim, device=dev)
        else:
            # Legacy: single stream over all atoms, zero core context
            n_sub = aev_tensor.size(0)
            x_sub = _run_stream(
                self.sub_input_proj, self.sub_gin_layers,
                aev_tensor, charges, atom_ids,
                bond_edge_index, bond_edge_attr,
            )
            core_summary = torch.zeros(n_sub, self.hidden_dim, device=dev)

        # Hybrid attentional pooling
        # Gate: scaffold-aware importance score per sub atom
        gates   = torch.softmax(self.gate_nn(torch.cat([x_sub, core_summary], dim=-1)), dim=0)  # [n_sub, 1]
        # Features: pure substituent identity
        features = self.pool_nn(x_sub)                                                          # [n_sub, emb]
        return (gates * features).sum(0)                                                        # [emb]