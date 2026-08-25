"""Edge value policy built on top of a node encoder.

For each directed edge this policy produces a vector of means and a vector
of log-standard-deviations (one per predicted coefficient). The MLP outputs
concatenated [mu_1,...,mu_D, logsigma_1,...,logsigma_D] which are split
and used to parameterize a per-edge independent Gaussian distribution.
The agent samples continuous actions v_ij ~ N(mu_ij, sigma_ij^2) for every
directed edge and returns sampled values plus per-edge log-probabilities.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mllf.cb.bayesian_head import BayesianLinearHead


class BiasHeadMLP(nn.Module):
    """Single fully-independent MLP for one MSLD bias type.

    Takes the full edge input and produces a (mean, log_std) pair
    for one bias coefficient.  Because each bias type has its own complete
    stack — input → hidden → output — gradient from one type cannot
    physically corrupt the weights used by any other type.

    In **Bayesian mode** (``bayesian=True``, for NeuralLinear + Thompson
    Sampling) the same ``trunk`` feature extractor is kept, but the final
    deterministic ``Linear(64, 2)`` readout is replaced by a
    :class:`~mllf.cb.bayesian_head.BayesianLinearHead`: a closed-form Bayesian
    linear regression on the trunk's 64-D output. ``forward()`` still returns
    an ``[E, 2]`` tensor for drop-in compatibility with existing callers
    (column 0 = posterior mean, column 1 = ``0.5*log(var)`` in place of
    ``log_std``), but the head is intended to be driven through
    ``forward_features`` / the ``BayesianLinearHead`` API directly for
    Thompson-sampled actions and closed-form posterior updates.
    """

    def __init__(self, in_dim: int, bayesian: bool = False, prior_precision: float = 1.0):
        super().__init__()
        self.bayesian = bool(bayesian)
        self.feat_dim = 64
        # Every single bias head gets its own private projection layers.
        # No cross-talk or gradient pollution can occur between bias types.
        # This trunk ("phi") is shared by both the legacy deterministic
        # readout and the Bayesian last layer — it is the feature extractor
        # a NeuralLinear model sits its Bayesian linear regression on top of.
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),           # 1024 → 512
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),       # 512 → 256
            nn.ReLU(),
            nn.Linear(in_dim // 4, self.feat_dim),    # 256 → 64
            nn.ReLU(),
        )
        if self.bayesian:
            self.bayesian_head = BayesianLinearHead(feat_dim=self.feat_dim,
                                                      prior_precision=prior_precision)
            self.readout = None
        else:
            self.bayesian_head = None
            self.readout = nn.Linear(self.feat_dim, 2)  # 64 → 2 (mean, log_std)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[E, 64]`` trunk features ``z = phi(x)`` (pre-readout)."""
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [E, 2] tensor of (mean_raw, log_std_raw)."""
        z = self.trunk(x)
        if self.bayesian:
            mean, var = self.bayesian_head.predict(z)
            log_std = 0.5 * torch.log(var.clamp(min=1e-12))
            return torch.stack([mean, log_std], dim=-1)
        return self.readout(z)


class EdgeValueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_bias_types: int = 4,
                 bias_embed_dim: int = 16, bayesian: bool = False,
                 prior_precision: float = 1.0):
        """Four completely independent MLPs, one per MSLD bias type.

        Replaces the previous shared-trunk + 4-heads design.  Each
        :class:`BiasHeadMLP` owns its entire feature-extraction stack, so
        the gradient from one bias type (e.g. the noisy signal)
        cannot overwrite features learned by another type (e.g. linear).

        The public API is identical to the old class — ``forward(x)`` returns
        ``[E, 2*num_bias_types]`` — so no callers need to change.

        Args:
            in_dim: Input dimension.
            hidden: Unused; kept for API compatibility with old constructor.
            num_bias_types: Number of bias types (default 4).
            bias_embed_dim: Unused; kept for API compatibility.
            bayesian: If True, each :class:`BiasHeadMLP` uses a
                :class:`~mllf.cb.bayesian_head.BayesianLinearHead` last layer
                instead of a deterministic ``Linear(64, 2)`` (NeuralLinear mode).
        """
        super().__init__()
        self.num_bias_types = num_bias_types
        self.bayesian = bool(bayesian)
        # One fully-independent MLP per bias type.
        # Exposed as ``mlps`` so the optimizer can create per-head param groups.
        self.mlps = nn.ModuleList([
            BiasHeadMLP(in_dim, bayesian=self.bayesian, prior_precision=prior_precision)
            for _ in range(num_bias_types)
        ])

    def forward_features(self, x: torch.Tensor,
                          edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return per-bias-type trunk features, routed or not.

        Args:
            x: ``[E, in_dim]`` edge input features.
            edge_type: ``[E]`` integer relation-type index. When provided,
                each edge is routed to ``mlps[edge_type // 2]`` (matches
                ``forward``'s routed path — used by the pretraining
                fully-connected 4x-edges-per-pair graph structure). When
                ``None`` (the online-training graph structure: one edge per
                pair, every bias type independently meaningful for every
                edge — matches ``forward``'s legacy no-routing path), every
                edge gets a feature vector from **each** of the ``D`` trunks.

        Returns:
            Routed (``edge_type`` given): ``[E, 64]``, one trunk's output per
            edge. Unrouted (``edge_type=None``): ``[E, D, 64]``, all ``D``
            trunks' outputs for every edge (index the bias-type dim to get a
            per-head ``[E, 64]`` slice, e.g. ``z[:, d, :]``).
        """
        D = self.num_bias_types
        feat_dim = self.mlps[0].feat_dim
        if edge_type is None:
            # Legacy/online-training path: every edge is independently
            # meaningful to every bias type's trunk (no routing to discard).
            return torch.stack([mlp.forward_features(x) for mlp in self.mlps], dim=1)  # [E, D, feat]

        bias_type = edge_type // 2  # [E], values 0..D-1
        out = torch.zeros(x.size(0), feat_dim, dtype=x.dtype, device=x.device)
        for i, mlp in enumerate(self.mlps):
            mask = (bias_type == i)
            if mask.any():
                out[mask] = mlp.forward_features(x[mask])
        return out

    def forward(self, x: torch.Tensor,
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run independent MLPs and return concatenated mean/log_std output.

        Args:
            x: ``[E, in_dim]`` edge input features.
            edge_type: ``[E]`` integer relation-type index (0-indexed; values
                ``0``/``1`` = linear fwd/bwd, ``2``/``3`` = quadratic, etc.).
                When provided, each edge is routed only to its own MLP
                (``mlps[edge_type // 2]``).  Gradients cannot cross bias types.
                When ``None``, all MLPs process all edges (legacy path).

        Returns:
            ``[E, 2*D]``: ``[mean_0, ..., mean_D-1, logstd_0, ..., logstd_D-1]``.
            In the routed path, only the relevant ``(mean_d, logstd_d)`` slot is
            populated for each edge; all other slots remain zero.
        """
        D = self.num_bias_types
        if edge_type is None:
            # Legacy path: all MLPs process all edges (used when edge_type not available).
            outputs = [mlp(x) for mlp in self.mlps]   # list of [E, 2]
            stacked = torch.stack(outputs, dim=1)       # [E, D, 2]
            means = stacked[:, :, 0]                    # [E, D]
            log_stds = stacked[:, :, 1]                 # [E, D]
            return torch.cat([means, log_stds], dim=-1)  # [E, 2*D]

        # Routed path: each edge processed only by mlps[edge_type // 2].
        # bias_type_index: linear_fwd=0, linear_bwd=0, quad_fwd=1, quad_bwd=1, ...
        bias_type = edge_type // 2  # [E], values 0..D-1
        out = torch.zeros(x.size(0), 2 * D, dtype=x.dtype, device=x.device)
        for i, mlp in enumerate(self.mlps):
            mask = (bias_type == i)
            if mask.any():
                mlp_out = mlp(x[mask])        # [E_i, 2]
                out[mask, i] = mlp_out[:, 0]      # mean for dim i
                out[mask, i + D] = mlp_out[:, 1]  # log_std for dim i
        return out  # [E, 2*D]


class EdgePolicy(nn.Module):
    def __init__(self, encoder: nn.Module, emb_dim: int, edge_feat_dim: int = 0,
                 mlp_hidden: int = 64, mlp_out_dim: int = 4, p1_dim: int = 0):
        super().__init__()
        self.encoder = encoder
        self.mlp_out_dim = int(mlp_out_dim)
        self.p1_dim = int(p1_dim)
        # Input to edge-mlp: [P1_src, P2_src, P1_dst, P2_dst, edge_feat] when p1_dim>0
        # otherwise: [P2_src, P2_dst, edge_feat] (backward-compatible)
        in_dim = 2 * (emb_dim + p1_dim) + edge_feat_dim
        self.edge_mlp = EdgeValueMLP(in_dim, mlp_hidden, num_bias_types=self.mlp_out_dim)
        # Scale factors for the 4 MSLD bias types: [linear, quadratic, skew, end].
        # Empirical max across 20K+ pretraining runs with ~10% headroom.
        # Registered as non-persistent so they are NOT saved in state_dict
        # (they are constants, not learned state).
        self.register_buffer(
            'scale_factors',
            torch.tensor([305.0, 520.0, 85.0, 30.0]),
            persistent=False,
        )

    def forward_node_embeddings(self, x, edge_index, edge_type):
        return self.encoder(x, edge_index, edge_type)

    def edge_inputs(self, node_emb: torch.Tensor, edge_index: torch.LongTensor,
                    edge_feat: Optional[torch.Tensor] = None,
                    p1_emb: Optional[torch.Tensor] = None):
        """Build per-edge input tensor.

        When p1_emb is provided (skip connection), concatenates
        [P1_src, P2_src, P1_dst, P2_dst, edge_feat]; otherwise [P2_src, P2_dst, edge_feat].
        """
        # edge_index: [2, E]
        src = edge_index[0]
        dst = edge_index[1]
        if p1_emb is not None:
            parts = [p1_emb[src], node_emb[src], p1_emb[dst], node_emb[dst]]
        else:
            parts = [node_emb[src], node_emb[dst]]
        if edge_feat is not None:
            parts.append(edge_feat)
        return torch.cat(parts, dim=-1)

    def forward_edges(self, node_emb: torch.Tensor, edge_index: torch.LongTensor,
                      edge_feat: Optional[torch.Tensor] = None,
                      p1_emb: Optional[torch.Tensor] = None):
        inp = self.edge_inputs(node_emb, edge_index, edge_feat, p1_emb)
        out = self.edge_mlp(inp)  # [E, 2*D]
        # ensure 2D
        if out.dim() == 1:
            out = out.unsqueeze(0)
        E, C = out.shape
        D = self.mlp_out_dim
        if C != 2 * D:
            # fallback: if out dimension doesn't match expectation, treat as D=1
            D = C // 2 if C >= 2 else 1
        mean = out[:, :D]
        log_std = out[:, D: D + D]
        
        # Scale mean outputs using softsign instead of tanh.
        # softsign(z) = z/(1+|z|) has the same [-1,1] range as tanh but gradient
        # 1/(1+|z|)^2 stays alive even at large |z|, eliminating the saturation
        # zone that causes 'dead head' collapse after ~20 training epochs.
        mean = F.softsign(mean) * self.scale_factors.unsqueeze(0)
        
        # Clamp log_std to prevent extreme standard deviations that can cause NaN
        # exp(-20) ≈ 2e-9, exp(2.0) ≈ 7.4 — provides exploration while preventing extreme outliers
        # Note: Higher values (e.g., 3.5 → std≈33) can produce samples far beyond intended ranges
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        return mean, log_std

    def get_actions(self, x, edge_index, edge_type, edge_feat: Optional[torch.Tensor] = None, deterministic: bool = False):
        """Return sampled actions and log-probabilities for every directed edge.

        Returns:
            actions: Tensor of shape [E] or [E, D]
            logp: per-edge scalar log-prob Tensor [E]
            mean: Tensor of shape [E, D]
            log_std: Tensor of shape [E, D]
        """
        node_emb = self.forward_node_embeddings(x, edge_index, edge_type)
        p1_for_skip = x if self.p1_dim > 0 else None
        mean, log_std = self.forward_edges(node_emb, edge_index, edge_feat, p1_for_skip)
        # ensure 2D
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(-1)

        if deterministic:
            actions = mean
            logp_per = torch.zeros_like(mean)
        else:
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            actions = dist.rsample()
            logp_per = dist.log_prob(actions)
            
            # Clip sampled actions to the scale-factor range (+5% margin).
            # Matches the tanh scale factors in forward_edges.
            clip_limits = self.scale_factors * 1.05
            actions = torch.clamp(actions, -clip_limits.unsqueeze(0), clip_limits.unsqueeze(0))

        # sum logp across output dims to get per-edge scalar logp
        logp = logp_per.sum(dim=-1)
        # if single-dim, squeeze actions to [E]
        if actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        return actions, logp, mean, log_std

    def evaluate_logp(
        self,
        x,
        edge_index,
        edge_type,
        edge_feat: Optional[torch.Tensor],
        saved_actions: torch.Tensor,
    ):
        """Evaluate log π_θ(saved_actions | x) under current policy parameters.

        Used for on-policy REINFORCE: the actions that were actually submitted to
        the simulation are treated as fixed (no gradient through them), while the
        distribution parameters (mean, log_std) retain gradients so that the policy
        update correctly reinforces or suppresses the actions that led to the reward.

        Args:
            saved_actions: Tensor [E] or [E, D] — actions from the simulation run
                (i.e., the clipped actions stored in epoch_results.pt).

        Returns:
            logp_per_dim: [E, D] per-dimension log-prob (for multi-output REINFORCE).
            log_std: [E, D] current log_std (for entropy regularisation caller).
        """
        node_emb = self.forward_node_embeddings(x, edge_index, edge_type)
        p1_for_skip = x if self.p1_dim > 0 else None
        mean, log_std = self.forward_edges(node_emb, edge_index, edge_feat, p1_for_skip)
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(-1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        # saved_actions treated as a fixed constant — no gradient through them
        a = saved_actions.detach().to(mean.device)
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        # Return per-dimension log probs [E, D] to enable proper multi-output REINFORCE
        # where each dimension gets gradient proportional to its own log prob, not the sum
        logp_per_dim = dist.log_prob(a)  # [E, D]
        return logp_per_dim, log_std

    @classmethod
    def from_pyg_data(cls, encoder: nn.Module, emb_dim: int, data, mlp_hidden: int = 64,
                      mlp_out_dim: int = 1, p1_dim: int = 0):
        """Construct an EdgePolicy using a PyG ``data`` object to infer edge_feat_dim.

        Reads ``data.edge_attr`` (if present) for per-edge feature size.
        Pass ``p1_dim=data.x.size(1)`` to enable the skip-connection path where
        pre-RGCN node embeddings are concatenated alongside RGCN embeddings.

        Input dimension: 2*(emb_dim + p1_dim) + edge_feat_dim.
        """
        edge_feat_dim = 0
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            try:
                edge_feat_dim = int(data.edge_attr.shape[1])
            except Exception:
                edge_feat_dim = 0
        return cls(encoder, emb_dim, edge_feat_dim, mlp_hidden, mlp_out_dim, p1_dim)


class UnimolPolicy(nn.Module):
    """Direct policy using pre-computed Uni-Mol embeddings.
    
    Supports two embedding modes:
    
    **Standard Mode** (use_dual_embeddings=False):
    - Single 512D embedding per node (ligand + environment context)
    - Edge input: [diff(512), mean(512)] = 1024D
    - Backward compatible with original design
    
    **Dual Embedding Mode** (use_dual_embeddings=True):
    - Two 512D embeddings per node stacked as [ligand_only, full]
    - Ligand-only: core + sub (captures substituent-specific information)
    - Full: core + sub + environment + ref_subs (captures ligand+environment context)
    - Edge input: [diff_ligand(512), mean_full(512)] = 1024D
      - diff_ligand captures how substituents differ (antisymmetric, sub-dependent)
      - mean_full captures environmental context (symmetric, environment-dependent)
    - Allows MLPs to simultaneously learn substituent-dependent and environment-dependent information
    
    Each edge is passed as 1024D directly to EdgeValueMLP, which contains four
    independent BiasHeadMLPs (one per bias type). Each BiasHeadMLP learns its
    own feature extraction: 1024D → 512D → 256D → 64D → 2D (mean, log_std).
    This allows each bias type to independently discover what information is
    important for its prediction, without a shared projection bottleneck.
    
    No RGCN encoder, no site pooling, no shared projection — purely feed-forward
    edge classification based on pre-trained molecular representations.
    
    Args:
        unimol_dim: Dimension of each Uni-Mol embedding (default: 512).
        mlp_hidden: Hidden size for EdgeValueMLP (default: 64, unused in this design).
        mlp_out_dim: Number of bias types / output coefficients (default: 4).
        use_dual_embeddings: If True, use dual embedding mode; if False, use standard mode (default: False).
        use_bayesian_heads: If True (NeuralLinear + Thompson Sampling mode), each
            bias head's final layer is a :class:`~mllf.cb.bayesian_head.BayesianLinearHead`
            instead of a deterministic ``Linear(64, 2)``. The trunk feature extractor
            (shared with the deterministic mode) is unchanged. Default False keeps the
            original REINFORCE-compatible architecture exactly as before.
        bayesian_prior_precision: Prior precision (ridge strength) for each
            ``BayesianLinearHead`` when ``use_bayesian_heads=True``. Unused otherwise.
    """

    def __init__(self, unimol_dim: int = 512, mlp_hidden: int = 64,
                 mlp_out_dim: int = 4, use_dual_embeddings: bool = False,
                 use_bayesian_heads: bool = False,
                 bayesian_prior_precision: float = 1.0):
        super().__init__()
        self.unimol_dim = unimol_dim
        self.mlp_out_dim = int(mlp_out_dim)
        self.use_dual_embeddings = use_dual_embeddings
        self.use_bayesian_heads = bool(use_bayesian_heads)
        
        # When using dual embeddings:
        # - Input format: [N, 1024] where first 512 = ligand-only, last 512 = full system with environment
        # - Edge input: [diff_ligand, mean_full] = 1024D
        # Otherwise:
        # - Input format: [N, 512] ligand+environment
        # - Edge input: [diff, mean] = 1024D
        in_dim = 2 * unimol_dim if not use_dual_embeddings else 1024
        self.edge_mlp = EdgeValueMLP(in_dim, hidden=mlp_hidden, num_bias_types=mlp_out_dim,
                                      bayesian=self.use_bayesian_heads,
                                      prior_precision=bayesian_prior_precision)
        
        # Scale factors for the 4 MSLD bias types: [linear, quadratic, skew, end].
        # Empirical max across 20K+ pretraining runs with ~10% headroom.
        self.register_buffer(
            'scale_factors',
            torch.tensor([305.0, 520.0, 85.0, 30.0]),
            persistent=False,
        )
    
    def _build_edge_input(self, unimol_embeddings: torch.Tensor,
                          edge_index: torch.LongTensor) -> torch.Tensor:
        """Build the raw ``[E, in_dim]`` per-edge input from node embeddings.

        Shared by ``_forward_edges`` (legacy deterministic/REINFORCE path) and
        ``forward_features`` (NeuralLinear path) so the edge-input construction
        logic exists in exactly one place.

        Args:
            unimol_embeddings: [N, emb_dim] pre-computed Uni-Mol embeddings.
                - When use_dual_embeddings=False: [N, 512] ligand+environment
                - When use_dual_embeddings=True: [N, 1024] stacked [ligand_only, full]
            edge_index: [2, E] directed edge index.

        Returns:
            [E, in_dim] edge input tensor (in_dim = 1024 either way).
        """
        src = edge_index[0]
        dst = edge_index[1]

        if self.use_dual_embeddings:
            # Dual embedding mode: [ligand_only, full]
            # Split the 1024D input into two 512D components
            src_emb = unimol_embeddings[src]  # [E, 1024]
            dst_emb = unimol_embeddings[dst]  # [E, 1024]

            src_ligand = src_emb[:, :512]      # [E, 512] ligand-only
            src_full = src_emb[:, 512:]        # [E, 512] full
            dst_ligand = dst_emb[:, :512]      # [E, 512] ligand-only
            dst_full = dst_emb[:, 512:]        # [E, 512] full

            # Build edge input using:
            # - Antisymmetric component from ligand (captures substituent variation)
            # - Symmetric component from full (captures environment effects)
            diff_ligand = src_ligand - dst_ligand     # [E, 512] - antisymmetric (sub-dependent)
            mean_full = (src_full + dst_full) / 2.0   # [E, 512] - symmetric (environment-dependent)
            return torch.cat([diff_ligand, mean_full], dim=-1)  # [E, 1024]
        else:
            # Standard mode: ligand+environment
            # Symmetric-aware decomposition
            src_emb = unimol_embeddings[src]  # [E, 512]
            dst_emb = unimol_embeddings[dst]  # [E, 512]
            diff = src_emb - dst_emb                        # [E, 512] - antisymmetric
            mean = (src_emb + dst_emb) / 2.0                # [E, 512] - symmetric
            return torch.cat([diff, mean], dim=-1)          # [E, 1024]

    def _forward_edges(self, unimol_embeddings: torch.Tensor,
                       edge_index: torch.LongTensor,
                       edge_type: Optional[torch.Tensor] = None):
        """Compute (mean, log_std) tensors for all edges.

        Args:
            unimol_embeddings: [N, emb_dim] pre-computed Uni-Mol embeddings.
                - When use_dual_embeddings=False: [N, 512] ligand+environment
                - When use_dual_embeddings=True: [N, 1024] stacked [ligand_only, full]
            edge_index: [2, E] directed edge index.
            edge_type: [E] integer relation index. When provided, each edge is
                routed to mlps[edge_type // 2] for gradient isolation.

        Returns:
            mean: [E, mlp_out_dim] softsign-scaled bias coefficient means.
            log_std: [E, mlp_out_dim] clamped log standard deviations.
        """
        edge_input = self._build_edge_input(unimol_embeddings, edge_index)

        # Forward through edge MLP with routing if edge_type is provided
        out = self.edge_mlp(edge_input, edge_type)  # [E, 2*mlp_out_dim]
        if out.dim() == 1:
            out = out.unsqueeze(0)
        
        D = self.mlp_out_dim
        mean = out[:, :D]
        log_std = out[:, D:2*D]
        
        # Scale mean outputs using softsign (avoids saturation zone of tanh)
        mean = F.softsign(mean) * self.scale_factors.unsqueeze(0)
        
        # Clamp log_std: exp(-20) ≈ 2e-9, exp(2.0) ≈ 7.4
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)

        return mean, log_std

    # ------------------------------------------------------------------
    # NeuralLinear + Thompson Sampling API (only meaningful when
    # use_bayesian_heads=True; each method routes every edge to its own
    # bias type's BayesianLinearHead via edge_type // 2, exactly like the
    # deterministic routing above).
    # ------------------------------------------------------------------

    def forward_features(self, unimol_embeddings: torch.Tensor,
                         edge_index: torch.LongTensor,
                         edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return per-edge trunk features ``z = phi(x)``, routed or not.

        Args:
            unimol_embeddings: [N, emb_dim] pre-computed Uni-Mol embeddings.
            edge_index: [2, E] directed edge index.
            edge_type: [E] integer relation index. When given, routes each
                edge to its own bias type's trunk (the pretraining
                fully-connected 4x-edges-per-pair graph structure). When
                ``None`` (the online-training graph structure — one edge per
                pair, every bias type independently meaningful for every
                edge — this is what ``build_graph_and_data`` in
                ``training/workflow.py`` always produces), every edge gets
                features from **every** bias type's trunk.

        Returns:
            Routed: ``[E, 64]``. Unrouted (``edge_type=None``): ``[E, D, 64]``.
        """
        edge_input = self._build_edge_input(unimol_embeddings, edge_index)
        return self.edge_mlp.forward_features(edge_input, edge_type)

    def _preact_to_action(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Map a bias head's raw (pre-activation) output into physical bias-
        coefficient units for dimension ``dim``: ``scale_factors[dim] *
        softsign(x)``.

        This is the exact transform ``_forward_edges`` applies to every
        deterministic head's raw output, and it's why ``BayesianLinearHead``
        is seeded from the deterministic readout's raw (pre-transform)
        weights (see ``pretrain_policy.initialize_bayesian_heads_from_pretrained``)
        — the posterior lives in the *same* pre-activation space, so its
        output needs the same transform applied before it's a valid action.
        Skipping this (as an earlier version of this code did) produces raw,
        effectively-unbounded linear-regression outputs used directly as bias
        coefficients — silently wrong-scale actions that pin combos into
        degenerate, low-transition states.
        """
        return F.softsign(x) * self.scale_factors[dim]

    def _action_to_preact(self, y: torch.Tensor, dim: int, eps: float = 1e-3) -> torch.Tensor:
        """Inverse of ``_preact_to_action``: map an observed physical bias
        coefficient back into the pre-activation space the posterior actually
        models, so ``update_bayesian_posteriors`` regresses toward the same
        quantity ``sample_weights()``/``mu`` predict.

        ``softsign(x) = x / (1 + |x|)`` inverts to ``x = s / (1 - |s|)`` for
        ``s = y / scale`` (clamped away from ``±1`` for numerical stability —
        an observed action at or beyond the scale bound would otherwise
        invert to ``±inf``).
        """
        s = (y / self.scale_factors[dim]).clamp(-1.0 + eps, 1.0 - eps)
        return s / (1.0 - s.abs())

    def predict_uncertainty(self, unimol_embeddings: torch.Tensor,
                            edge_index: torch.LongTensor,
                            edge_type: Optional[torch.Tensor] = None) -> tuple:
        """Posterior predictive mean/variance for every edge's bias head(s).

        Used for uncertainty-driven combo selection: average/max predictive
        variance over a combo's edges ranks how much this combo would teach the
        model if simulated next.

        Returns:
            ``mean`` is in **physical bias-coefficient units**
            (``scale_factors[d] * softsign(raw_mean)``, matching what
            ``_forward_edges``/the analysis script report — see
            ``_preact_to_action``). ``var`` stays in the posterior's native
            pre-activation space (variance doesn't transform simply through a
            nonlinearity); it's only ever used for *relative* combo ranking
            in ``select_combos_by_uncertainty``, where that's sufficient
            (softsign is monotonic, so relative ordering by pre-activation
            uncertainty tracks relative physical-space uncertainty).

            Routed (``edge_type`` given): ``mean [E]``, ``var [E]`` — the
            edge's own routed head's posterior predictive mean/variance.
            Unrouted (``edge_type=None``): ``mean [E, D]``, ``var [E, D]`` —
            every head's prediction for every edge (matches ``_forward_edges``'s
            legacy-path shape).
        """
        z = self.forward_features(unimol_embeddings, edge_index, edge_type)
        D = self.mlp_out_dim

        if edge_type is None:
            E = z.size(0)
            mean = torch.zeros(E, D, device=z.device, dtype=z.dtype)
            var = torch.zeros(E, D, device=z.device, dtype=z.dtype)
            for d in range(D):
                head = self.edge_mlp.mlps[d].bayesian_head
                mean_raw, var[:, d] = head.predict(z[:, d, :])
                mean[:, d] = self._preact_to_action(mean_raw, d)
            return mean, var

        bias_type = edge_type // 2
        E = z.size(0)
        mean = torch.zeros(E, device=z.device, dtype=z.dtype)
        var = torch.zeros(E, device=z.device, dtype=z.dtype)
        for d in range(D):
            mask = bias_type == d
            if not mask.any():
                continue
            head = self.edge_mlp.mlps[d].bayesian_head
            m_d, v_d = head.predict(z[mask])
            mean[mask] = self._preact_to_action(m_d, d)
            var[mask] = v_d
        return mean, var

    def get_actions_thompson(self, unimol_embeddings: torch.Tensor,
                             edge_index: torch.LongTensor,
                             edge_type: Optional[torch.Tensor] = None,
                             deterministic: bool = False) -> tuple:
        """Sample actions via Thompson sampling from each bias head's posterior.

        One weight vector is sampled per bias type per call (i.e. one Thompson
        sample per decision/combo — not per edge), matching how a single
        REINFORCE ``get_actions`` call produces one policy query per combo.

        Args:
            unimol_embeddings: [N, emb_dim] pre-computed Uni-Mol embeddings.
            edge_index: [2, E] directed edge index.
            edge_type: [E] integer relation index. See ``forward_features``
                for the routed vs. unrouted (``None``, the online-training
                default) distinction.
            deterministic: If True, use the posterior mean ``mu`` instead of a
                sample (no exploration — for eval/replay).

        Returns:
            actions: [E, mlp_out_dim] bias coefficients, in the same physical
                units as REINFORCE's ``get_actions``/``_forward_edges``
                (``scale_factors[d] * softsign(raw_sample)`` — see
                ``_preact_to_action``; the posterior itself lives in raw
                pre-activation space, seeded from the deterministic readout's
                pre-transform weights, so this transform is required to turn
                a sample into a valid action). Routed: only the routed column
                per edge is meaningful, others are 0 (same convention as
                ``get_actions``'s routed path). Unrouted: every column is a
                real prediction for every edge.
            mean, var: predictive mean/variance, same routed-vs-unrouted shape
                and mean/var unit convention as ``predict_uncertainty``.
        """
        z = self.forward_features(unimol_embeddings, edge_index, edge_type)
        D = self.mlp_out_dim
        E = z.size(0)
        actions = torch.zeros(E, D, device=z.device, dtype=z.dtype)

        if edge_type is None:
            mean = torch.zeros(E, D, device=z.device, dtype=z.dtype)
            var = torch.zeros(E, D, device=z.device, dtype=z.dtype)
            for d in range(D):
                head = self.edge_mlp.mlps[d].bayesian_head
                z_d = z[:, d, :]
                w = head.mu if deterministic else head.sample_weights()
                actions[:, d] = self._preact_to_action(head.act(z_d, w), d)
                mean_raw, var[:, d] = head.predict(z_d)
                mean[:, d] = self._preact_to_action(mean_raw, d)
        else:
            bias_type = edge_type // 2
            mean = torch.zeros(E, device=z.device, dtype=z.dtype)
            var = torch.zeros(E, device=z.device, dtype=z.dtype)
            for d in range(D):
                mask = bias_type == d
                if not mask.any():
                    continue
                head = self.edge_mlp.mlps[d].bayesian_head
                z_d = z[mask]
                w = head.mu if deterministic else head.sample_weights()
                actions[mask, d] = self._preact_to_action(head.act(z_d, w), d)
                m_d, v_d = head.predict(z_d)
                mean[mask] = self._preact_to_action(m_d, d)
                var[mask] = v_d

        # No extra clamp needed: softsign(x) is strictly within (-1, 1) for
        # any finite x, so _preact_to_action already guarantees actions lie
        # strictly within (-scale_factors[d], scale_factors[d]).
        return actions, mean, var

    def update_bayesian_posteriors(self, unimol_embeddings: torch.Tensor,
                                   edge_index: torch.LongTensor,
                                   edge_type: Optional[torch.Tensor],
                                   targets: torch.Tensor,
                                   weights: Optional[torch.Tensor] = None) -> None:
        """Fold observed ``(z, target)`` pairs into each bias head's posterior.

        No gradient, no optimizer — this is the closed-form replacement for
        ``policy_loss.backward(); optimizer.step()``.

        Args:
            unimol_embeddings: [N, emb_dim] pre-computed Uni-Mol embeddings.
            edge_index: [2, E] directed edge index.
            edge_type: [E] integer relation index. See ``forward_features``
                for the routed vs. unrouted (``None``, the online-training
                default) distinction. Unrouted: every edge updates every head
                (using that head's own target column) — no masking, since
                every column is independently meaningful for every edge.
            targets: [E] or [E, mlp_out_dim] regression targets — the observed
                action (bias coefficient, in physical units) each edge
                actually submitted. Internally inverse-transformed
                (``_action_to_preact``) into the posterior's native
                pre-activation space before regressing — the posterior was
                seeded from the deterministic readout's pre-transform
                weights (see
                ``pretrain_policy.initialize_bayesian_heads_from_pretrained``)
                and ``get_actions_thompson`` samples in that same space, so
                the update must too, or the posterior mean drifts away from
                what's actually being sampled/acted on.
            weights: Optional per-edge confidence weight for the update (e.g.
                reward-derived AWR weight combined with the existing
                DDG-transition confidence weight). Either [E] (same weight
                reused for every head — the only sensible shape when
                ``edge_type`` routes each edge to one head anyway) or
                [E, mlp_out_dim] (a genuinely per-head weight per edge — e.g.
                each bias type has its own AWR weight since
                ``compute_pair_reward`` gives each dimension its own reward).
                Defaults to all-ones.
        """
        z = self.forward_features(unimol_embeddings, edge_index, edge_type)
        D = self.mlp_out_dim
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1).expand(-1, D)

        def _weights_for_dim(d: int) -> Optional[torch.Tensor]:
            if weights is None:
                return None
            return weights if weights.dim() == 1 else weights[:, d]

        if edge_type is None:
            for d in range(D):
                head = self.edge_mlp.mlps[d].bayesian_head
                r_d_raw = self._action_to_preact(targets[:, d], d)
                head.update(z[:, d, :], r_d_raw, weights=_weights_for_dim(d))
            return

        bias_type = edge_type // 2
        for d in range(D):
            mask = bias_type == d
            if not mask.any():
                continue
            head = self.edge_mlp.mlps[d].bayesian_head
            z_d = z[mask]
            r_d_raw = self._action_to_preact(targets[mask, d], d)
            w_all = _weights_for_dim(d)
            w_d = w_all[mask] if w_all is not None else None
            head.update(z_d, r_d_raw, weights=w_d)

    def get_actions(self, unimol_embeddings: torch.Tensor,
                    edge_index: torch.LongTensor,
                    edge_type: Optional[torch.Tensor] = None,
                    deterministic: bool = False):
        """Sample actions and log-probabilities for every directed edge.
        
        Args:
            unimol_embeddings: [N, unimol_dim] pre-computed Uni-Mol embeddings.
            edge_index: [2, E] directed edge index.
            edge_type: [E] integer relation index (optional).
            deterministic: If True, returns mean actions with zero logp.
        
        Returns:
            actions: [E, mlp_out_dim] sampled bias coefficients.
            logp: [E] per-edge sum log-probability.
            mean: [E, mlp_out_dim] distribution means.
            log_std: [E, mlp_out_dim] distribution log standard deviations.
        """
        mean, log_std = self._forward_edges(unimol_embeddings, edge_index, edge_type)
        
        if deterministic:
            actions = mean
            logp = torch.zeros(mean.size(0), device=mean.device)
        else:
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            actions = dist.rsample()  # [E, D]
            logp = dist.log_prob(actions)  # [E, D]
            
            # Routing: zero out non-relevant dimensions if edge_type provided
            if edge_type is not None:
                rel_mask = torch.zeros_like(logp, dtype=torch.bool)
                bias_type = edge_type // 2  # [E], values 0..D-1
                for d in range(self.mlp_out_dim):
                    rel_mask[bias_type == d, d] = True
                logp = logp * rel_mask.float()
            
            logp = logp.sum(dim=-1)  # [E] sum over output dims
        
        return actions, logp, mean, log_std
    
    def evaluate_logp(self, unimol_embeddings: torch.Tensor,
                      edge_index: torch.LongTensor,
                      edge_type: Optional[torch.Tensor] = None,
                      saved_actions: Optional[torch.Tensor] = None):
        """Evaluate log-probability of saved actions.
        
        Args:
            unimol_embeddings: [N, unimol_dim] pre-computed Uni-Mol embeddings.
            edge_index: [2, E] directed edge index.
            edge_type: [E] integer relation index (optional).
            saved_actions: [E] or [E, D] actions to evaluate.
        
        Returns:
            logp: [E] scalar log-prob per edge summed over output dims.
            log_std: [E, D] current log_std (for entropy regularisation).
        """
        mean, log_std = self._forward_edges(unimol_embeddings, edge_index, edge_type)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        
        if saved_actions.dim() == 1:
            saved_actions = saved_actions.unsqueeze(-1)
        
        logp = dist.log_prob(saved_actions)  # [E, D]
        
        # Routing: zero out non-relevant dimensions if edge_type provided
        if edge_type is not None:
            rel_mask = torch.zeros_like(logp, dtype=torch.bool)
            bias_type = edge_type // 2  # [E], values 0..D-1
            for d in range(self.mlp_out_dim):
                rel_mask[bias_type == d, d] = True
            logp = logp * rel_mask.float()
        
        return logp, log_std


# class SitePoolMLPPolicy(nn.Module):
#     """Direct MLP policy: no RGCN, site-conditioned pooling as system context.

#     Replaces the RGCN encoder with a simple site-level sum-pool of the frozen
#     AtomBondGNN P1 embeddings.  For each directed edge (A → B) the input is:

#         concat(P1_A[p1_dim], P1_B[p1_dim], site_pool_A[p1_dim])  = 3*p1_dim D

#     where ``site_pool_A`` is the sum of all P1 embeddings at site A.  This
#     gives the edge MLP a lightweight system-context signal without message
#     passing over the perturbation-network graph topology.
#     ``edge_attr`` (the one-hot relation type) is no longer part of the input:
#     each ``BiasHeadMLP`` is routed only its own edge type (``edge_type // 2``),
#     so there is no ambiguity about which bias type an edge represents.

#     During training a *block dropout* mask is applied to the entire site_pool
#     slice with probability ``context_dropout_p``.  This forces the trunk to
#     learn a pairwise-sufficient predictor that degrades gracefully to zero
#     context, preventing over-reliance on the site-context signal.

#     The ``edge_mlp`` (EdgeValueMLP) and ``scale_factors`` are identical to
#     ``EdgePolicy``, so the same BC pretraining and REINFORCE update logic apply.

#     Args:
#         p1_dim: Dimension of AtomBondGNN node embeddings (default: 64).
#         edge_attr_dim: Dimension of per-edge one-hot relation features (default: 8).
#         mlp_hidden: Hidden size for EdgeValueMLP trunk and heads (default: 64).
#         mlp_out_dim: Number of bias types / output coefficients (default: 4).
#         context_dropout_p: Probability of zeroing the entire site_pool block per
#             edge during training (block dropout).  Default: 0.3.
#     """

#     def __init__(self, p1_dim: int = 64, edge_attr_dim: int = 8,
#                  mlp_hidden: int = 64, mlp_out_dim: int = 4,
#                  context_dropout_p: float = 0.3):
#         super().__init__()
#         self.p1_dim = int(p1_dim)
#         self.mlp_out_dim = int(mlp_out_dim)
#         self.context_dropout_p = float(context_dropout_p)
#         # Edge input: [P1_src, P1_dst, site_pool_src]  (no edge_attr).
#         # Each BiasHeadMLP is routed only its own edges via edge_type, so the
#         # one-hot relation type is redundant as an input feature.
#         # edge_attr_dim is kept for API compatibility but does not affect in_dim.
#         in_dim = 3 * p1_dim
#         self.edge_mlp = EdgeValueMLP(in_dim, mlp_hidden, num_bias_types=self.mlp_out_dim)
#         self.register_buffer(
#             'scale_factors',
#             torch.tensor([305.0, 520.0, 85.0, 30.0]),
#             persistent=False,
#         )

#     # ------------------------------------------------------------------
#     # Site pooling
#     # ------------------------------------------------------------------

#     # def _site_pool(self, x: torch.Tensor, site_index: torch.Tensor) -> torch.Tensor:
#     #     """Compute per-site mean of P1 embeddings, then expand back to node dim.

#     #     Args:
#     #         x: [N, p1_dim] node (substituent) embeddings.
#     #         site_index: [N] 0-indexed site assignment for each node.

#     #     Returns:
#     #         [N, p1_dim]: each node replaced by its site's mean embedding.
#     #     """
#     #     num_sites = int(site_index.max().item()) + 1
#     #     pool = torch.zeros(num_sites, x.size(1), dtype=x.dtype, device=x.device)
#     #     count = torch.zeros(num_sites, dtype=x.dtype, device=x.device)
#     #     pool.scatter_add_(0, site_index.unsqueeze(1).expand_as(x), x)
#     #     count.scatter_add_(0, site_index,
#     #                        torch.ones(x.size(0), dtype=x.dtype, device=x.device))
#     #     pool = pool / count.unsqueeze(1).clamp(min=1.0)
#     #     return pool[site_index]  # [N, p1_dim]

#     def _site_pool(self, x: torch.Tensor, site_index: torch.Tensor) -> torch.Tensor:
#         """Compute per-site sum of P1 embeddings, then expand back to node dim.

#         Args:
#             x: [N, p1_dim] node (substituent) embeddings.
#             site_index: [N] 0-indexed site assignment for each node.

#         Returns:
#             [N, p1_dim]: each node replaced by its site's sum embedding.
#         """
#         num_sites = int(site_index.max().item()) + 1
#         pool = torch.zeros(num_sites, x.size(1), dtype=x.dtype, device=x.device)

#         pool.scatter_add_(0, site_index.unsqueeze(1).expand_as(x), x)

#         return pool[site_index]  # [N, p1_dim]

#     # ------------------------------------------------------------------
#     # Edge input construction
#     # ------------------------------------------------------------------

#     def _build_edge_inputs(self, x: torch.Tensor, edge_index: torch.LongTensor,
#                            edge_attr: Optional[torch.Tensor] = None,
#                            site_index: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """Build [E, 3*p1_dim] edge input tensor.

#         concat(P1_src, P1_dst, site_pool_src)
#         edge_attr is accepted for API compatibility but is no longer used;
#         directionality (fwd vs bwd) is already encoded by the src/dst P1 swap.
#         During training, the entire site_pool block is zeroed with probability
#         context_dropout_p (block dropout) to prevent over-reliance on context.
#         """
#         site_pool = self._site_pool(x, site_index)  # [N, p1_dim]
#         src, dst = edge_index[0], edge_index[1]
#         ctx = site_pool[src]  # [E, p1_dim]
#         # Block dropout: zero the entire context slice per edge during training
#         if self.training and self.context_dropout_p > 0.0:
#             # [E, 1] Bernoulli keep-mask broadcast over p1_dim
#             keep = (torch.rand(ctx.size(0), 1, device=ctx.device)
#                     >= self.context_dropout_p).float()
#             # Rescale to preserve expected value (inverted dropout)
#             ctx = ctx * keep / (1.0 - self.context_dropout_p)
#         return torch.cat([x[src], x[dst], ctx], dim=-1)  # [E, 3*p1_dim]

#     # ------------------------------------------------------------------
#     # Forward helpers
#     # ------------------------------------------------------------------

#     def _forward_edges(self, x: torch.Tensor, edge_index: torch.LongTensor,
#                        edge_attr: Optional[torch.Tensor] = None,
#                        site_index: Optional[torch.Tensor] = None,
#                        edge_type: Optional[torch.Tensor] = None):
#         """Compute (mean, log_std) tensors for all edges.

#         Args:
#             edge_type: [E] integer relation index.  When provided, each edge is
#                 routed to ``mlps[edge_type // 2]`` so only the relevant MLP
#                 computes output for that edge.

#         Returns:
#             mean: [E, mlp_out_dim] softsign-scaled bias coefficient means.
#             log_std: [E, mlp_out_dim] clamped log standard deviations.
#         """
#         inp = self._build_edge_inputs(x, edge_index, edge_attr, site_index)
#         out = self.edge_mlp(inp, edge_type)  # [E, 2*D]
#         if out.dim() == 1:
#             out = out.unsqueeze(0)
#         D = self.mlp_out_dim
#         mean = F.softsign(out[:, :D]) * self.scale_factors.unsqueeze(0)
#         log_std = torch.clamp(out[:, D:], min=-20.0, max=2.0)
#         return mean, log_std

#     # ------------------------------------------------------------------
#     # Public API (mirrors EdgePolicy)
#     # ------------------------------------------------------------------

#     def get_actions(self, x, edge_index, edge_type=None, edge_attr=None,
#                     site_index=None, deterministic: bool = False):
#         """Sample actions and log-probabilities for every directed edge.

#         Args:
#             x: [N, p1_dim] frozen AtomBondGNN node embeddings.
#             edge_index: [2, E] directed edge index.
#             edge_type: unused (kept for API compatibility with EdgePolicy callers).
#             edge_attr: [E, edge_attr_dim] one-hot relation features.
#             site_index: [N] 0-indexed site assignment per node.
#             deterministic: if True, returns mean actions with zero logp.

#         Returns:
#             actions: [E, D] sampled bias coefficients.
#             logp: [E] per-edge sum log-probability.
#             mean: [E, D] distribution means.
#             log_std: [E, D] distribution log standard deviations.
#         """
#         mean, log_std = self._forward_edges(x, edge_index, edge_attr, site_index, edge_type)
#         if deterministic:
#             actions = mean
#             logp = torch.zeros(mean.size(0), device=mean.device)
#         else:
#             std = torch.exp(log_std)
#             dist = torch.distributions.Normal(mean, std)
#             actions = dist.rsample()
#             clip_limits = self.scale_factors * 1.05
#             actions = torch.clamp(actions,
#                                   -clip_limits.unsqueeze(0), clip_limits.unsqueeze(0))
#             logp = dist.log_prob(actions).sum(dim=-1)
#         return actions, logp, mean, log_std

#     def evaluate_logp(self, x, edge_index, edge_type=None, edge_attr=None,
#                       site_index=None, saved_actions=None):
#         """Evaluate log π_θ(saved_actions | x) under current policy parameters.

#         Args:
#             saved_actions: [E] or [E, D] actions from the simulation run.

#         Returns:
#             logp: [E] scalar log-prob per edge summed over output dims.
#             log_std: [E, D] current log_std (for entropy regularisation).
#         """
#         mean, log_std = self._forward_edges(x, edge_index, edge_attr, site_index, edge_type)
#         std = torch.exp(log_std)
#         dist = torch.distributions.Normal(mean, std)
#         a = saved_actions.detach().to(mean.device)
#         if a.dim() == 1:
#             a = a.unsqueeze(-1)
#         # Per-dimension log-probs [E, D] for per-head REINFORCE weighting.
#         logp = dist.log_prob(a)  # [E, D]
#         # When routing is active, zero out non-relevant dimensions so that only
#         # the MLP that actually produced the action for each edge contributes
#         # a gradient.  This eliminates cross-type reward contamination.
#         if edge_type is not None:
#             rel_mask = torch.zeros_like(logp, dtype=torch.bool)
#             bias_type = edge_type // 2  # [E], values 0..D-1
#             for d in range(self.mlp_out_dim):
#                 rel_mask[bias_type == d, d] = True
#             logp = logp * rel_mask.float()
#         return logp, log_std
