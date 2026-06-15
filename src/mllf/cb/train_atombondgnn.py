"""Training functions for AtomBondGNN and DeepSet autoencoder pretraining.

Public API
----------
supervised_nt_xent_loss   — Supervised NT-Xent contrastive loss.
load_bond_datasets        — Load per-substituent bond-topology datasets.
load_flat_datasets        — Load flat atom-feature tensors (DeepSet mode).
train_atombondgnn         — AtomBondGNN autoencoder training loop.
train_deepset             — DeepSet (flat MLP) autoencoder training loop.
"""

import json
import math
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mllf.cb.deepset_autoencoder import AtomBondGNNAutoencoder, create_autoencoder


# ============================================================================
# Contrastive loss
# ============================================================================

def supervised_nt_xent_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised NT-Xent contrastive loss (Khosla et al., 2020).

    For each anchor all samples sharing its label are treated as positives;
    all others are negatives.  Anchors with no positive partner in the batch
    contribute zero loss (are excluded from the mean).

    Args:
        embeddings: [B, D] — raw (unnormalized) substituent embeddings.
        labels:     [B]    — integer class labels.
        temperature: Scaling temperature τ (default: 0.07).

    Returns:
        Scalar loss tensor (differentiable).
    """
    B = embeddings.size(0)
    if B < 2:
        return embeddings.new_zeros(())

    z = F.normalize(embeddings, dim=-1)               # [B, D] unit sphere
    sim = torch.mm(z, z.t()) / temperature             # [B, B]

    # Masks
    eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye  # [B, B]

    has_pos = pos_mask.any(dim=1)                      # anchors with ≥1 positive
    if not has_pos.any():
        return embeddings.new_zeros(())

    # Numerically stable log-softmax denominator (exclude self)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * ~eye                    # zero out diagonal
    log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)  # [B]

    # Log-probability for each positive pair
    log_prob = sim - log_denom.unsqueeze(1)            # [B, B]

    # Average over positives per anchor; average over anchors that have positives
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)  # [B]
    loss_per = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos  # [B]
    return loss_per[has_pos].mean()


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Hypersphere uniformity loss (Wang & Isola, 2020).

    L_unif = log mean_{i≠j} exp(-t · ‖z_i − z_j‖²)

    Minimising this encourages embeddings to spread uniformly over the unit
    sphere.  z should be L2-normalised before calling.

    Args:
        z: [B, D] unit-normalised embeddings.
        t: Bandwidth parameter (default 2.0, as in the original paper).

    Returns:
        Scalar loss (lower = more uniform).
    """
    sq_pdist = torch.pdist(z, p=2).pow(2)   # upper-triangle pairwise squared dists
    return sq_pdist.mul(-t).exp().mean().log()


def make_stratified_batches(
    idx_list: List[int],
    contrastive_labels: List[int],
    batch_size: int,
    rng: random.Random,
) -> List[List[int]]:
    """Round-robin batches interleaved across contrastive labels.

    Each label contributes at most one sample per batch-slot, so a batch of
    *batch_size* samples drawn from more than *batch_size* distinct labels will
    have all-distinct labels.  Labels with multiple samples (e.g. the same
    substituent in solvent + vacuum + protein datasets) are spread across
    different batches rather than co-located, preventing artificial inflation
    of within-label positive pairs.

    Algorithm: maintain a deque of per-label queues; cycle round-robin,
    popping one sample from the front of each queue and re-appending the
    queue to the back if it still has items.  O(N) time and memory.

    Args:
        idx_list:           Indices into the global substituent list.
        contrastive_labels: Parallel integer labels (same length as substituents).
        batch_size:         Target number of samples per returned batch.
        rng:                Seeded random.Random instance for reproducibility.

    Returns:
        List of batches; each batch is a list of indices.  The last batch may
        be smaller than *batch_size*.
    """
    from collections import defaultdict, deque as _deque

    # Group indices by label and shuffle within each group.
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i in idx_list:
        buckets[contrastive_labels[i]].append(i)

    shuffled_queues = []
    for idxs in buckets.values():
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        shuffled_queues.append(_deque(shuffled))
    rng.shuffle(shuffled_queues)
    active = _deque(shuffled_queues)

    batches: List[List[int]] = []
    current: List[int] = []

    while active:
        q = active.popleft()
        current.append(q.popleft())
        if q:                        # queue still has items — return to back
            active.append(q)
        if len(current) == batch_size:
            batches.append(current)
            current = []

    if current:
        batches.append(current)

    return batches


# ============================================================================
# Dataset loaders
# ============================================================================

def load_bond_datasets(dataset_dir: Path) -> List[Dict]:
    """Load all per-substituent bond-topology datasets from *dataset_dir*.

    Returns a flat list of substituent dicts (each with keys
    ``aev``, ``charges``, ``atom_ids``, ``bond_edge_index``, ``bond_edge_attr``).
    """
    dataset_dir = Path(dataset_dir)
    all_substituents: List[Dict] = []
    pt_files = sorted(dataset_dir.glob('*_training_data.pt'))
    if not pt_files:
        raise FileNotFoundError(f"No *_training_data.pt files found in {dataset_dir}")

    print(f"Loading bond datasets from {dataset_dir} ({len(pt_files)} files) …")
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False, map_location='cpu')
        if data.get('dataset_type') != 'bond_topology':
            warnings.warn(
                f"{pt_file.name} is not a bond_topology dataset "
                f"(dataset_type={data.get('dataset_type')!r}). Skipping."
            )
            continue
        all_substituents.extend(data['substituents'])

    print(f"  Total substituents loaded: {len(all_substituents):,}")
    return all_substituents


def load_flat_datasets(dataset_dir: Path) -> torch.Tensor:
    """Load all flat atom tensors from *dataset_dir* and concatenate.

    Returns a single [total_atoms, feature_dim] tensor.
    """
    dataset_dir = Path(dataset_dir)
    tensors = []
    pt_files = sorted(dataset_dir.glob('*_training_data.pt'))
    if not pt_files:
        raise FileNotFoundError(f"No *_training_data.pt files found in {dataset_dir}")

    print(f"Loading flat datasets from {dataset_dir} ({len(pt_files)} files) …")
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False, map_location='cpu')
        if 'features' not in data:
            warnings.warn(f"{pt_file.name} has no 'features' tensor. Skipping.")
            continue
        tensors.append(data['features'])

    combined = torch.cat(tensors, dim=0)
    print(f"  Total atoms loaded: {combined.shape[0]:,}, feature dim: {combined.shape[1]}")
    return combined


# ============================================================================
# Training loops
# ============================================================================

def train_atombondgnn(
    substituents: List[Dict],
    output_dir: Path,
    aev_length: int = 2288,
    num_atom_types: int = 11,
    embedding_dim: int = 64,
    hidden_dim: int = 256,
    epochs: int = 300,
    patience: int = 25,
    lr: float = 1e-3,
    checkpoint_every: int = 10,
    val_fraction: float = 0.1,
    contrastive_batch_size: int = 32,
    contrastive_alpha: float = 0.5,
    contrastive_temperature: float = 0.07,
    uniformity_gamma: float = 0.0,
    device_str: str = 'cpu',
    seed: int = 42,
) -> None:
    """Train AtomBondGNNAutoencoder with reconstruction + supervised contrastive loss.

    Training objective (per mini-batch of *contrastive_batch_size* substituents):

      loss = MSE_reconstruction  +  alpha * NT-Xent_contrastive

    The contrastive signal uses the **full-ligand** atom-type composition as the
    similarity label: two substituents are *positive pairs* when the CGenFF
    atom-type integer multisets of their entire full-ligand graphs (focus sub +
    core + ref-subs at other sites) are identical.

    Concretely, for sub_i: label_key = Counter(atom_ids.tolist()).

    This achieves two goals simultaneously:
      - Invariant to sub/core boundary: sub + core = total, so the label is the
        same regardless of how the prep splits atoms between sub and core.
      - Ligand-context sensitivity: same substituent chemistry in a different
        ligand (e.g. H in 14benz vs H in dnaligase) gets a *different* label
        because the total compositions differ → pushed apart by contrastive loss.
      - Cross-system equivalence: 14benz and 12benz with the same total
        composition for a given sub get the *same* label → pulled together.
    Positive pairs in practice are the same (ligand, sub) in solvent vs vacuum
    datasets, which are both loaded during training.

    Training batches are constructed with :func:`make_stratified_batches`, which
    round-robins across contrastive labels so each label contributes at most one
    sample per batch slot.  This prevents systems with few substituents from
    being over-represented and ensures inter-system negatives appear in every
    batch.

    An optional uniformity term (Wang & Isola, 2020) can be added via
    *uniformity_gamma* to directly encourage spreading on the hypersphere:

      loss = MSE_reconstruction  +  alpha * NT-Xent  +  gamma * L_unif

    Args:
        substituents: Flat list of substituent dicts with keys ``aev``,
            ``charges``, ``atom_ids``, ``bond_edge_index``, ``bond_edge_attr``,
            ``sub_mask``, ``n_sub``, and ``distinct_atom_types``.
        output_dir: Directory for checkpoints and ``training_history.json``.
        aev_length: AEV feature dimension.
        num_atom_types: Number of element species.
        embedding_dim: Substituent embedding bottleneck dimension.
        hidden_dim: GINConv hidden dimension.
        epochs: Maximum number of training epochs.
        patience: Early-stop if validation loss does not improve.
        lr: Adam learning rate.
        checkpoint_every: Save periodic checkpoint interval.
        val_fraction: Fraction of substituents held out for validation.
        contrastive_batch_size: Number of substituents per mini-batch gradient step.
        contrastive_alpha: Weight of contrastive loss relative to reconstruction.
        contrastive_temperature: NT-Xent temperature τ.
        uniformity_gamma: Weight of uniformity loss (0.0 disables it).
        device_str: Compute device.
        seed: Random seed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # ── Build initial contrastive labels from full-ligand atom-type composition ────
    # Label = sorted tuple of (CGenFF_atom_type_int, count) pairs for ALL atoms
    # in the full-ligand graph: focus sub + core + ref-subs at other sites.
    #
    # Using the TOTAL composition (not background-only or sub-only) gives two
    # critical invariances:
    #
    # 1. Sub/core boundary invariance: the sub/core partition is arbitrary and
    #    can differ between systems (e.g. the same ring fragment might be counted
    #    as "core" in one prep and "sub" in another).  Since sub + core = total,
    #    the total type multiset is fixed regardless of where the boundary is
    #    drawn.  Background-only (core) labels would differ if the boundary
    #    shifts.
    #
    # 2. Cross-system equivalence: 14benz and 12benz may have different core
    #    PDB files, but if sub2 + 14benz_core == sub2 + 12benz_core in atom-type
    #    composition, they map to the same label and are pulled together.
    #    Conversely, H in 14benz vs H in dnaligase have different total
    #    compositions (different scaffolds) and are pushed apart.
    #
    # Positive pairs in practice: the same physical (ligand, substituent)
    # appearing in multiple datasets, most commonly the solvent and vacuum
    # versions of the same system, which are both loaded during training.

    def _build_contrastive_labels(epoch_seed: int = 0) -> List[int]:
        """Build contrastive labels from element-only atom IDs (11 types).
        
        Uses full-ligand atom-type composition multiset (counting each element).
        Element-only IDs (not CGenFF types) are used to group custom atom types
        (e.g., C000, CA00 from legacy RTFs) into standard elements (e.g., C).
        This reduces label cardinality and prevents extreme separation of rare types.
        
        CGenFF types are too specific—custom types appear in only a few subs,
        making those subs' contrastive labels unique and pushing them far away
        via uniformity loss. Element-only IDs instead create broader labels that
        facilitate positive pairs and prevent outlier separation.
        
        Args:
            epoch_seed: Ignored. Labels are fixed per dataset; no per-epoch randomization.
        
        Returns:
            List of integer labels (one per substituent).
        """
        label_map: Dict = {}
        contrastive_labels_local: List[int] = []
        
        for sub in substituents:
            # Always use element-only atom_ids (0-10 mapping)
            # These group custom atom types into standard elements
            atom_ids_list = sub['atom_ids'].tolist() if hasattr(sub['atom_ids'], 'tolist') else list(sub['atom_ids'])
            
            # Create label from sorted Counter of element IDs
            # Example: (1, 20), (6, 5), (7, 3) for a mix of H, C, N
            key = tuple(sorted(Counter(atom_ids_list).items()))
            
            if key not in label_map:
                label_map[key] = len(label_map)
            contrastive_labels_local.append(label_map[key])
        
        return contrastive_labels_local

    # Build initial labels for validation and logging
    contrastive_labels = _build_contrastive_labels(epoch_seed=0)

    n_labels = len(set(contrastive_labels))
    label_counts: Dict[int, int] = {}
    for lbl in contrastive_labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    n_pos_labels = sum(1 for c in label_counts.values() if c > 1)
    print(f"  Contrastive labels: {n_labels} unique types, "
          f"{n_pos_labels} with ≥2 substituents (positive pairs possible)")

    # ── Train / val split ──────────────────────────────────────────────────
    rng = random.Random(seed)
    indices = list(range(len(substituents)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    print(f"Train: {len(train_idx):,} substituents | Val: {len(val_idx):,} substituents")
    print(f"Mini-batch size: {contrastive_batch_size} | "
          f"Contrastive α={contrastive_alpha} | τ={contrastive_temperature} | "
          f"Uniformity γ={uniformity_gamma}")

    model = AtomBondGNNAutoencoder(
        aev_length=aev_length,
        num_atom_types=num_atom_types,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        include_charge=True,
        include_atom_id=True,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction='mean')

    history: List[Dict] = []
    best_val_loss = math.inf
    patience_counter = 0
    best_epoch = 0

    def _run_epoch(batches: List[List[int]], train: bool):
        """Process pre-built batches; return (mean_recon, mean_cont, mean_unif)."""
        total_recon = 0.0
        total_cont  = 0.0
        total_unif  = 0.0
        n_batches   = 0

        for batch in batches:
            batch_recon = torch.zeros((), device=device)
            embeddings  = []
            lbl_list    = []

            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                for sub_i in batch:
                    sub      = substituents[sub_i]
                    aev      = sub['aev'].to(device)
                    charges  = sub['charges'].to(device)
                    atom_ids = sub['atom_ids'].to(device)
                    bei      = sub['bond_edge_index'].to(device)
                    bea      = sub['bond_edge_attr'].to(device)
                    sub_mask = sub['sub_mask'].to(device) if 'sub_mask' in sub else None
                    n_sub    = int(sub['n_sub']) if 'n_sub' in sub else aev.shape[0]

                    out = model(aev, charges, atom_ids, bei, bea, sub_mask=sub_mask)
                    batch_recon = batch_recon + mse(out['reconstruction'], out['input']) / n_sub
                    embeddings.append(out['embedding'])
                    lbl_list.append(contrastive_labels[sub_i])

                batch_recon = batch_recon / len(batch)

                if len(embeddings) > 1:
                    emb_t  = torch.stack(embeddings)                       # [B, D]
                    lbl_t  = torch.tensor(lbl_list, dtype=torch.long, device=device)
                    cont   = supervised_nt_xent_loss(emb_t, lbl_t, contrastive_temperature)
                    z_norm = F.normalize(emb_t, dim=-1)
                    unif   = uniformity_loss(z_norm) if uniformity_gamma > 0.0 \
                             else torch.zeros((), device=device)
                else:
                    cont = torch.zeros((), device=device)
                    unif = torch.zeros((), device=device)

                combined = (batch_recon
                            + contrastive_alpha * cont
                            + uniformity_gamma  * unif)

            if train:
                optimizer.zero_grad()
                combined.backward()
                optimizer.step()

            total_recon += batch_recon.item()
            total_cont  += cont.item()
            total_unif  += unif.item()
            n_batches   += 1

        return total_recon / n_batches, total_cont / n_batches, total_unif / n_batches

    # Pre-build sequential val batches (no stratification needed for eval).
    val_batches = [
        val_idx[s: s + contrastive_batch_size]
        for s in range(0, len(val_idx), contrastive_batch_size)
    ]

    print(f"\nStarting AtomBondGNN autoencoder training ({epochs} max epochs) …\n")

    for epoch in range(1, epochs + 1):
        # Rebuild stratified train batches each epoch (fresh shuffle).
        train_batches = make_stratified_batches(
            train_idx, contrastive_labels, contrastive_batch_size, rng
        )
        model.train()
        train_recon, train_cont, train_unif = _run_epoch(train_batches, train=True)
        train_loss = (train_recon
                      + contrastive_alpha * train_cont
                      + uniformity_gamma  * train_unif)

        model.eval()
        val_recon, val_cont, val_unif = _run_epoch(val_batches, train=False)
        val_loss = (val_recon
                    + contrastive_alpha * val_cont
                    + uniformity_gamma  * val_unif)

        record = {
            'epoch': epoch,
            'train_loss': train_loss, 'train_recon': train_recon,
            'train_cont': train_cont, 'train_unif':  train_unif,
            'val_loss':   val_loss,   'val_recon':   val_recon,
            'val_cont':   val_cont,   'val_unif':    val_unif,
        }
        history.append(record)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            model.save_encoder(str(output_dir / 'best_encoder.pt'))
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:4d}/{epochs} | "
            f"train={train_loss:.5f} (r={train_recon:.5f} c={train_cont:.5f} u={train_unif:.5f})  "
            f"val={val_loss:.5f} (r={val_recon:.5f} c={val_cont:.5f} u={val_unif:.5f})"
            + (" ★" if improved else f"  [{patience_counter}/{patience}]")
        )

        if epoch % checkpoint_every == 0:
            model.save_encoder(str(output_dir / f'checkpoint_epoch_{epoch}.pt'))

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val loss {best_val_loss:.6f} at epoch {best_epoch})")
            break

    model.save_encoder(str(output_dir / 'final_encoder.pt'))
    hist_path = output_dir / 'training_history.json'
    with open(hist_path, 'w') as fh:
        json.dump(history, fh, indent=2)
    print(f"\nTraining history saved to {hist_path}")
    print(f"Best encoder saved to {output_dir / 'best_encoder.pt'} "
          f"(epoch {best_epoch}, val_loss={best_val_loss:.6f})")


def train_deepset(
    features: torch.Tensor,
    output_dir: Path,
    hidden_dim: int = 256,
    embedding_dim: int = 64,
    epochs: int = 300,
    patience: int = 25,
    batch_size: int = 1024,
    lr: float = 1e-3,
    checkpoint_every: int = 10,
    val_fraction: float = 0.1,
    device_str: str = 'cpu',
    seed: int = 42,
) -> None:
    """Train the original DeepSet (flat MLP) autoencoder on atom feature tensors.

    Args:
        features: [total_atoms, feature_dim] concatenated atom feature tensor.
        output_dir: Directory for checkpoints and training history.
        hidden_dim: Hidden layer dimension.
        embedding_dim: Bottleneck embedding dimension.
        epochs: Maximum epochs.
        patience: Early-stop patience.
        batch_size: Mini-batch size (number of atoms per gradient step).
        lr: Adam learning rate.
        checkpoint_every: Checkpoint save interval.
        val_fraction: Fraction of atoms held out for validation.
        device_str: Target compute device.
        seed: Random seed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    torch.manual_seed(seed)
    n = features.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(n * val_fraction))
    val_feat = features[perm[:n_val]].to(device)
    train_feat = features[perm[n_val:]].to(device)
    print(f"Train atoms: {train_feat.shape[0]:,} | Val atoms: {val_feat.shape[0]:,}")

    input_dim = features.shape[1]
    model = create_autoencoder(input_dim, hidden_dim, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    history: List[Dict] = []
    best_val_loss = math.inf
    patience_counter = 0
    best_epoch = 0

    print(f"\nStarting DeepSet autoencoder training ({epochs} max epochs) …\n")

    for epoch in range(1, epochs + 1):
        model.train()
        perm_e = torch.randperm(train_feat.shape[0])
        train_loss = 0.0
        n_batches = 0
        for start in range(0, train_feat.shape[0], batch_size):
            batch = train_feat[perm_e[start:start + batch_size]]
            out = model(batch)
            loss = mse(out['reconstruction'], batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        model.eval()
        with torch.no_grad():
            val_out = model(val_feat)
            val_loss = mse(val_out['reconstruction'], val_feat).item()

        record = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(record)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            model.save_encoder(str(output_dir / 'best_encoder.pt'))
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:4d}/{epochs} | "
            f"train={train_loss:.6f}  val={val_loss:.6f}"
            + (" (best)" if improved else f"  [patience {patience_counter}/{patience}]")
        )

        if epoch % checkpoint_every == 0:
            model.save_encoder(str(output_dir / f'checkpoint_epoch_{epoch}.pt'))

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val loss {best_val_loss:.6f} at epoch {best_epoch})")
            break

    model.save_encoder(str(output_dir / 'final_encoder.pt'))
    with open(output_dir / 'training_history.json', 'w') as fh:
        json.dump(history, fh, indent=2)
    print(f"Best DeepSet encoder saved (epoch {best_epoch}, val={best_val_loss:.6f})")
