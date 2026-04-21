"""AtomBondGNN / DeepSet autoencoder pretraining script.

Orchestrates the two-step self-supervised pretraining pipeline:

  Step 1 — Dataset generation
      Iterate over all substituent PDB files in every pretraining system
      directory.  For each substituent compute context-aware AEVs (protein /
      solvent / vacuum), extract partial charges and element IDs, and build the
      bond edge-index from the RTF BOND section.  Each system's data is saved
      as a separate ``.pt`` file so steps can be re-run independently.

  Step 2 — Autoencoder training
      Load all generated datasets, train a reconstruction autoencoder to learn
      compressed per-substituent embeddings, then save an encoder-only
      checkpoint compatible with ``AtomBondGNN`` (or ``DeepSetEncoder``).

Saved files (default ``--output-dir deepset_pretraining_output/``):
  bond_datasets/<system>_training_data.pt  — per-substituent bond data
  flat_datasets/<system>_training_data.pt  — flat atom tensors (deepset mode)
  trained_models/best_encoder.pt           — best checkpoint (AtomBondGNN format)
  trained_models/checkpoint_epoch_N.pt     — periodic checkpoints
  trained_models/training_history.json     — loss per epoch

Usage::

    python examples/run_deepset_pretraining.py \\
        --pretraining-dir pretraining \\
        --output-dir pretraining/deepset_pretraining_output \\
        --model-type atombondgnn \\
        --steps all \\
        --epochs 300 --patience 25 --lr 1e-3 --device cpu

    # Skip dataset re-generation if .pt files already exist:
    python examples/run_deepset_pretraining.py --steps train ...
"""

import argparse
import json
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# ── project imports ──────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mllf.cb.deepset_pretraining_dataset import (
    generate_all_pretraining_datasets,
    generate_all_bond_pretraining_datasets,
)
from mllf.cb.deepset_autoencoder import (
    create_autoencoder,
    AtomBondGNNAutoencoder,
)


# ============================================================================
# Training helpers
# ============================================================================

def _load_bond_datasets(dataset_dir: Path) -> List[Dict]:
    """Load all per-substituent bond-topology datasets from *dataset_dir*.

    Returns a flat list of substituent dicts (each with keys
    ``aev``, ``charges``, ``atom_ids``, ``bond_edge_index``, ``bond_edge_attr``).
    """
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


def _load_flat_datasets(dataset_dir: Path) -> torch.Tensor:
    """Load all flat atom tensors from *dataset_dir* and concatenate.

    Returns a single [total_atoms, feature_dim] tensor.
    """
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


# ---------------------------------------------------------------------------
# AtomBondGNN training loop
# ---------------------------------------------------------------------------

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
    device_str: str = 'cpu',
    seed: int = 42,
) -> None:
    """Train AtomBondGNNAutoencoder on bond-topology substituent data.

    Training objective: MSE reconstruction of the full input feature vector
    (AEV + charge + atom-type one-hot) from GINConv per-atom hidden states,
    *before* global pooling.  Equal per-substituent weighting (loss normalised
    by number of atoms in each substituent).

    Args:
        substituents: Flat list of substituent dicts (``aev``, ``charges``,
            ``atom_ids``, ``bond_edge_index``, ``bond_edge_attr``).
        output_dir: Directory for checkpoints and ``training_history.json``.
        aev_length: AEV feature dimension.
        num_atom_types: Number of element species.
        embedding_dim: Substituent embedding bottleneck dimension.
        hidden_dim: GINConv hidden dimension.
        epochs: Maximum number of training epochs.
        patience: Early-stop if validation loss does not improve for this many
            consecutive epochs.
        lr: Adam learning rate.
        checkpoint_every: Save ``checkpoint_epoch_N.pt`` interval.
        val_fraction: Fraction of substituents held out for validation.
        device_str: ``'cpu'``, ``'cuda'``, or ``'cuda:N'``.
        seed: Random seed for train/val split and epoch shuffling.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    rng = random.Random(seed)
    indices = list(range(len(substituents)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    print(f"Train: {len(train_idx):,} substituents | Val: {len(val_idx):,} substituents")

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

    def _run_substituents(idx_list: List[int], train: bool) -> float:
        """Iterate over substituents, accumulate MSE, optionally update."""
        total_loss = 0.0
        if train:
            optimizer.zero_grad()

        for sub_i in idx_list:
            sub = substituents[sub_i]
            aev = sub['aev'].to(device)
            charges = sub['charges'].to(device)
            atom_ids = sub['atom_ids'].to(device)
            bei = sub['bond_edge_index'].to(device)
            bea = sub['bond_edge_attr'].to(device)

            if train:
                out = model(aev, charges, atom_ids, bei, bea)
                loss_sub = mse(out['reconstruction'], out['input'])
                # scale by atom count so each substituent contributes equally
                # to the gradient magnitude regardless of size
                (loss_sub / aev.shape[0]).backward()
                total_loss += loss_sub.item() / aev.shape[0]
            else:
                with torch.no_grad():
                    out = model(aev, charges, atom_ids, bei, bea)
                    loss_sub = mse(out['reconstruction'], out['input'])
                    total_loss += loss_sub.item() / aev.shape[0]

        if train:
            optimizer.step()

        return total_loss / len(idx_list)

    print(f"\nStarting AtomBondGNN autoencoder training ({epochs} max epochs) …\n")

    for epoch in range(1, epochs + 1):
        # Shuffle training order each epoch
        rng.shuffle(train_idx)
        model.train()
        train_loss = _run_substituents(train_idx, train=True)

        model.eval()
        val_loss = _run_substituents(val_idx, train=False)

        record = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(record)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_path = output_dir / 'best_encoder.pt'
            model.save_encoder(str(best_path))
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:4d}/{epochs} | "
            f"train={train_loss:.6f}  val={val_loss:.6f}"
            + (" (best)" if improved else f"  [patience {patience_counter}/{patience}]")
        )

        if epoch % checkpoint_every == 0:
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            model.save_encoder(str(ckpt_path))

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(best val loss {best_val_loss:.6f} at epoch {best_epoch})")
            break

    # Save final encoder
    model.save_encoder(str(output_dir / 'final_encoder.pt'))

    # Persist training history
    hist_path = output_dir / 'training_history.json'
    with open(hist_path, 'w') as fh:
        json.dump(history, fh, indent=2)
    print(f"\nTraining history saved to {hist_path}")
    print(f"Best encoder saved to {output_dir / 'best_encoder.pt'} "
          f"(epoch {best_epoch}, val_loss={best_val_loss:.6f})")


# ---------------------------------------------------------------------------
# DeepSet (flat MLP) training loop
# ---------------------------------------------------------------------------

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


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Pretrain AtomBondGNN or DeepSet autoencoder on MSLD data.'
    )
    p.add_argument(
        '--pretraining-dir', required=True, type=Path,
        help='Root directory containing pretraining system sub-directories.',
    )
    p.add_argument(
        '--output-dir', required=True, type=Path,
        help='Root output directory for datasets and model checkpoints.',
    )
    p.add_argument(
        '--model-type', default='atombondgnn',
        choices=['atombondgnn', 'deepset'],
        help='Encoder architecture to pretrain (default: atombondgnn).',
    )
    p.add_argument(
        '--steps', default='all',
        choices=['all', 'dataset', 'train'],
        help='Which steps to run (default: all).',
    )
    p.add_argument(
        '--epochs', type=int, default=300,
        help='Maximum training epochs (default: 300).',
    )
    p.add_argument(
        '--patience', type=int, default=25,
        help='Early-stopping patience epochs (default: 25).',
    )
    p.add_argument(
        '--batch-size', type=int, default=1024,
        help='Atom mini-batch size for DeepSet training (default: 1024).',
    )
    p.add_argument(
        '--lr', type=float, default=1e-3,
        help='Adam learning rate (default: 1e-3).',
    )
    p.add_argument(
        '--hidden-dim', type=int, default=256,
        help='Hidden layer / GINConv dimension (default: 256).',
    )
    p.add_argument(
        '--embedding-dim', type=int, default=64,
        help='Substituent embedding bottleneck dimension (default: 64).',
    )
    p.add_argument(
        '--aev-cutoff', type=float, default=5.1,
        help='AEV spatial cutoff in Angstroms (default: 5.1).',
    )
    p.add_argument(
        '--device', default='cpu',
        help='Compute device: cpu / cuda / cuda:N (default: cpu).',
    )
    p.add_argument(
        '--skip-systems', nargs='*', default=None,
        help='System directory names to skip during dataset generation.',
    )
    p.add_argument(
        '--verbose', action='store_true',
        help='Print verbose AEV context info for first substituent per system.',
    )
    p.add_argument(
        '--checkpoint-every', type=int, default=10,
        help='Save a checkpoint every N epochs (default: 10).',
    )
    p.add_argument(
        '--val-fraction', type=float, default=0.1,
        help='Fraction of data held out for validation (default: 0.1).',
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42).',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pretraining_root: Path = args.pretraining_dir.resolve()
    output_root: Path = args.output_dir.resolve()

    if not pretraining_root.exists():
        sys.exit(f"ERROR: --pretraining-dir does not exist: {pretraining_root}")

    # Subdirectory layout
    if args.model_type == 'atombondgnn':
        dataset_subdir = output_root / 'bond_datasets'
    else:
        dataset_subdir = output_root / 'flat_datasets'
    models_dir = output_root / 'trained_models'

    skip = set(args.skip_systems or [])
    # Always exclude the output directory itself to avoid recursive processing
    skip.add(output_root.name)
    # Common non-system directories
    skip.update(['14benz_pair_combos', '1_analysis_scripts'])

    # ── Step 1: dataset generation ───────────────────────────────────────────
    if args.steps in ('all', 'dataset'):
        print(f"\n{'='*70}")
        print(f"STEP 1 — Dataset Generation  (model_type={args.model_type})")
        print(f"{'='*70}")
        dataset_subdir.mkdir(parents=True, exist_ok=True)

        if args.model_type == 'atombondgnn':
            generate_all_bond_pretraining_datasets(
                pretraining_root=pretraining_root,
                output_root=dataset_subdir,
                skip_systems=list(skip),
                aev_cutoff=args.aev_cutoff,
                verbose=args.verbose,
            )
        else:
            generate_all_pretraining_datasets(
                pretraining_root=pretraining_root,
                output_root=dataset_subdir,
                skip_systems=list(skip),
                aev_cutoff=args.aev_cutoff,
                verbose=args.verbose,
            )

    # ── Step 2: training ─────────────────────────────────────────────────────
    if args.steps in ('all', 'train'):
        print(f"\n{'='*70}")
        print(f"STEP 2 — Autoencoder Training  (model_type={args.model_type})")
        print(f"{'='*70}")
        models_dir.mkdir(parents=True, exist_ok=True)

        if args.model_type == 'atombondgnn':
            substituents = _load_bond_datasets(dataset_subdir)
            if not substituents:
                sys.exit("ERROR: No substituents loaded. Run --steps dataset first.")

            # Infer aev_length from first substituent
            aev_length = substituents[0]['aev'].shape[1]
            print(f"  aev_length inferred from data: {aev_length}")

            train_atombondgnn(
                substituents=substituents,
                output_dir=models_dir,
                aev_length=aev_length,
                num_atom_types=11,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                checkpoint_every=args.checkpoint_every,
                val_fraction=args.val_fraction,
                device_str=args.device,
                seed=args.seed,
            )

        else:
            features = _load_flat_datasets(dataset_subdir)
            train_deepset(
                features=features,
                output_dir=models_dir,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                checkpoint_every=args.checkpoint_every,
                val_fraction=args.val_fraction,
                device_str=args.device,
                seed=args.seed,
            )

    print(f"\nDone.  Encoder checkpoint: {models_dir / 'best_encoder.pt'}")


if __name__ == '__main__':
    main()
