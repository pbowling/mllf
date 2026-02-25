"""
Training script for DeepSet autoencoder pretraining.

This implements Step 3 of the 4-step pretraining process:
Train the autoencoder using MSE loss between input and reconstruction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import time
from typing import Optional, Dict

from mllf.cb.deepset_autoencoder import create_autoencoder


class AtomFeatureDataset(Dataset):
    """PyTorch Dataset for atom features."""
    
    def __init__(self, data_path):
        """
        Args:
            data_path: Path to .pt file containing training data
        """
        data = torch.load(data_path, map_location='cpu')
        self.features = data['features']  # [num_atoms, feature_dim]
        self.system_name = data['system_name']
        self.num_atoms = len(self.features)
        self.feature_dim = data['feature_dim']
        
        print(f"Loaded {self.system_name}: {self.num_atoms:,} atoms, {self.feature_dim}D features")
    
    def __len__(self):
        return self.num_atoms
    
    def __getitem__(self, idx):
        return self.features[idx]


def train_autoencoder(
    train_data_path: Path,
    output_dir: Path,
    input_dim: int = 2289,
    hidden_dim: int = 256,
    embedding_dim: int = 64,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    patience: int = 10,
    device: str = 'cpu',
    save_interval: int = 10,
) -> Dict:
    """Train a DeepSet autoencoder on atom features.
    
    Args:
        train_data_path: Path to training data .pt file
        output_dir: Directory to save checkpoints and final model
        input_dim: Input feature dimension (AEV + charge)
        hidden_dim: Hidden layer dimension
        embedding_dim: Embedding/bottleneck dimension
        batch_size: Training batch size
        learning_rate: Adam learning rate
        num_epochs: Maximum number of epochs
        patience: Early stopping patience (epochs without improvement)
        device: 'cpu' or 'cuda'
        save_interval: Save checkpoint every N epochs
        
    Returns:
        dict with training statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading training data from {train_data_path}...")
    dataset = AtomFeatureDataset(train_data_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=(device == 'cuda')
    )
    
    # Create model
    print(f"\nCreating autoencoder:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Embedding dim: {embedding_dim}")
    
    model = create_autoencoder(input_dim, hidden_dim, embedding_dim)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Device: {device}")
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    training_history = {
        'epoch_losses': [],
        'best_loss': None,
        'best_epoch': None,
        'total_time': 0,
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_features in dataloader:
            batch_features = batch_features.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch_features)
            
            # Compute loss
            loss = criterion(output['reconstruction'], batch_features)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for epoch
        avg_loss = epoch_loss / num_batches
        training_history['epoch_losses'].append(avg_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")
        
        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            training_history['best_loss'] = best_loss
            training_history['best_epoch'] = epoch + 1
            epochs_without_improvement = 0
            
            # Save best model
            best_model_path = output_dir / 'best_encoder.pt'
            model.save_encoder(best_model_path)
            print(f"  → New best model saved (loss: {best_loss:.6f})")
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path.name}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs")
            break
    
    total_time = time.time() - start_time
    training_history['total_time'] = total_time
    
    # Save final encoder
    final_encoder_path = output_dir / 'final_encoder.pt'
    model.save_encoder(final_encoder_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Best loss: {best_loss:.6f} (epoch {training_history['best_epoch']})")
    print(f"  Final loss: {avg_loss:.6f}")
    print(f"  Best encoder saved to: {output_dir / 'best_encoder.pt'}")
    print(f"  Final encoder saved to: {final_encoder_path}")
    print(f"{'='*70}\n")
    
    return training_history


def train_all_systems(
    data_root: Path,
    output_root: Path,
    **training_kwargs
) -> Dict[str, Dict]:
    """Train autoencoders for all available pretraining systems.
    
    Args:
        data_root: Directory containing *_training_data.pt files
        output_root: Root directory for output (creates subdirs per system)
        **training_kwargs: Additional arguments passed to train_autoencoder
        
    Returns:
        dict mapping system_name -> training statistics
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    # Find all training data files
    data_files = sorted(data_root.glob('*_training_data.pt'))
    
    if not data_files:
        raise ValueError(f"No training data files found in {data_root}")
    
    print(f"Found {len(data_files)} systems to train")
    
    all_results = {}
    
    for data_file in data_files:
        system_name = data_file.stem.replace('_training_data', '')
        output_dir = output_root / system_name
        
        print(f"\n{'='*70}")
        print(f"TRAINING SYSTEM: {system_name}")
        print(f"{'='*70}")
        
        try:
            results = train_autoencoder(
                train_data_path=data_file,
                output_dir=output_dir,
                **training_kwargs
            )
            all_results[system_name] = results
        except Exception as e:
            print(f"ERROR: Failed to train {system_name}: {e}")
            all_results[system_name] = {'error': str(e)}
            continue
    
    # Save summary
    summary_path = output_root / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ALL TRAINING COMPLETED")
    print(f"  Successfully trained: {sum(1 for r in all_results.values() if 'error' not in r)}/{len(data_files)}")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DeepSet autoencoder')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data .pt file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for trained models')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--all', action='store_true',
                        help='Train all systems in data directory')
    
    args = parser.parse_args()
    
    data_path = Path(args.data)
    output_path = Path(args.output)
    
    training_args = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_epochs': args.epochs,
        'patience': args.patience,
        'device': args.device,
    }
    
    if args.all:
        # Train all systems
        train_all_systems(data_path, output_path, **training_args)
    else:
        # Train single system
        train_autoencoder(data_path, output_path, **training_args)
