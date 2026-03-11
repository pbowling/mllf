#!/usr/bin/env python
"""
Complete pipeline for DeepSet autoencoder pretraining.

This orchestrates all 4 steps of the pretraining process:
1. Generate offline datasets (AEV + charges) for all systems
2. Build autoencoder models
3. Train autoencoders with MSE loss
4. Save pretrained encoders for RGCN integration

Usage:
    python run_deepset_pretraining.py --pretraining-dir /path/to/pretraining \\
                                      --output-dir /path/to/output \\
                                      --steps all
"""

import argparse
from pathlib import Path
import sys
import json

from mllf.cb.deepset_pretraining_dataset import generate_all_pretraining_datasets
from mllf.cb.train_deepset_autoencoder import train_combined_model


def run_pipeline(
    pretraining_dir: Path,
    output_dir: Path,
    steps: str = 'all',
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    patience: int = 10,
    device: str = 'cpu',
    aev_cutoff: float = 5.1,
    skip_systems: list = None,
    verbose: bool = False,
):
    """Run the complete pretraining pipeline.
    
    Args:
        pretraining_dir: Root directory with pretraining systems
        output_dir: Root output directory
        steps: Which steps to run ('dataset', 'train', or 'all')
        batch_size: Training batch size
        learning_rate: Adam learning rate
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        device: 'cpu' or 'cuda'
        aev_cutoff: AEV spatial cutoff in Angstroms (default: 5.1, matches ANI-2x radial cutoff)
        skip_systems: List of system names to skip
        verbose: If True, print detailed context information
    """
    pretraining_dir = Path(pretraining_dir)
    output_dir = Path(output_dir)
    
    if not pretraining_dir.exists():
        raise ValueError(f"Pretraining directory does not exist: {pretraining_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Directories for intermediate data
    data_dir = output_dir / 'datasets'
    models_dir = output_dir / 'trained_models'
    
    print(f"{'='*80}")
    print("DEEPSET AUTOENCODER PRETRAINING PIPELINE")
    print(f"{'='*80}")
    print(f"Pretraining directory: {pretraining_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Steps to run: {steps}")
    print(f"AEV cutoff: {aev_cutoff} Å")
    print(f"{'='*80}\n")
    
    # Step 1: Generate datasets
    if steps in ['all', 'dataset']:
        print("\n" + "="*80)
        print("STEP 1: GENERATING OFFLINE DATASETS")
        print("="*80 + "\n")
        
        dataset_stats = generate_all_pretraining_datasets(
            pretraining_root=pretraining_dir,
            output_root=data_dir,
            skip_systems=skip_systems,
            aev_cutoff=aev_cutoff,
            verbose=verbose
        )
        
        # Save dataset statistics
        stats_path = output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        print(f"\nDataset statistics saved to: {stats_path}")
    else:
        print("\nSkipping dataset generation (using existing datasets)")
        if not data_dir.exists():
            raise ValueError(f"Dataset directory not found: {data_dir}")
    
    # Step 2 & 3: Train autoencoders
    if steps in ['all', 'train']:
        print("\n" + "="*80)
        print("STEPS 2 & 3: TRAINING AUTOENCODERS")
        print("="*80 + "\n")
        
        training_results = train_combined_model(
            data_root=data_dir,
            output_dir=models_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            device=device,
        )
        
        print(f"\nTraining complete! Model saved to: {models_dir / 'best_encoder.pt'}")
    else:
        print("\nSkipping training (dataset generation only)")
    
    # Step 4: Summary
    print("\n" + "="*80)
    print("STEP 4: DEPLOYMENT READY")
    print("="*80)
    print("\nPretrained encoders are ready to use!")
    print("\nTo use a pretrained encoder in your RGCN:")
    print("```python")
    print("from mllf.cb.deepset_autoencoder import load_pretrained_deepset")
    print("")
    print("# Load pretrained DeepSet (frozen weights)")
    print("deepset = load_pretrained_deepset(")
    print("    'path/to/trained_models/best_encoder.pt',")
    print("    freeze_weights=True")
    print(")")
    print("")
    print("# Use in graph_utils.build_pyg_graph_from_mllf_graph()")
    print("# or compute_deepset_embedding_for_node() — both already")
    print("# use minimized.pdb context extraction automatically.")
    print("```")
    print("\nNext step:")
    print("  Run CB training with pretrained embeddings (see examples/run_workflow.py)")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run DeepSet autoencoder pretraining pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline (dataset + training)
  python run_deepset_pretraining.py --pretraining-dir /home/pbowling/mllf/pretraining \\
                                    --output-dir /home/pbowling/mllf/pretraining_output \\
                                    --steps all --device cuda
  
  # Generate datasets only
  python run_deepset_pretraining.py --pretraining-dir /home/pbowling/mllf/pretraining \\
                                    --output-dir /home/pbowling/mllf/pretraining_output \\
                                    --steps dataset
  
  # Train only (using existing datasets)
  python run_deepset_pretraining.py --pretraining-dir /home/pbowling/mllf/pretraining \\
                                    --output-dir /home/pbowling/mllf/pretraining_output \\
                                    --steps train --device cuda --epochs 200
        """
    )
    
    parser.add_argument('--pretraining-dir', type=str, required=True,
                        help='Path to pretraining directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for datasets and models')
    parser.add_argument('--steps', type=str, default='all',
                        choices=['all', 'dataset', 'train'],
                        help='Which steps to run (default: all)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Training batch size (default: 1024)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Training device (default: cpu)')
    parser.add_argument('--skip-systems', type=str, nargs='+',
                        default=['14benz_pair_combos', '1_analysis_scripts', 'deepset_pretraining_output'],
                        help='Systems to skip (default: 14benz_pair_combos 1_analysis_scripts deepset_pretraining_output)')
    parser.add_argument('--aev-cutoff', type=float, default=5.1,
                        help='AEV spatial cutoff in Angstroms (default: 5.1, matches ANI-2x radial cutoff)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed AEV context information for each system')
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            pretraining_dir=Path(args.pretraining_dir),
            output_dir=Path(args.output_dir),
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            aev_cutoff=args.aev_cutoff,
            skip_systems=args.skip_systems,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
