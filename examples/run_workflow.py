"""Complete training workflow runner for CB-optimized MSLD simulations.

This script implements the full training loop:
  1. Generate combinations from site/sub fragment files
  2. Split into train/val/test sets
  3. For each epoch:
     a. Build graph from RTF fragments
     b. Sample actions (bias coefficients) from policy
     c. Write variables.py with predicted biases
     d. Submit and run MSLD simulations
     e. Parse simulation outputs (transitions, populations)
     f. Compute reward from simulation metrics
     g. Update policy with REINFORCE
  4. Save checkpoints and track progress

Usage:
  python examples/run_workflow.py [config.yaml]

If no config is provided, uses examples/workflow_sample.yaml by default.
"""
from pathlib import Path
import sys
import yaml
import torch
import numpy as np
from typing import Dict, List

from mllf.file_handling.generate_combinations import create_combination_dirs
from mllf.cli.workflow import (
    split_manifest,
    build_data_and_targets_from_combo,
    write_variables_from_actions,
)
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy
from mllf.cb.train import compute_msld_reward
from mllf.cli.sim import run_simulation_batch


def load_manifest(manifest_path: str) -> List[str]:
    """Load list of combo directories from manifest file."""
    with open(manifest_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def train_epoch(
    encoder: RGCNEncoder,
    policy: EdgePolicy,
    optimizer: torch.optim.Optimizer,
    combos: List[str],
    epoch: int,
    config: Dict,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Run one training epoch over all combos.
    
    Args:
        encoder: RGCN node encoder.
        policy: Edge-level policy network.
        optimizer: Optimizer for policy parameters.
        combos: List of combo directory paths.
        epoch: Current epoch number.
        config: Full config dict with simulation settings.
        device: Device for computation ('cpu' or 'cuda').
    
    Returns:
        Dict with epoch statistics (loss, avg_reward, etc.).
    """
    policy.train()
    encoder.train()
    
    epoch_loss = 0.0
    epoch_rewards = []
    
    for combo_idx, combo_dir in enumerate(combos):
        combo_path = Path(combo_dir)
        
        # Create epoch subdirectory for outputs
        epoch_dir = combo_path / f"run_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        
        print(f"Epoch {epoch}, Combo {combo_idx+1}/{len(combos)}: {combo_path.name}")
        
        # 1. Build graph from RTF fragments
        try:
            data, targets, extras = build_data_and_targets_from_combo(str(combo_path))
            data = data.to(device)
        except Exception as e:
            print(f"  Error building graph: {e}")
            continue
        
        # 2. Sample actions from policy (stochastic)
        actions, logp, mean, log_std = policy.get_actions(
            data.x, data.edge_index, data.edge_type, data.edge_attr,
            deterministic=False
        )
        
        # 3. Write variables.py with sampled biases
        variables_path = epoch_dir / 'variables.py'
        try:
            write_variables_from_actions(
                str(combo_path), data, extras, actions,
                out_name=str(variables_path.relative_to(combo_path))
            )
            print(f"  Wrote variables to {variables_path}")
        except Exception as e:
            print(f"  Error writing variables: {e}")
            continue
        
        # 4. Run simulation (blocking or submit to queue)
        if config.get('run_sims', False):
            # Create epoch-specific run script that uses the correct variables.py
            epoch_run_script = epoch_dir / 'run_epoch.sh'
            slurm_output = epoch_dir / f"{combo_path.name}_ep{epoch}.%j.out"
            run_script_content = f"""#!/bin/bash
#SBATCH --job-name={combo_path.name}_ep{epoch}
#SBATCH --output={slurm_output}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu2080 --gres=gpu:1 
#SBATCH --export=ALL
#SBATCH --time=01:00:00

module load charmm/charmm/c51a1

# Change to combo directory and run with epoch-specific variables
cd {combo_path}
python3 msld_flat.py --vars-file {variables_path.relative_to(combo_path)} --out-dir {epoch_dir.relative_to(combo_path)} > {epoch_dir.relative_to(combo_path)}/output.out 2>&1
"""
            epoch_run_script.write_text(run_script_content)
            epoch_run_script.chmod(0o755)
            
            print(f"  Submitting job via sbatch...")
            import subprocess
            try:
                # Submit the job and capture job ID
                result = subprocess.run(
                    ['sbatch', str(epoch_run_script)],
                    cwd=str(epoch_dir),
                    capture_output=True,
                    text=True,
                    check=True
                )
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted job {job_id}")
                print(f"  SLURM output will be in: {slurm_output}")
                print(f"  Simulation output will be in: {epoch_dir}/output.out")
                
                # Wait for job to complete if blocking mode
                if config.get('wait_for_jobs', True):
                    print(f"  Waiting for job {job_id} to complete...")
                    wait_result = subprocess.run(
                        ['squeue', '--job', job_id, '--noheader'],
                        capture_output=True,
                        text=True
                    )
                    # Poll until job is no longer in queue
                    import time
                    timeout = config.get('timeout', 3600)
                    start_time = time.time()
                    while wait_result.returncode == 0 and wait_result.stdout.strip():
                        if time.time() - start_time > timeout:
                            print(f"  Job {job_id} timeout after {timeout}s")
                            break
                        time.sleep(10)
                        wait_result = subprocess.run(
                            ['squeue', '--job', job_id, '--noheader'],
                            capture_output=True,
                            text=True
                        )
                    print(f"  Job {job_id} completed")
                    
            except subprocess.CalledProcessError as e:
                print(f"  sbatch failed: {e.stderr}")
                continue
            except Exception as e:
                print(f"  Simulation failed: {e}")
                continue
        
        # 5. Compute reward from simulation outputs
        reward_config = config.get('reward', {})
        reward = compute_msld_reward(
            str(epoch_dir),  # Look in epoch subdirectory for outputs
            w_P=reward_config.get('w_P', 0.5),
            w_T=reward_config.get('w_T', 0.5),
            gamma=reward_config.get('gamma', 10.0),
            P_baseline=reward_config.get('P_baseline', 1000.0),
            T_baseline=reward_config.get('T_baseline', 100.0)
        )
        epoch_rewards.append(reward)
        print(f"  Reward: {reward:.4f}")
        
        # 6. REINFORCE update with baseline and entropy regularization
        # Baseline: running average of rewards
        baseline = np.mean(epoch_rewards) if len(epoch_rewards) > 1 else 0.0
        advantage = reward - baseline
        lambda_entropy = reward_config.get('lambda_entropy', 0.01)
        
        # Policy gradient loss: -E[(R - b) * log π(a|G)]
        policy_loss = -(logp.sum() * advantage)
        
        # Entropy regularization: encourages exploration
        # H(π) = 0.5 * log(2πe * σ²) for Gaussian
        entropy_loss = 0.0
        if lambda_entropy > 0:
            entropy = 0.5 * torch.log(2 * np.pi * np.e * torch.exp(2 * log_std)).sum()
            entropy_loss = -lambda_entropy * entropy
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(combos) if combos else 0.0
    avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
    
    return {
        'loss': avg_loss,
        'avg_reward': avg_reward,
        'num_combos': len(combos)
    }


def main():
    # Load config
    if len(sys.argv) < 2:
        print('No config provided, using examples/workflow_sample.yaml')
        cfg_path = str(Path(__file__).parent / 'workflow_sample.yaml')
    else:
        cfg_path = sys.argv[1]
    
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {cfg_path}")
    
    # Step 1: Generate combinations (if requested)
    if 'create_combos' in config:
        print("\n=== Generating Combinations ===")
        cc = config['create_combos']
        input_dir = Path(cc['input_dir'])
        out_dir = Path(cc['out_dir'])
        include = cc.get('include_patterns', [])
        
        created = create_combination_dirs(input_dir, out_dir, include_patterns=include)
        print(f"Created {len(created)} combination directories")
        
        # Create manifest
        manifest_path = out_dir / 'manifest.txt'
        with manifest_path.open('w') as f:
            for combo_path in created:
                f.write(str(combo_path) + '\n')
        print(f"Wrote manifest to {manifest_path}")
    else:
        manifest_path = Path(config.get('manifest', 'manifest.txt'))
    
    # Step 2: Split into train/val/test
    print("\n=== Splitting Train/Val/Test ===")
    split_config = config.get('split', {})
    train_frac = split_config.get('train_frac', 0.7)
    val_frac = split_config.get('val_frac', 0.15)
    seed = split_config.get('seed', 42)
    
    # Load all combos
    all_combos = load_manifest(str(manifest_path))
    print(f"Total combos: {len(all_combos)}")
    
    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_combos))
    rng.shuffle(indices)
    
    n_train = int(len(all_combos) * train_frac)
    n_val = int(len(all_combos) * val_frac)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    train_combos = [all_combos[i] for i in train_indices]
    val_combos = [all_combos[i] for i in val_indices]
    test_combos = [all_combos[i] for i in test_indices]
    
    print(f"Train: {len(train_combos)}, Val: {len(val_combos)}, Test: {len(test_combos)}")
    
    # Save splits
    manifest_dir = manifest_path.parent
    (manifest_dir / 'train_manifest.txt').write_text('\n'.join(train_combos) + '\n')
    (manifest_dir / 'val_manifest.txt').write_text('\n'.join(val_combos) + '\n')
    (manifest_dir / 'test_manifest.txt').write_text('\n'.join(test_combos) + '\n')
    
    # Step 3: Initialize model
    print("\n=== Initializing Model ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Build sample graph to get dimensions
    sample_combo = train_combos[0]
    sample_data, _, sample_extras = build_data_and_targets_from_combo(sample_combo)
    
    train_config = config.get('training', {})
    encoder_config = train_config.get('encoder', {})
    policy_config = train_config.get('policy', {})
    
    encoder = RGCNEncoder(
        in_dim=sample_data.x.size(1),
        hidden_dims=encoder_config.get('hidden_dims', [64, 64]),
        out_dim=encoder_config.get('out_dim', 32),
        num_relations=sample_data.edge_type.max().item() + 1
    ).to(device)
    
    policy = EdgePolicy.from_pyg_data(
        encoder=encoder,
        emb_dim=encoder_config.get('out_dim', 32),
        data=sample_data,
        mlp_hidden=policy_config.get('mlp_hidden', 64),
        mlp_out_dim=len(sample_extras['relation_names']) // 2
    ).to(device)
    
    optimizer_config = train_config.get('optimizer', {})
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(policy.parameters()),
        lr=optimizer_config.get('lr', 0.001)
    )
    
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())} params")
    print(f"Policy: {sum(p.numel() for p in policy.parameters())} params")
    
    # Step 4: Training loop
    print("\n=== Training ===")
    num_epochs = train_config.get('num_epochs', 5)
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Train on training set
        stats = train_epoch(
            encoder, policy, optimizer,
            train_combos, epoch, config, device
        )
        
        print(f"Epoch {epoch+1} Stats:")
        print(f"  Loss: {stats['loss']:.4f}")
        print(f"  Avg Reward: {stats['avg_reward']:.4f}")
        
        # Save checkpoint
        output_config = config.get('output', {})
        if output_config.get('save_checkpoints', False):
            checkpoint_freq = output_config.get('checkpoint_freq', 1)
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_dir = Path(output_config.get('base_dir', 'checkpoints'))
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state': encoder.state_dict(),
                    'policy_state': policy.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'stats': stats,
                }, checkpoint_path)
                print(f"  Saved checkpoint to {checkpoint_path}")
    
    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()
