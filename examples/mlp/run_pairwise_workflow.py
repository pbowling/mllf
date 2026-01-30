#!/usr/bin/env python3
"""Pairwise MLP training workflow for CB-optimized MSLD simulations.

This script implements the full training loop using the pairwise MLP policy:
  1. Generate combinations from site/sub fragment files
  2. Split into train/val/test sets
  3. For each epoch:
     a. Load substituent features from RTF fragments
     b. Sample actions (bias coefficients) from pairwise MLP policy
     c. Write variables.py with predicted biases
     d. Submit and run MSLD simulations
     e. Parse simulation outputs (transitions, populations)
     f. Compute reward from simulation metrics
     g. Update policy with REINFORCE
  4. Save checkpoints and track progress

Usage:
  python run_pairwise_workflow.py workflow_14benz_mlp.yaml
"""
import sys
import json
import yaml
import torch
import numpy as np
import subprocess
import time
from pathlib import Path
from typing import Dict, List

from mllf.file_handling.generate_combinations import create_single_combination_dir
from mllf.cb.pairwise_mlp_policy import PairwiseMLPPolicy
from mllf.cb.pairwise_utils import (
    load_substituent_features_from_combo,
    write_variables_from_pairwise_predictions,
)
from mllf.cb.train_improved import compute_msld_reward_improved
from mllf.cb.workflow_utils import (
    load_manifest,
    fix_msld_flat_for_single_site,
    check_simulation_success,
    parse_simulation_metrics
)


def filter_combos_by_curriculum(combos: List[Path], 
                                 min_subs: int, max_subs: int,
                                 min_sites: int, max_sites: int) -> List[Path]:
    """Filter combinations based on curriculum stage criteria.
    
    Args:
        combos: List of combination directory paths
        min_subs: Minimum number of substituents
        max_subs: Maximum number of substituents
        min_sites: Minimum number of sites
        max_sites: Maximum number of sites
    
    Returns:
        Filtered list of combinations matching curriculum criteria
    """
    import re
    
    filtered = []
    
    for combo_path in combos:
        if isinstance(combo_path, str):
            combo_name = Path(combo_path).name
        else:
            combo_name = combo_path.name
        
        # Parse combination name: comb_NNNN_site1_1__site1_2__site2_3...
        parts = combo_name.split('__')
        
        if len(parts) < 1:
            continue
        
        sites_seen = set()
        num_subs = 0
        
        for part in parts:
            # Extract siteX_Y pattern (handles both 'site1_1' and 'comb_0001_site1_1')
            if 'site' in part:
                site_match = re.search(r'site(\d+)_(\d+)', part)
                if site_match:
                    site_id = int(site_match.group(1))
                    sites_seen.add(site_id)
                    num_subs += 1
        
        num_sites = len(sites_seen)
        
        if (min_subs <= num_subs <= max_subs and 
            min_sites <= num_sites <= max_sites):
            filtered.append(combo_path)
    
    return filtered


# System-specific configuration
SITE_ATOMS_MAPPING = {}


def ensure_combo_dir_exists(combo_path: Path, config: Dict) -> Path:
    """Ensure a combination directory exists, creating it lazily if needed.
    
    Args:
        combo_path: Path to the combination directory
        config: Config dict with 'create_combos' section
    
    Returns:
        Path to the combo directory (guaranteed to exist)
    """
    if combo_path.exists():
        return combo_path
    
    # Load combo metadata
    combo_metadata_path = combo_path.parent / 'combo_metadata.json'
    if not combo_metadata_path.exists():
        raise FileNotFoundError(f"Combo metadata not found at {combo_metadata_path}")
    
    with open(combo_metadata_path, 'r') as f:
        all_combo_info = json.load(f)
    
    # Find the matching combo info
    combo_info = None
    for info in all_combo_info:
        if info['path'] == str(combo_path):
            combo_info = info
            break
    
    if combo_info is None:
        raise ValueError(f"Combo info not found for {combo_path}")
    
    # Get input/output directories from config
    create_config = config.get('create_combos', {})
    input_dir = Path(create_config.get('input_dir', 'prep_files'))
    output_dir = Path(create_config.get('out_dir', create_config.get('output_dir', 'generated_combos')))
    include_patterns = create_config.get('include_patterns', [])
    
    # Create the directory
    print(f"  Creating combo directory on-demand: {combo_path.name}")
    created_path = create_single_combination_dir(input_dir, output_dir, combo_info, include_patterns=include_patterns)
    
    return created_path


def train_epoch(
    policy: PairwiseMLPPolicy,
    optimizer: torch.optim.Optimizer,
    combos: List[str],
    epoch: int,
    config: Dict,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Run one training epoch over all combos.
    
    Args:
        policy: Pairwise MLP policy network
        optimizer: Optimizer for policy parameters
        combos: List of combo directory paths
        epoch: Current epoch number
        config: Full config dict with simulation settings
        device: Device for computation ('cpu' or 'cuda')
    
    Returns:
        Dict with epoch statistics (loss, avg_reward, etc.)
    """
    policy.train()
    
    epoch_loss = 0.0
    epoch_rewards = []
    
    # Track submitted jobs for concurrent execution
    job_queue = []  # List of (combo_path, epoch_dir, job_id, actions, logp, retry_count)
    max_concurrent = config.get('max_concurrent_jobs', 10)
    
    for combo_idx, combo_dir in enumerate(combos):
        combo_path = Path(combo_dir)
        
        # Lazily create combo directory if it doesn't exist yet
        combo_path = ensure_combo_dir_exists(combo_path, config)
        
        # Fix msld_flat.py for single-site combinations if needed
        fix_msld_flat_for_single_site(combo_path, SITE_ATOMS_MAPPING)
        
        # Create epoch subdirectory for outputs
        epoch_dir = combo_path / f"run_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Epoch {epoch}, Combo {combo_idx+1}/{len(combos)}: {combo_path.name}")
        
        # Check if this epoch was already completed (for resume capability)
        epoch_results_file = epoch_dir / 'epoch_results.pt'
        if epoch_results_file.exists():
            print(f"  Loading cached results from {epoch_results_file}")
            cached = torch.load(epoch_results_file, map_location=device)
            
            epoch_loss += cached['loss']
            epoch_rewards.append(cached['reward'])
            
            print(f"  Reward: {cached['reward']:.4f}, Loss: {cached['loss']:.4f}")
            continue
        
        # 1. Load substituent features from RTF fragments
        try:
            features, pairs, metadata = load_substituent_features_from_combo(
                str(combo_path),
                solvent_override=None
            )
            features = features.to(device)
            nsubs_per_site = metadata['nsubs_per_site']
            
        except Exception as e:
            print(f"  Error loading features: {e}")
            continue
        
        # 2. Sample actions from policy (stochastic)
        pairs_tensor = torch.tensor(pairs, dtype=torch.long, device=device)
        actions, logp, mean, log_std = policy.get_actions(
            features, pairs_tensor, deterministic=False
        )
        
        # 3. Write variables.py with sampled biases
        variables_path = epoch_dir / 'variables.py'
        try:
            write_variables_from_pairwise_predictions(
                str(epoch_dir),
                actions.detach().cpu(),
                pairs,
                nsubs_per_site,
                output_filename='variables.py'
            )
        except Exception as e:
            print(f"  Error writing variables: {e}")
            continue
        
        # 4. Submit simulation to queue (don't wait yet)
        if config.get('run_sims', False):
            # Create epoch-specific run script that uses the correct variables.py
            epoch_run_script = epoch_dir / 'run_epoch.sh'
            slurm_output = epoch_dir / f"{combo_path.name}_ep{epoch}.%j.out"
            run_script_content = f"""#!/bin/bash
#SBATCH --job-name={combo_path.name}_ep{epoch}
#SBATCH --output={slurm_output}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p ada6000 --gres=gpu:1 
#SBATCH --export=ALL
#SBATCH --time=01:00:00

module load charmm/charmm/c51a1

# Change to combo directory and run with epoch-specific variables
cd {combo_path}
python3 msld_flat.py --vars-file {variables_path.relative_to(combo_path)} --out-dir {epoch_dir.relative_to(combo_path)} > {epoch_dir.relative_to(combo_path)}/output.out 2>&1
"""
            epoch_run_script.write_text(run_script_content)
            epoch_run_script.chmod(0o755)
            
            try:
                # Wait if we're at the concurrent job limit
                while len(job_queue) >= max_concurrent:
                    # Check which jobs have completed
                    completed = []
                    for i, (jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, retry_count) in enumerate(job_queue):
                        result = subprocess.run(
                            ['squeue', '--job', jjob_id, '--noheader'],
                            capture_output=True,
                            text=True
                        )
                        if not result.stdout.strip():
                            # Job completed
                            completed.append(i)
                            print(f"  Job {jjob_id} completed")
                    
                    # Remove completed jobs (will process after all submitted)
                    for i in reversed(completed):
                        job_queue.pop(i)
                    
                    if len(job_queue) >= max_concurrent:
                        time.sleep(5)  # Wait before checking again
                
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
                
                # Add to queue for later processing
                # Detach tensors to avoid gradient graph issues across iterations
                job_queue.append((combo_path, epoch_dir, job_id, actions.detach(), logp.detach(), 0))
                    
            except subprocess.CalledProcessError as e:
                print(f"  sbatch failed: {e.stderr}")
                continue
            except Exception as e:
                print(f"  Simulation submission failed: {e}")
                continue
            
            # If queue is full, wait for some jobs to complete
            while len(job_queue) >= max_concurrent:
                # Check which jobs have completed
                completed = []
                for i, (jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, retry_count) in enumerate(job_queue):
                    result = subprocess.run(
                        ['squeue', '--job', jjob_id, '--noheader'],
                        capture_output=True,
                        text=True
                    )
                    if not result.stdout.strip():
                        # Job completed
                        completed.append(i)
                        print(f"  Job {jjob_id} completed")
                
                # Remove completed jobs (will process after all submitted)
                for i in reversed(completed):
                    job_queue.pop(i)
                
                if len(job_queue) >= max_concurrent:
                    time.sleep(5)  # Wait before checking again
    
    # Wait for remaining jobs to complete
    max_retries = 3  # Maximum number of retry attempts per simulation
    print(f"\nWaiting for {len(job_queue)} remaining jobs...")
    while job_queue:
        completed = []
        for i, (jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, retry_count) in enumerate(job_queue):
            result = subprocess.run(
                ['squeue', '--job', jjob_id, '--noheader'],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                # Job completed
                completed.append(i)
                print(f"  Job {jjob_id} completed")
        
        # Remove completed jobs and process them
        for i in reversed(completed):
            jcombo_path, jepoch_dir, jjob_id, jactions_old, jlogp_old, retry_count = job_queue.pop(i)
            
            # Check if simulation terminated normally
            output_file = jepoch_dir / 'output.out'
            simulation_success = check_simulation_success(output_file)
            
            if not simulation_success:
                print(f"  Warning: Could not verify simulation success for {output_file}")
            
            # If simulation failed and we haven't exceeded max retries, resubmit
            if not simulation_success and retry_count < max_retries:
                print(f"  Warning: Simulation failed for {jcombo_path.name} in {jepoch_dir.name}")
                print(f"  Retrying (attempt {retry_count + 2}/{max_retries + 1})...")
                
                try:
                    # Resubmit the same job
                    epoch_run_script = jepoch_dir / 'run_epoch.sh'
                    result = subprocess.run(
                        ['sbatch', str(epoch_run_script)],
                        cwd=str(jepoch_dir),
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    new_job_id = result.stdout.strip().split()[-1]
                    print(f"  Resubmitted job {new_job_id}")
                    
                    # Add back to queue with incremented retry count
                    job_queue.append((jcombo_path, jepoch_dir, new_job_id, jactions_old, jlogp_old, retry_count + 1))
                except Exception as e:
                    print(f"  Error resubmitting job: {e}")
                    print(f"  Skipping combo {jcombo_path.name} - manual intervention required")
                
                continue  # Skip processing this job for now
            
            # If simulation still failed after max retries, stop the entire training
            if not simulation_success:
                print("\n" + "=" * 70)
                print(f"FATAL ERROR: Simulation failed after {max_retries + 1} attempts")
                print(f"  Combo: {jcombo_path.name}")
                print(f"  Epoch directory: {jepoch_dir}")
                print(f"  Output file: {output_file}")
                print("\nTraining cannot continue with failed simulations.")
                print("Please check the simulation logs and fix the issue before restarting.")
                print("=" * 70)
                sys.exit(1)
            
            # Simulation succeeded - parse outputs and update policy
            raw_metrics = parse_simulation_metrics(output_file)
            
            # Compute reward with current config using IMPROVED reward function
            reward_config = config.get('reward', {})
            reward = compute_msld_reward_improved(
                str(jepoch_dir),
                w_P=reward_config.get('w_P', 0.5),
                w_T=reward_config.get('w_T', 0.5),
                w_U=reward_config.get('w_U', 0.3),
                gamma=reward_config.get('gamma', 4.0),
                P_baseline=reward_config.get('P_baseline', 500.0),
                T_baseline=reward_config.get('T_baseline', 50.0),
                min_transitions_per_site=reward_config.get('min_transitions_per_site', 10),
                min_coverage_ratio=reward_config.get('min_coverage_ratio', 0.5),
                entropy_bonus=reward_config.get('entropy_bonus', 8.0),
                concentration_penalty_threshold=reward_config.get('concentration_penalty_threshold', 0.8)
            )
            epoch_rewards.append(reward)
            print(f"  Combo {jcombo_path.name} reward: {reward:.4f}")
            
            # Save epoch results with RAW METRICS for flexible recomputation
            epoch_results_file = jepoch_dir / 'epoch_results.pt'
            torch.save({
                'reward': reward,
                'actions': jactions_old.cpu(),
                'logp': jlogp_old.cpu(),
                'epoch': epoch,
                'combo': str(jcombo_path.name),
                # Raw simulation metrics for reward recomputation
                'populations': raw_metrics['populations'],
                'transitions': raw_metrics['transitions'],
                'reward_config': reward_config  # Save config used for this reward
            }, epoch_results_file)
            
            # REINFORCE update: Recompute forward pass for this combo to get fresh gradients
            try:
                features, pairs, metadata = load_substituent_features_from_combo(
                    str(jcombo_path),
                    solvent_override=None
                )
                features = features.to(device)
                pairs_tensor = torch.tensor(pairs, dtype=torch.long, device=device)
                
                # Recompute actions and logp with gradients enabled
                _, logp_new, _, _ = policy.get_actions(
                    features, pairs_tensor, deterministic=False
                )
                
                # REINFORCE loss with baseline
                baseline = np.mean(epoch_rewards) if len(epoch_rewards) > 1 else 0.0
                advantage = reward - baseline
                
                policy_loss = -(logp_new.sum() * advantage)
                
                # Entropy regularization
                lambda_entropy = reward_config.get('lambda_entropy', 0.01)
                entropy_loss = 0.0
                if lambda_entropy > 0:
                    # Simple entropy approximation
                    entropy_loss = -lambda_entropy * logp_new.var()
                
                loss = policy_loss + entropy_loss
                
                # Check for NaN or inf in loss before backprop
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: NaN/inf loss detected for {jcombo_path.name}, skipping update")
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
            except Exception as e:
                print(f"  Warning: Could not recompute forward pass for {jcombo_path.name}: {e}")
                # Still count the reward even if policy update fails
                continue
        
        if job_queue:
            time.sleep(10)  # Wait before checking again
    
    avg_loss = epoch_loss / len(epoch_rewards) if epoch_rewards else 0.0
    avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
    
    return {
        'loss': avg_loss,
        'avg_reward': avg_reward,
        'num_combos': len(combos)
    }


def main():
    # Load config
    if len(sys.argv) < 2:
        print('Usage: python run_pairwise_workflow.py <config.yaml>')
        print('Example: python run_pairwise_workflow.py workflow_14benz_mlp.yaml')
        return 1
    
    cfg_path = sys.argv[1]
    
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {cfg_path}")
    
    # Step 1: Generate combination metadata (if requested)
    if 'create_combos' in config:
        print("\n=== Generating Combination Metadata ===")
        cc = config['create_combos']
        input_dir = Path(cc['input_dir'])
        out_dir = Path(cc.get('output_dir', cc.get('out_dir', 'generated_combos')))
        
        from mllf.file_handling.generate_combinations import list_possible_combinations
        
        all_combo_info = list_possible_combinations(input_dir, out_dir)
        print(f"Found {len(all_combo_info)} possible combinations")
        
        # Create manifest
        manifest_path = out_dir / 'manifest.txt'
        out_dir.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('w') as f:
            for info in all_combo_info:
                f.write(f"{info['path']}\n")
        print(f"Wrote manifest to {manifest_path}")
        
        # Save combo metadata
        combo_metadata_path = out_dir / 'combo_metadata.json'
        with open(combo_metadata_path, 'w') as f:
            json.dump(all_combo_info, f, indent=2)
        print(f"Saved combination metadata to {combo_metadata_path}")
    else:
        manifest_path = Path(config.get('manifest', 'manifest.txt'))
    
    # Step 2: Split into train/val/test
    print("\n=== Splitting Train/Val/Test ===")
    split_config = config.get('split', {})
    train_frac = split_config.get('train_frac', 0.7)
    val_frac = split_config.get('val_frac', 0.15)
    seed = split_config.get('seed', 42)
    
    all_combos = load_manifest(str(manifest_path))
    print(f"Total combos: {len(all_combos)}")
    
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_combos))
    rng.shuffle(indices)
    
    curriculum_config = config.get('curriculum', {})
    curriculum_enabled = curriculum_config.get('enabled', False)
    
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
    
    # Step 3: Initialize pairwise MLP policy
    print("\n=== Initializing Pairwise MLP Policy ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_config = config.get('training', {})
    policy_config = train_config.get('policy', {})
    
    # Get feature dimension from a sample combo or config
    sample_combo = train_combos[0]
    sample_combo_path = ensure_combo_dir_exists(Path(sample_combo), config)
    sample_features, _, sample_metadata = load_substituent_features_from_combo(str(sample_combo_path))
    feature_dim = sample_metadata['feature_dim']
    
    print(f"Auto-detected feature dimension: {feature_dim}")
    
    policy = PairwiseMLPPolicy(
        feature_dim=feature_dim,
        hidden_dims=policy_config.get('hidden_dims', [256, 128]),
        num_bias_types=policy_config.get('num_bias_types', 4),
        bias_embed_dim=policy_config.get('bias_embed_dim', 16),
        dropout=policy_config.get('dropout', 0.1)
    ).to(device)
    
    optimizer_config = train_config.get('optimizer', {})
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=optimizer_config.get('lr', 0.001)
    )
    
    print(f"Policy: {sum(p.numel() for p in policy.parameters()):,} parameters")
    
    # Step 4: Load pretrained policy or resume from checkpoint
    output_config = config.get('output', {})
    checkpoint_dir = Path(output_config.get('base_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    
    # Check for pretrained policy
    pretrain_config = config.get('pretrain', {})
    pretrain_path = pretrain_config.get('model_path', None)
    
    if pretrain_path and Path(pretrain_path).exists():
        print(f"\nLoading pretrained policy from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=device)
        policy.load_state_dict(checkpoint['policy_state'])
        print("  Pretrained policy loaded successfully")
    elif checkpoint_dir.exists():
        # Resume from latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"\nResuming from checkpoint: {latest}")
            checkpoint = torch.load(latest, map_location=device)
            policy.load_state_dict(checkpoint['policy_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            print(f"  Resumed from epoch {start_epoch}")
    
    # Step 5: Training loop
    print("\n=== Training ===")
    
    # Determine total epochs
    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        num_epochs = sum(stage['epochs'] for stage in stages)
    else:
        num_epochs = train_config.get('num_epochs', 10)
    
    all_stats = []
    current_stage_idx = 0
    current_stage_epoch = 0
    
    # Determine active combinations for curriculum
    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        current_stage = stages[current_stage_idx]
        active_combos = filter_combos_by_curriculum(
            train_combos,
            current_stage['min_subs'],
            current_stage['max_subs'],
            current_stage['min_sites'],
            current_stage['max_sites']
        )
        max_train = current_stage.get('max_train_combos', None)
        if max_train and len(active_combos) > max_train:
            active_combos = rng.choice(active_combos, max_train, replace=False).tolist()
        
        print(f"\nStage 1/{len(stages)}: {current_stage['name']}")
        print(f"  Active combos: {len(active_combos)}")
    else:
        active_combos = train_combos
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Check curriculum progression
        if curriculum_enabled:
            current_stage_epoch += 1
            if current_stage_epoch >= current_stage['epochs']:
                # Advance to next stage
                current_stage_idx += 1
                if current_stage_idx < len(stages):
                    current_stage_epoch = 0
                    current_stage = stages[current_stage_idx]
                    
                    active_combos = filter_combos_by_curriculum(
                        train_combos,
                        current_stage['min_subs'],
                        current_stage['max_subs'],
                        current_stage['min_sites'],
                        current_stage['max_sites']
                    )
                    max_train = current_stage.get('max_train_combos', None)
                    if max_train and len(active_combos) > max_train:
                        active_combos = rng.choice(active_combos, max_train, replace=False).tolist()
                    
                    print(f"\n{'='*60}")
                    print(f"Stage {current_stage_idx+1}/{len(stages)}: {current_stage['name']}")
                    print(f"  Active combos: {len(active_combos)}")
                    print(f"{'='*60}\n")
        
        # Run training epoch
        stats = train_epoch(policy, optimizer, active_combos, epoch+1, config, device)
        all_stats.append(stats)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {stats['loss']:.4f}")
        print(f"  Avg Reward: {stats['avg_reward']:.4f}")
        print(f"  Combos: {stats['num_combos']}")
        
        # Save checkpoint
        if output_config.get('save_checkpoints', True):
            checkpoint_freq = output_config.get('checkpoint_freq', 5)
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'policy_state': policy.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'stats': all_stats,
                    'config': config
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = checkpoint_dir / 'final_pairwise_policy.pt'
    torch.save({
        'epoch': num_epochs,
        'policy_state': policy.state_dict(),
        'stats': all_stats,
        'config': config
    }, final_path)
    print(f"\n=== Training Complete ===")
    print(f"Final model saved to: {final_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
