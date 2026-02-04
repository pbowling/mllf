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
import json
from typing import Dict, List
import re

from mllf.file_handling.generate_combinations import create_combination_dirs, create_single_combination_dir
from mllf.cli.workflow import (
    build_data_and_targets_from_combo,
    write_variables_from_actions,
)
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy
from mllf.cb.train_improved import compute_msld_reward_improved
from mllf.cb.workflow_utils import (
    load_manifest,
    fix_msld_flat_for_single_site,
    check_simulation_success,
    parse_simulation_metrics
)
from mllf.cli.sim import run_simulation_batch


def filter_combos_by_curriculum(combos: List[Path], 
                                 min_subs: int, max_subs: int,
                                 min_sites: int, max_sites: int) -> List[Path]:
    """Filter combinations based on curriculum stage criteria.
    
    Parses combination directory names to extract number of sites and substituents,
    then filters based on curriculum constraints.
    
    Args:
        combos: List of combination directory paths
        min_subs: Minimum number of substituents
        max_subs: Maximum number of substituents
        min_sites: Minimum number of sites
        max_sites: Maximum number of sites
    
    Returns:
        Filtered list of combinations matching curriculum criteria
    """
    filtered = []
    
    for combo_path in combos:
        # Handle both string paths and Path objects
        if isinstance(combo_path, str):
            combo_name = Path(combo_path).name
        else:
            combo_name = combo_path.name
        
        # Parse combination name: comb_NNNN_site1_1__site1_2__site2_3...
        # Count unique sites and total substituents
        parts = combo_name.split('__')
        
        if len(parts) < 2:
            continue  # Invalid format
        
        sites_seen = set()
        num_subs = 0
        
        for part in parts:
            # Each part is like 'site1_1' or from comb prefix 'comb_0001_site1_1'
            if 'site' in part:
                # Extract siteX_Y pattern
                site_match = re.search(r'site(\d+)_(\d+)', part)
                if site_match:
                    site_id = int(site_match.group(1))
                    sites_seen.add(site_id)
                    num_subs += 1
        
        num_sites = len(sites_seen)
        
        # Check if this combo matches curriculum criteria
        if (min_subs <= num_subs <= max_subs and 
            min_sites <= num_sites <= max_sites):
            filtered.append(combo_path)
    
    return filtered


# ============================================================================
# SYSTEM-SPECIFIC CONFIGURATION
# ============================================================================
# Atoms to delete for single-site combinations (prevents overlap with base structure)
# Customize this mapping for your molecular system:
# Example: {1: 'C4 H4', 2: 'C5 H5'} for old 14benz_solv_5.5 system
# Leave empty {} if no atoms need to be deleted
SITE_ATOMS_MAPPING = {}
# ============================================================================


def ensure_combo_dir_exists(combo_path: Path, config: Dict) -> Path:
    """Ensure a combination directory exists, creating it lazily if needed.
    
    Args:
        combo_path: Path to the combination directory.
        config: Config dict with 'create_combos' section containing input_dir and output_dir.
    
    Returns:
        Path to the combo directory (guaranteed to exist).
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
    # Support both 'out_dir' and 'output_dir' keys
    output_dir = Path(create_config.get('out_dir', create_config.get('output_dir', 'generated_combos')))
    include_patterns = create_config.get('include_patterns', [])
    
    # Create the directory
    print(f"  Creating combo directory on-demand: {combo_path.name}")
    created_path = create_single_combination_dir(input_dir, output_dir, combo_info, include_patterns=include_patterns)
    
    return created_path


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
    import subprocess
    import time
    
    policy.train()
    encoder.train()
    
    epoch_loss = 0.0
    epoch_rewards = []
    
    # Track submitted jobs for concurrent execution
    job_queue = []  # List of (combo_path, epoch_dir, job_id, actions, logp)
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
            try:
                cached = torch.load(epoch_results_file, weights_only=False)
                reward_config = config.get('reward', {})
                
                # Check if we need to recompute reward with new config
                # This enables "pretraining" data reuse with different reward functions
                cached_reward_config = cached.get('reward_config', {})
                reward_changed = (
                    cached_reward_config.get('w_P') != reward_config.get('w_P', 0.5) or
                    cached_reward_config.get('w_T') != reward_config.get('w_T', 0.75) or
                    cached_reward_config.get('w_U') != reward_config.get('w_U', 0.3) or
                    cached_reward_config.get('gamma') != reward_config.get('gamma', 4.0) or
                    cached_reward_config.get('P_baseline') != reward_config.get('P_baseline', 500.0) or
                    cached_reward_config.get('T_baseline') != reward_config.get('T_baseline', 50.0) or
                    cached_reward_config.get('min_transitions_per_site') != reward_config.get('min_transitions_per_site', 10) or
                    cached_reward_config.get('min_coverage_ratio') != reward_config.get('min_coverage_ratio', 0.5) or
                    cached_reward_config.get('entropy_bonus') != reward_config.get('entropy_bonus', 8.0) or
                    cached_reward_config.get('concentration_penalty_threshold') != reward_config.get('concentration_penalty_threshold', 0.8)
                )
                
                if reward_changed:
                    # Recompute reward with new configuration using improved reward function
                    print(f"  Reward config changed - recomputing with improved reward function")
                    
                    reward = compute_msld_reward_improved(
                        str(epoch_dir),
                        w_P=reward_config.get('w_P', 0.5),
                        w_T=reward_config.get('w_T', 0.75),
                        w_U=reward_config.get('w_U', 0.3),
                        gamma=reward_config.get('gamma', 4.0),
                        P_baseline=reward_config.get('P_baseline', 500.0),
                        T_baseline=reward_config.get('T_baseline', 50.0),
                        min_transitions_per_site=reward_config.get('min_transitions_per_site', 10),
                        min_coverage_ratio=reward_config.get('min_coverage_ratio', 0.5),
                        entropy_bonus=reward_config.get('entropy_bonus', 8.0),
                        concentration_penalty_threshold=reward_config.get('concentration_penalty_threshold', 0.8)
                    )
                    print(f"  Old reward: {cached['reward']:.4f} -> New reward: {reward:.4f}")
                    
                    # Update cached reward for this run
                    cached['reward'] = reward
                    cached['reward_config'] = reward_config
                else:
                    # Use cached reward as-is
                    reward = cached['reward']
                
                epoch_rewards.append(reward)
                
                # REINFORCE update: Recompute forward pass to get fresh gradients
                try:
                    data, targets, extras = build_data_and_targets_from_combo(str(combo_path))
                    data = data.to(device)
                    
                    # Recompute logp with gradients enabled
                    _, logp_new, _, _ = policy.get_actions(
                        data.x, data.edge_index, data.edge_type, data.edge_attr,
                        deterministic=False
                    )
                    
                    baseline = np.mean(epoch_rewards) if len(epoch_rewards) > 1 else 0.0
                    advantage = reward - baseline
                    
                    policy_loss = -(logp_new.sum() * advantage)
                    
                    # Entropy regularization
                    lambda_entropy = reward_config.get('lambda_entropy', 0.01)
                    entropy_loss = 0.0
                    if lambda_entropy > 0:
                        entropy_loss = -lambda_entropy * logp_new.var()
                    
                    loss = policy_loss + entropy_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    print(f"  Using cached reward: {reward:.4f}")
                    continue
                except Exception as e:
                    print(f"  Error recomputing forward pass from cache: {e}, recomputing full simulation...")
            except Exception as e:
                print(f"  Error loading cached results: {e}, recomputing...")
        
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
            train_config = config.get('training', {})
            bias_clip = train_config.get('bias_clip', 1000.0)
            write_variables_from_actions(
                str(combo_path), data, extras, actions,
                out_name=str(variables_path.relative_to(combo_path)),
                bias_clip=bias_clip
            )
            print(f"  Wrote variables to {variables_path}")
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
#SBATCH --time=00:10:00

module load charmm/charmm/c51a1

# Change to combo directory and run with epoch-specific variables
cd {combo_path}
python3 msld_flat.py --vars-file {variables_path.relative_to(combo_path)} --out-dir {epoch_dir.relative_to(combo_path)} > {epoch_dir.relative_to(combo_path)}/output.out 2>&1
"""
            epoch_run_script.write_text(run_script_content)
            epoch_run_script.chmod(0o755)
            
            #print(f"  Submitting job via sbatch...")
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
                # Include retry count (0 = first attempt)
                job_queue.append((combo_path, epoch_dir, job_id, actions.detach(), logp.detach(), 0))
                    
            except subprocess.CalledProcessError as e:
                print(f"  sbatch failed: {e.stderr}")
                continue
            except Exception as e:
                print(f"  Simulation submission failed: {e}")
                continue
    
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
                    epoch_run_script = jepoch_dir / 'run.sh'
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
                data, targets, extras = build_data_and_targets_from_combo(str(jcombo_path))
                data = data.to(device)
                
                # Recompute actions and logp with gradients enabled
                _, logp_new, _, _ = policy.get_actions(
                    data.x, data.edge_index, data.edge_type, data.edge_attr,
                    deterministic=False
                )
                
                # REINFORCE loss with current policy
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
        print('No config provided, using examples/workflow_sample.yaml')
        cfg_path = str(Path(__file__).parent / 'workflow_sample.yaml')
    else:
        cfg_path = sys.argv[1]
    
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from {cfg_path}")
    
    # Step 1: Generate combination metadata (if requested)
    if 'create_combos' in config:
        print("\n=== Generating Combination Metadata ===")
        cc = config['create_combos']
        input_dir = Path(cc['input_dir'])
        out_dir = Path(cc['out_dir'])
        include = cc.get('include_patterns', [])
        
        # Import the new function
        from mllf.file_handling.generate_combinations import list_possible_combinations
        
        # List all possible combinations without creating directories
        all_combo_info = list_possible_combinations(input_dir, out_dir)
        print(f"Found {len(all_combo_info)} possible combinations")
        
        # Create manifest with paths (directories don't exist yet)
        manifest_path = out_dir / 'manifest.txt'
        out_dir.mkdir(parents=True, exist_ok=True)
        with manifest_path.open('w') as f:
            for combo_info in all_combo_info:
                f.write(combo_info['path'] + '\n')
        print(f"Wrote manifest to {manifest_path}")
        
        # Store combo_info for later lazy creation
        # Save to a JSON file so we can recreate directories on demand
        combo_metadata_path = out_dir / 'combo_metadata.json'
        import json
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
    
    # Load all combos
    all_combos = load_manifest(str(manifest_path))
    print(f"Total combos: {len(all_combos)}")
    
    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_combos))
    rng.shuffle(indices)
    
    # Check if curriculum learning is enabled
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
    
    # If curriculum learning is enabled, prepare stage information
    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        if not stages:
            print("Warning: Curriculum enabled but no stages defined. Disabling curriculum.")
            curriculum_enabled = False
        else:
            print(f"\n=== Curriculum Learning Enabled ===")
            print(f"Total stages: {len(stages)}")
            for i, stage in enumerate(stages, 1):
                print(f"  Stage {i}: {stage['name']} - "
                      f"{stage['min_subs']}-{stage['max_subs']} subs, "
                      f"{stage['min_sites']}-{stage['max_sites']} sites, "
                      f"{stage['epochs']} epochs")
    
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
    sample_combo_path = ensure_combo_dir_exists(Path(sample_combo), config)
    sample_data, _, sample_extras = build_data_and_targets_from_combo(str(sample_combo_path))
    
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
    # Optimizer: policy.parameters() already includes encoder since encoder is a submodule
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=optimizer_config.get('lr', 0.001)
    )
    
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())} params")
    print(f"Policy: {sum(p.numel() for p in policy.parameters())} params")
    
    # Step 4: Load pretrained policy or resume from checkpoint
    output_config = config.get('output', {})
    checkpoint_dir = Path(output_config.get('base_dir', 'checkpoints'))
    start_epoch = 0
    
    # Check for pretrained policy
    pretrain_config = config.get('pretrain', {})
    pretrain_path = pretrain_config.get('model_path', None)
    
    if pretrain_path and Path(pretrain_path).exists():
        print(f"\n=== Loading pretrained policy: {pretrain_path} ===")
        pretrained = torch.load(pretrain_path, map_location=device, weights_only=False)
        encoder.load_state_dict(pretrained['encoder_state'])
        policy.load_state_dict(pretrained['policy_state'])
        print(f"Loaded pretrained policy from epoch {pretrained.get('epoch', 'unknown')}")
        if 'avg_reward' in pretrained:
            print(f"Pretraining avg reward: {pretrained['avg_reward']:.4f}")
    elif checkpoint_dir.exists():
        # Find the latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"\n=== Resuming from checkpoint: {latest_checkpoint} ===")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder_state'])
            policy.load_state_dict(checkpoint['policy_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
    
    # Step 5: Training loop
    print("\n=== Training ===")
    train_config = config.get('training', {})
    
    # Determine total epochs based on curriculum or direct config
    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        num_epochs = sum(stage['epochs'] for stage in stages)
        print(f"Total curriculum epochs: {num_epochs}")
    else:
        num_epochs = train_config.get('num_epochs', 5)
        print(f"Training for {num_epochs} epochs")
    
    # Training loop with curriculum support
    all_stats = []
    current_stage_idx = 0
    current_stage_epoch = 0
    
    # Determine active combinations for curriculum
    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        current_stage = stages[current_stage_idx]
        print(f"\n=== Starting Stage 1/{len(stages)}: {current_stage['name']} ===")
        
        # Filter train combos for this stage
        filtered_combos = filter_combos_by_curriculum(
            train_combos,
            min_subs=current_stage['min_subs'],
            max_subs=current_stage['max_subs'],
            min_sites=current_stage['min_sites'],
            max_sites=current_stage['max_sites']
        )
        print(f"Filtered to {len(filtered_combos)} training combinations for this stage")
        
        # Apply max_train_combos limit if specified (stage-specific or global)
        max_combos = current_stage.get('max_train_combos', curriculum_config.get('max_train_combos_per_stage'))
        if max_combos is not None and len(filtered_combos) > max_combos:
            print(f"Limiting to {max_combos} random training combos (from {len(filtered_combos)} available)")
            # Use same RNG seed for reproducibility
            stage_rng = np.random.RandomState(seed + current_stage_idx)
            selected_indices = stage_rng.choice(len(filtered_combos), size=max_combos, replace=False)
            active_train_combos = [filtered_combos[i] for i in selected_indices]
        else:
            active_train_combos = filtered_combos
    else:
        active_train_combos = train_combos
    
    for epoch in range(start_epoch, num_epochs):
        # Check if we need to advance to next curriculum stage
        if curriculum_enabled and current_stage_idx < len(stages):
            current_stage = stages[current_stage_idx]
            current_stage_epoch += 1
            
            # Check if current stage is complete
            if current_stage_epoch > current_stage['epochs']:
                # Check progression criteria
                progression_type = curriculum_config.get('progression', {}).get('type', 'epoch')
                reward_threshold = curriculum_config.get('progression', {}).get('reward_threshold', 0.0)
                
                can_advance = False
                if progression_type == 'epoch':
                    can_advance = True
                elif progression_type == 'reward':
                    recent_rewards = [s['avg_reward'] for s in all_stats[-5:] if 'avg_reward' in s]
                    avg_recent_reward = np.mean(recent_rewards) if recent_rewards else -999
                    can_advance = avg_recent_reward >= reward_threshold
                elif progression_type == 'both':
                    recent_rewards = [s['avg_reward'] for s in all_stats[-5:] if 'avg_reward' in s]
                    avg_recent_reward = np.mean(recent_rewards) if recent_rewards else -999
                    can_advance = avg_recent_reward >= reward_threshold
                else:
                    can_advance = True
                
                if can_advance and current_stage_idx + 1 < len(stages):
                    # Advance to next stage
                    current_stage_idx += 1
                    current_stage_epoch = 1
                    current_stage = stages[current_stage_idx]
                    
                    print(f"\n{'='*60}")
                    print(f"=== Advancing to Stage {current_stage_idx + 1}/{len(stages)}: {current_stage['name']} ===")
                    print(f"{'='*60}")
                    
                    # Filter train combos for new stage
                    filtered_combos = filter_combos_by_curriculum(
                        train_combos,
                        min_subs=current_stage['min_subs'],
                        max_subs=current_stage['max_subs'],
                        min_sites=current_stage['min_sites'],
                        max_sites=current_stage['max_sites']
                    )
                    print(f"Filtered to {len(filtered_combos)} training combinations for this stage")
                    
                    # Apply max_train_combos limit if specified (stage-specific or global)
                    max_combos = current_stage.get('max_train_combos', curriculum_config.get('max_train_combos_per_stage'))
                    if max_combos is not None and len(filtered_combos) > max_combos:
                        print(f"Limiting to {max_combos} random training combos (from {len(filtered_combos)} available)")
                        # Use same RNG seed for reproducibility
                        stage_rng = np.random.RandomState(seed + current_stage_idx)
                        selected_indices = stage_rng.choice(len(filtered_combos), size=max_combos, replace=False)
                        active_train_combos = [filtered_combos[i] for i in selected_indices]
                    else:
                        active_train_combos = filtered_combos
                elif not can_advance:
                    print(f"\nStage {current_stage_idx + 1} not meeting progression criteria, continuing...")
        
        if curriculum_enabled:
            stage_name = stages[current_stage_idx]['name'] if current_stage_idx < len(stages) else 'final'
            print(f"\n--- Epoch {epoch+1}/{num_epochs} - Stage {current_stage_idx + 1}/{len(stages)}: {stage_name} (epoch {current_stage_epoch}/{current_stage['epochs']}) ---")
        else:
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Apply stage-specific reward overrides if defined
        epoch_config = config.copy()
        if curriculum_enabled and current_stage_idx < len(stages):
            current_stage = stages[current_stage_idx]
            if 'reward_override' in current_stage:
                # Merge reward overrides into config for this epoch
                base_reward = epoch_config.get('reward', {}).copy()
                base_reward.update(current_stage['reward_override'])
                epoch_config['reward'] = base_reward
                print(f"  Applying reward overrides: {current_stage['reward_override']}")
        
        # Train on active training set
        stats = train_epoch(
            encoder, policy, optimizer,
            active_train_combos, epoch, epoch_config, device
        )
        all_stats.append(stats)
        
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
    
    # Step 6: Archive combinations if requested
    archive_config = config.get('archive', {})
    if archive_config.get('enabled', False):
        print("\n=== Archiving Combinations ===")
        from mllf.file_handling.generate_combinations import archive_combo_dirs
        import shutil
        
        # Archive from the generated_combos directory
        combos_dir = manifest_path.parent if 'create_combos' in config else Path(config.get('manifest', 'manifest.txt')).parent
        pattern = archive_config.get('pattern', 'comb_*')
        remove_after = archive_config.get('remove_after', False)
        archive_dir = Path(archive_config.get('archive_dir', combos_dir / 'archives'))
        
        # Create archive directory if it doesn't exist
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Archive the combinations
        print(f"Archiving combinations from {combos_dir} matching '{pattern}'")
        archived = archive_combo_dirs(combos_dir, pattern=pattern, remove=remove_after)
        
        # Move archives to archive directory if specified and different from source
        if archive_dir != combos_dir:
            for archive_path in archived:
                dest_path = archive_dir / archive_path.name
                print(f"  Moving {archive_path.name} to {archive_dir}")
                shutil.move(str(archive_path), str(dest_path))
        
        print(f"Archived {len(archived)} combination directories")
        if remove_after:
            print(f"  Removed original directories after archiving")


if __name__ == '__main__':
    main()
