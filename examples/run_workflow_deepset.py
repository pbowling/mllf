"""DeepSet-enhanced CB policy training workflow for MSLD simulations.

This script uses the existing EdgePolicy + REINFORCE + ValueNetwork infrastructure,
with DeepSet embeddings providing richer node features for the RGCN encoder.

The 4-step DeepSet pipeline integrates as follows:
  1. Atom-Level Physical Representation: Extract AEVs, charges, and atom IDs from PDB files
  2. Shared MLP: Process atom features through DeepSet feature extractor  
  3. Permutation-Invariant Pooling: Max-pool to get substituent embeddings (64D)
  4. These embeddings become RGCN node features (replacing count-based features)

The training workflow:
  1. Generate combinations from site/sub fragment files
  2. Split into train/val/test sets
  3. For each epoch:
     a. Build graph from RTF fragments
     b. Compute DeepSet embeddings from PDB files → node features
     c. Build PyG graph with DeepSet node features + environmental context
     d. Sample bias coefficients from EdgePolicy (RGCN + MLP)
     e. Write variables.py with predicted biases
     f. Submit and run MSLD simulations
     g. Parse simulation outputs (transitions, populations)
     h. Compute reward from simulation metrics
     i. Update policy with REINFORCE using learned value baseline
  4. Save checkpoints and track progress

Usage:
  python examples/run_workflow_deepset.py [config.yaml]

If no config is provided, uses examples/workflow_deepset.yaml by default.
"""
from pathlib import Path
import sys
import json
import yaml
import time
import subprocess
import numpy as np
import torch
from typing import Dict, List

from mllf.file_handling.generate_combinations import create_combination_dirs, create_single_combination_dir
from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.cb.graph import Graph
from mllf.cb.deepset import DeepSetFeatureExtractor
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy
from mllf.cb.value_net import ValueNetwork
from mllf.cb.graph_utils import build_pyg_graph_from_mllf_graph
from mllf.cb.train_improved import compute_msld_reward_improved
from mllf.cb.workflow_utils import (
    load_manifest,
    fix_msld_flat_for_single_site,
    check_simulation_success,
    parse_simulation_metrics
)
from mllf.cli.workflow import write_variables_from_actions
from mllf.cli.sim import run_simulation_batch


def filter_combos_by_curriculum(combos: List[Path], 
                                 min_subs: int, max_subs: int,
                                 min_sites: int, max_sites: int) -> List[Path]:
    """Filter combinations based on curriculum stage criteria."""
    filtered = []
    
    for combo_path in combos:
        if isinstance(combo_path, str):
            combo_name = Path(combo_path).name
        else:
            combo_name = combo_path.name
        
        parts = combo_name.split('__')
        if len(parts) < 2:
            continue
        
        sites_seen = set()
        num_subs = 0
        
        for part in parts:
            if not part.startswith('site'):
                continue
            try:
                site_num = int(part.split('_')[0].replace('site', ''))
                sites_seen.add(site_num)
                num_subs += 1
            except (ValueError, IndexError):
                continue
        
        num_sites = len(sites_seen)
        
        if (min_subs <= num_subs <= max_subs and 
            min_sites <= num_sites <= max_sites):
            filtered.append(combo_path)
    
    return filtered


# System-specific configuration for atoms to delete in single-site combinations
SITE_ATOMS_MAPPING = {}


def ensure_combo_dir_exists(combo_path: Path, config: Dict) -> Path:
    """Ensure a combination directory exists, creating it lazily if needed."""
    if combo_path.exists():
        return combo_path
    
    combo_metadata_path = combo_path.parent / 'combo_metadata.json'
    if not combo_metadata_path.exists():
        raise FileNotFoundError(f"Combo metadata not found at {combo_metadata_path}")
    
    with open(combo_metadata_path, 'r') as f:
        all_combo_info = json.load(f)
    
    combo_info = None
    for info in all_combo_info:
        if info['path'] == str(combo_path):
            combo_info = info
            break
    
    if combo_info is None:
        raise ValueError(f"Combo info not found for {combo_path}")
    
    create_config = config.get('create_combos', {})
    input_dir = Path(create_config.get('input_dir', 'prep_files'))
    output_dir = Path(create_config.get('out_dir', create_config.get('output_dir', 'generated_combos')))
    include_patterns = create_config.get('include_patterns', [])
    
    print(f"  Creating combo directory on-demand: {combo_path.name}")
    created_path = create_single_combination_dir(input_dir, output_dir, combo_info, include_patterns=include_patterns)
    
    return created_path


def build_graph_and_data_with_deepset(
    combo_dir: str,
    deepset_model: DeepSetFeatureExtractor,
    config: Dict,
    device: str = 'cpu'
):
    """Build graph and PyG data with DeepSet embeddings.
    
    Args:
        combo_dir: Path to combination directory
        deepset_model: Trained DeepSetFeatureExtractor
        config: Full config dict
        device: Device for computation
        
    Returns:
        Tuple of (graph, data, extras) where:
        - graph: Graph object with node metadata
        - data: PyG Data with DeepSet node features
        - extras: Dict with relation names and other metadata
    """
    combo_path = Path(combo_dir)
    
    # Parse RTF files for graph construction and charges
    rtf_results = parse_rtf_dir(combo_dir)
    rtf_dir = combo_dir
    if not rtf_results:
        prep_dir = combo_path / 'prep'
        if prep_dir.exists() and prep_dir.is_dir():
            rtf_results = parse_rtf_dir(str(prep_dir))
            rtf_dir = str(prep_dir)
    
    if not rtf_results:
        raise FileNotFoundError(f"No RTF files found in {combo_dir} or {combo_dir}/prep")
    
    # Build graph from RTF
    solvent_state = config.get('system', {}).get('solvent_state', None)
    g = Graph.from_rtf_results(rtf_results, solvent_override=solvent_state, directory=rtf_dir)
    
    # Determine PDB directory (prefer prep/ subdirectory)
    pdb_dir = combo_path / 'prep'
    if not pdb_dir.exists():
        pdb_dir = combo_path
    
    # Vocabulary configuration
    vocab_config = config.get('vocabulary', {})
    toppar_dir = vocab_config.get('toppar_dir', None)
    toppar_files = vocab_config.get('toppar_files', None)
    warn_missing_types = vocab_config.get('warn_missing_types', True)
    
    # DeepSet configuration
    deepset_config = config.get('deepset', {})
    use_deepset_only = deepset_config.get('use_deepset_only', False)
    
    # System configuration for protein and spatial filtering
    system_config = config.get('system', {})
    protein_pdb_config = system_config.get('protein_pdb', None)
    aev_cutoff = system_config.get('aev_cutoff', 5.1)
    
    # Determine protein PDB path if needed
    protein_pdb_path = None
    if solvent_state == 'protein':
        if protein_pdb_config:
            protein_pdb_path = protein_pdb_config
        else:
            # Look for protein.pdb in prep directory
            default_protein = pdb_dir / "protein.pdb"
            if default_protein.exists():
                protein_pdb_path = str(default_protein)
    
    # Build PyG data with DeepSet embeddings
    data, extras = build_pyg_graph_from_mllf_graph(
        g,
        deepset_model=deepset_model,
        pdb_dir=str(pdb_dir),
        pdb_pattern="site{site}_sub{sub}.pdb",
        rtf_results=rtf_results,
        use_deepset_only=use_deepset_only,
        toppar_dir=toppar_dir,
        toppar_files=toppar_files,
        warn_missing_types=warn_missing_types,
        prep_dir=str(pdb_dir),  # Use pdb_dir as prep_dir for spatial filtering
        protein_pdb=protein_pdb_path,
        solvent_state=solvent_state,
        aev_cutoff=aev_cutoff
    )
    
    # Move data to device
    data = data.to(device)
    
    return g, data, extras


def train_epoch(
    encoder: RGCNEncoder,
    policy: EdgePolicy,
    value_network: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    deepset_model: DeepSetFeatureExtractor,
    combos: List[str],
    epoch: int,
    config: Dict,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Run one training epoch over all combos.
    
    This uses the standard EdgePolicy + REINFORCE + ValueNetwork infrastructure,
    with DeepSet-enhanced node features instead of count-based features.
    
    Args:
        encoder: RGCN node encoder
        policy: EdgePolicy for bias coefficient prediction
        value_network: ValueNetwork for baseline estimation
        optimizer: Optimizer for policy parameters
        value_optimizer: Optimizer for value network parameters
        deepset_model: DeepSetFeatureExtractor for node embeddings
        combos: List of combo directory paths
        epoch: Current epoch number
        config: Full config dict with simulation settings
        device: Device for computation ('cpu' or 'cuda')
    
    Returns:
        Dict with epoch statistics (loss, avg_reward, value_loss, etc.)
    """
    policy.train()
    encoder.train()
    value_network.train()
    deepset_model.eval()  # DeepSet in eval mode (no training during RL)
    
    epoch_loss = 0.0
    epoch_value_loss = 0.0
    epoch_rewards = []
    
    # Track submitted jobs for concurrent execution
    job_queue = []  # List of (combo_path, epoch_dir, job_id, actions, logp, baseline, retry_count)
    max_concurrent = config.get('max_concurrent_jobs', 10)
    
    for combo_idx, combo_dir in enumerate(combos):
        combo_path = Path(combo_dir)
        combo_path = ensure_combo_dir_exists(combo_path, config)
        
        # Fix msld_flat.py for single-site combinations if needed
        fix_msld_flat_for_single_site(combo_path, SITE_ATOMS_MAPPING)
        
        # Create epoch subdirectory
        epoch_dir = combo_path / f"run_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Epoch {epoch}, Combo {combo_idx+1}/{len(combos)}: {combo_path.name}")
        
        # Check if already completed
        epoch_results_file = epoch_dir / 'epoch_results.pt'
        if epoch_results_file.exists():
            print(f"  Loading cached results from {epoch_results_file}")
            try:
                cached = torch.load(epoch_results_file)
                epoch_loss += cached['loss']
                epoch_rewards.append(cached['reward'])
                continue
            except Exception as e:
                print(f"  Warning: Failed to load cached results: {e}")
        
        # 1. Build graph and PyG data with DeepSet embeddings
        try:
            g, data, extras = build_graph_and_data_with_deepset(
                str(combo_path), deepset_model, config, device
            )
        except Exception as e:
            print(f"  ERROR: Failed to build graph: {e}")
            continue
        
        # 2. Sample bias coefficients from EdgePolicy (stochastic)
        actions, logp, mean, log_std = policy.get_actions(
            data.x, data.edge_index, data.edge_type, data.edge_attr,
            deterministic=False
        )
        
        # 3. Compute value baseline (for variance reduction)
        with torch.no_grad():
            baseline = value_network(data.x, data.edge_index, data.edge_type).item()
        
        # 4. Write variables.py with sampled biases
        variables_path = epoch_dir / 'variables.py'
        try:
            write_variables_from_actions(
                actions_tensor=actions.cpu(),
                data=data,
                extras=extras,
                output_path=str(variables_path),
                graph=g,
                bias_clip=config.get('training', {}).get('bias_clip', 1000.0)
            )
        except Exception as e:
            print(f"  ERROR: Failed to write variables.py: {e}")
            continue
        
        # 5. Submit simulation
        if config.get('run_sims', False):
            job_id = run_simulation_batch(
                combo_dirs=[str(epoch_dir)],
                timeout=config.get('timeout', 1200)
            )[0]
            job_queue.append((combo_path, epoch_dir, job_id, actions, logp, baseline, 0))
            
            # Throttle submissions
            while len(job_queue) >= max_concurrent:
                time.sleep(5)
                completed = []
                for i, (jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, jbaseline, retry_count) in enumerate(job_queue):
                    if check_simulation_success(str(jepoch_dir)):
                        completed.append(i)
                
                for i in reversed(completed):
                    jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, jbaseline, retry_count = job_queue.pop(i)
                    
                    # Compute reward
                    reward_config = config.get('reward', {})
                    try:
                        reward = compute_msld_reward_improved(str(jepoch_dir), **reward_config)
                    except Exception as e:
                        print(f"  Warning: Failed to compute reward: {e}")
                        reward = -100.0
                    
                    epoch_rewards.append(reward)
                    
                    # REINFORCE update with learned value baseline
                    advantage = reward - jbaseline
                    policy_loss = -(jlogp * advantage).sum()
                    
                    # Entropy regularization (optional)
                    lambda_entropy = reward_config.get('lambda_entropy', 0.0)
                    if lambda_entropy > 0:
                        # Encourage exploration via entropy bonus
                        log_std = log_std if 'log_std' in locals() else torch.zeros_like(jlogp)
                        entropy = 0.5 * torch.log(2 * 3.14159 * 2.71828 * torch.exp(2 * log_std)).sum()
                        policy_loss = policy_loss - lambda_entropy * entropy
                    
                    # Update policy
                    optimizer.zero_grad()
                    policy_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += policy_loss.item()
                    
                    # Update value network (MSE loss)
                    # Rebuild graph for value update
                    try:
                        _, vdata, _ = build_graph_and_data_with_deepset(
                            str(jcombo_path), deepset_model, config, device
                        )
                        value_pred = value_network(vdata.x, vdata.edge_index, vdata.edge_type)
                        value_target = torch.tensor([reward], dtype=torch.float32, device=device)
                        value_loss = torch.nn.functional.mse_loss(value_pred, value_target)
                        
                        value_optimizer.zero_grad()
                        value_loss.backward()
                        value_optimizer.step()
                        
                        epoch_value_loss += value_loss.item()
                    except Exception as e:
                        print(f"  Warning: Failed to update value network: {e}")
                    
                    # Cache results
                    torch.save({
                        'loss': policy_loss.item(),
                        'reward': reward,
                        'baseline': jbaseline,
                        'actions': jactions.cpu(),
                        'logp': jlogp.cpu()
                    }, jepoch_dir / 'epoch_results.pt')
                
                if job_queue:
                    break
    
    # Wait for remaining jobs
    max_retries = 3
    print(f"\nWaiting for {len(job_queue)} remaining jobs...")
    while job_queue:
        completed = []
        for i, (jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, jbaseline, retry_count) in enumerate(job_queue):
            if check_simulation_success(str(jepoch_dir)):
                completed.append(i)
        
        for i in reversed(completed):
            jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, jbaseline, retry_count = job_queue.pop(i)
            
            reward_config = config.get('reward', {})
            try:
                reward = compute_msld_reward_improved(str(jepoch_dir), **reward_config)
            except Exception as e:
                print(f"  Warning: Failed to compute reward: {e}")
                reward = -100.0
            
            epoch_rewards.append(reward)
            
            # REINFORCE update
            advantage = reward - jbaseline
            policy_loss = -(jlogp * advantage).sum()
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
            epoch_loss += policy_loss.item()
            
            # Update value network
            try:
                _, vdata, _ = build_graph_and_data_with_deepset(
                    str(jcombo_path), deepset_model, config, device
                )
                value_pred = value_network(vdata.x, vdata.edge_index, vdata.edge_type)
                value_target = torch.tensor([reward], dtype=torch.float32, device=device)
                value_loss = torch.nn.functional.mse_loss(value_pred, value_target)
                
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
                
                epoch_value_loss += value_loss.item()
            except Exception as e:
                print(f"  Warning: Failed to update value network: {e}")
            
            torch.save({
                'loss': policy_loss.item(),
                'reward': reward,
                'baseline': jbaseline,
                'actions': jactions.cpu(),
                'logp': jlogp.cpu()
            }, jepoch_dir / 'epoch_results.pt')
        
        if job_queue:
            time.sleep(5)
    
    avg_loss = epoch_loss / len(epoch_rewards) if epoch_rewards else 0.0
    avg_value_loss = epoch_value_loss / len(epoch_rewards) if epoch_rewards else 0.0
    avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
    
    return {
        'loss': avg_loss,
        'value_loss': avg_value_loss,
        'avg_reward': avg_reward,
        'num_combos': len(combos)
    }


def main():
    # Load config
    if len(sys.argv) < 2:
        print('No config provided, using examples/workflow_deepset.yaml')
        cfg_path = str(Path(__file__).parent / 'workflow_deepset.yaml')
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
        
        manifest_path = out_dir / 'manifest.txt'
        if not manifest_path.exists():
            print(f"Creating combinations in {out_dir}")
            create_combination_dirs(
                str(input_dir),
                str(out_dir),
                include_patterns=cc.get('include_patterns', []),
                create_dirs=cc.get('create_dirs_immediately', False)
            )
        else:
            print(f"Manifest already exists at {manifest_path}")
    else:
        out_dir = Path(config.get('output', {}).get('base_dir', 'generated_combos'))
        manifest_path = out_dir / 'manifest.txt'
    
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
    
    # Step 3: Initialize DeepSet + RGCN + EdgePolicy + ValueNetwork
    print("\n=== Initializing DeepSet-Enhanced CB Policy ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Build sample graph to get dimensions
    sample_combo = train_combos[0]
    sample_combo_path = ensure_combo_dir_exists(Path(sample_combo), config)
    
    # Initialize DeepSet model (for computing node embeddings)
    deepset_config = config.get('deepset', {})
    deepset_model = DeepSetFeatureExtractor(
        aev_length=deepset_config.get('aev_length', 2288),
        num_atom_types=deepset_config.get('num_atom_types', 11),
        embedding_dim=deepset_config.get('embedding_dim', 64),
        hidden_dim=deepset_config.get('hidden_dim', 256),
        include_charge=deepset_config.get('include_charge', True),
        include_atom_id=deepset_config.get('include_atom_id', True)
    ).to(device)
    
    # Build sample graph to infer dimensions
    sample_g, sample_data, sample_extras = build_graph_and_data_with_deepset(
        str(sample_combo_path), deepset_model, config, device
    )
    
    # Initialize standard CB policy infrastructure
    train_config = config.get('training', {})
    encoder_config = train_config.get('encoder', {})
    policy_config = train_config.get('policy', {})
    
    # RGCN Encoder (processes DeepSet node embeddings)
    encoder = RGCNEncoder(
        in_dim=sample_data.x.size(1),  # 66D = 64 (DeepSet) + 2 (environmental)
        hidden_dims=encoder_config.get('hidden_dims', [64, 64]),
        out_dim=encoder_config.get('out_dim', 32),
        num_relations=sample_data.edge_type.max().item() + 1
    ).to(device)
    
    # EdgePolicy (uses RGCN embeddings to predict bias coefficients)
    policy = EdgePolicy.from_pyg_data(
        encoder=encoder,
        emb_dim=encoder_config.get('out_dim', 32),
        data=sample_data,
        mlp_hidden=policy_config.get('mlp_hidden', 64),
        mlp_out_dim=len(sample_extras['relation_names']) // 2  # 4 bias types
    ).to(device)
    
    # ValueNetwork (learned baseline for variance reduction)
    value_config = train_config.get('value_network', {})
    value_network = ValueNetwork(
        emb_dim=encoder_config.get('out_dim', 32),
        hidden_dims=value_config.get('hidden_dims', [64, 32])
    ).to(device)
    
    # Optimizers
    optimizer_config = train_config.get('optimizer', {})
    optimizer = torch.optim.Adam(
        policy.parameters(),  # Includes encoder parameters
        lr=optimizer_config.get('lr', 0.001)
    )
    
    value_lr = value_config.get('lr', optimizer_config.get('lr', 0.001) * 10)
    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=value_lr
    )
    
    print(f"DeepSet: {sum(p.numel() for p in deepset_model.parameters())} params")
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())} params")
    print(f"Policy: {sum(p.numel() for p in policy.parameters())} params")
    print(f"Value Network: {sum(p.numel() for p in value_network.parameters())} params")
    
    # Step 4: Load pretrained model or resume from checkpoint
    output_config = config.get('output', {})
    checkpoint_dir = Path(output_config.get('base_dir', 'checkpoints'))
    start_epoch = 0
    
    pretrain_path = config.get('pretrain', {}).get('model_path', None)
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            policy.load_state_dict(checkpoint['policy_state_dict'])
            value_network.load_state_dict(checkpoint['value_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'deepset_state_dict' in checkpoint:
                deepset_model.load_state_dict(checkpoint['deepset_state_dict'])
    elif pretrain_path and Path(pretrain_path).exists():
        print(f"Loading pretrained policy from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=device)
        # Pretrained checkpoint should have the same structure
        if 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if 'policy_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'deepset_state_dict' in checkpoint:
            deepset_model.load_state_dict(checkpoint['deepset_state_dict'])
    
    # Step 5: Training loop
    print("\n=== Training ===")
    num_epochs = train_config.get('num_epochs', 100)
    all_stats = []
    
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{num_epochs-1}")
        print(f"{'='*80}")
        
        stats = train_epoch(
            encoder, policy, value_network, optimizer, value_optimizer,
            deepset_model, train_combos, epoch, config, device
        )
        all_stats.append(stats)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Policy Loss: {stats['loss']:.4f}")
        print(f"  Value Loss: {stats['value_loss']:.4f}")
        print(f"  Avg Reward: {stats['avg_reward']:.4f}")
        
        # Save checkpoint
        if output_config.get('save_checkpoints', True):
            if epoch % output_config.get('checkpoint_freq', 5) == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_{epoch:03d}.pt'
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'policy_state_dict': policy.state_dict(),
                    'value_state_dict': value_network.state_dict(),
                    'deepset_state_dict': deepset_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'value_optimizer_state_dict': value_optimizer.state_dict(),
                    'stats': stats
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
    
    print("\n=== Training Complete ===")
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value_network.state_dict(),
        'deepset_state_dict': deepset_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'value_optimizer_state_dict': value_optimizer.state_dict(),
        'all_stats': all_stats
    }, final_path)
    print(f"Saved final model: {final_path}")


if __name__ == '__main__':
    main()
