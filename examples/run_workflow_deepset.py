"""DeepSet-enhanced CB policy training workflow for MSLD simulations.

This script uses the existing EdgePolicy + REINFORCE + ValueNetwork infrastructure,
with DeepSet embeddings providing richer node features for the RGCN encoder.

The 4-step DeepSet pipeline integrates as follows:
  1. Atom-Level Physical Representation: Extract AEVs, charges, and atom IDs from PDB files
  2. Shared MLP: Process atom features through DeepSet feature extractor  
  3. Permutation-Invariant Pooling: Sum-pool to get substituent embeddings (64D)
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
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from mllf.file_handling.generate_combinations import create_combination_dirs, create_single_combination_dir, list_possible_combinations
from mllf.file_handling.read_rtf import parse_rtf_dir
from mllf.cb.graph import Graph
from mllf.cb.deepset_autoencoder import load_pretrained_atombondgnn
from mllf.cb.warmstart import WarmStartMapper
from mllf.cb.rgcn import RGCNEncoder
from mllf.cb.policy import EdgePolicy
from mllf.cb.value_net import ValueNetwork, QNetwork
from mllf.cb.graph_utils import build_pyg_graph_from_mllf_graph
from mllf.cb.train_improved import compute_msld_reward_improved
from mllf.cb.workflow_utils import (
    load_manifest,
    fix_msld_flat_for_single_site,
    check_simulation_success,
    parse_simulation_metrics,
    compute_pair_reward,
)
from mllf.cli.workflow import write_variables_from_actions
from mllf.cli.sim import run_simulation_batch


def filter_combos_by_curriculum(combos: List[Path],
                                 min_sites: int, max_sites: int,
                                 min_subs_per_site: int = 1,
                                 max_subs_per_site: Optional[int] = None) -> List[Path]:
    """Filter combinations based on curriculum stage criteria.

    Args:
        combos: List of combination directory paths
        min_sites: Minimum number of sites
        max_sites: Maximum number of sites
        min_subs_per_site: Minimum substituents at any single site (default 1)
        max_subs_per_site: Maximum substituents at any single site (default unconstrained)

    Returns:
        Filtered list of combinations matching curriculum criteria
    """
    filtered = []

    for combo_path in combos:
        if isinstance(combo_path, str):
            combo_name = Path(combo_path).name
        else:
            combo_name = combo_path.name

        parts = combo_name.split('__')

        subs_per_site: Dict[int, int] = {}

        for part in parts:
            if 'site' not in part:
                continue
            # New format: 'site{N}_subs_{a}_{b}_{c}' — count underscored sub IDs after 'subs_'
            site_match = re.search(r'site(\d+)_subs_([\d_]+)', part)
            if site_match:
                site_id = int(site_match.group(1))
                sub_ids = site_match.group(2).split('_')
                subs_per_site[site_id] = len(sub_ids)

        if not subs_per_site:
            continue

        num_sites = len(subs_per_site)
        site_counts = list(subs_per_site.values())

        if not (min_sites <= num_sites <= max_sites):
            continue

        if min(site_counts) < min_subs_per_site:
            continue

        if max_subs_per_site is not None and max(site_counts) > max_subs_per_site:
            continue

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
    deepset_model,
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
        pdb_pattern="site{site}_sub{sub}_frag.pdb",
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
    rl_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    deepset_model,
    combos: List[str],
    epoch: int,
    config: Dict,
    device: str = 'cpu',
    q_network: 'QNetwork' = None,
    q_optimizer: torch.optim.Optimizer = None,
    warmstart_mapper: 'WarmStartMapper' = None,
    warmstart_epoch: int = 0,
) -> Dict[str, float]:
    """Run one training epoch over all combos.

    Args:
        encoder: RGCN node encoder (frozen during RL — only updated by BC pretraining)
        policy: EdgePolicy for bias coefficient prediction
        value_network: ValueNetwork for diagnostic baseline (trained but not used in actor loss)
        rl_optimizer: Optimizer for ``policy.edge_mlp`` parameters only
        value_optimizer: Optimizer for value network parameters
        deepset_model: DeepSetFeatureExtractor / AtomBondGNN for node embeddings
        combos: List of combo directory paths
        epoch: Current epoch number
        config: Full config dict with simulation settings
        device: Device for computation ('cpu' or 'cuda')
        q_network: Per-edge Q-value critic (optional)
        q_optimizer: Optimizer for q_network parameters (optional)
        warmstart_mapper: Optional WarmStartMapper; if provided and
            ``epoch == warmstart_epoch``, its pretraining biases are used
            instead of sampling from the policy.
        warmstart_epoch: Epoch at which to apply the warm start (default 0).

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
    deferred_updates = []  # List of (combo_path, epoch_dir, reward, baseline) — batch-updated at epoch end

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
                cached = torch.load(epoch_results_file, weights_only=False)
                reward_config = config.get('reward', {})
                # Detect if reward config changed; recompute reward if needed
                cached_reward_config = cached.get('reward_config', {})
                reward_changed = any(
                    cached_reward_config.get(k) != reward_config.get(k)
                    for k in ('w_P', 'w_T', 'w_U', 'gamma', 'P_baseline', 'T_baseline',
                              'min_transitions_per_site', 'min_coverage_ratio',
                              'entropy_bonus', 'concentration_penalty_threshold')
                )
                if reward_changed:
                    print(f"  Reward config changed - recomputing reward")
                    try:
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
                        print(f"  Old reward: {cached.get('reward', 'N/A')} -> New reward: {reward:.4f}")
                    except Exception as _e:
                        print(f"  Warning: Could not recompute reward: {_e}")
                        reward = cached['reward']
                else:
                    reward = cached['reward']
                epoch_rewards.append(reward)
                # Defer to end-of-epoch batch REINFORCE update; compute baseline now (no grad)
                try:
                    _g, _data, _extras = build_graph_and_data_with_deepset(
                        str(combo_path), deepset_model, config, device
                    )
                    with torch.no_grad():
                        _node_emb = encoder(_data.x, _data.edge_index, _data.edge_type)
                        _baseline = value_network(_node_emb).item()
                    deferred_updates.append((combo_path, epoch_dir, reward, _baseline))
                    print(f"  Using cached reward: {reward:.4f}")
                    continue
                except Exception as _e:
                    print(f"  Error recomputing forward pass from cache: {_e}, rerunning simulation...")
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
        
        # 2. Sample bias coefficients from EdgePolicy — or use warm-start biases.
        # At warmstart_epoch, WarmStartMapper provides pretraining-calibrated
        # actions directly, breaking the cold-start floor-mask deadlock.
        # evaluate_logp is called at epoch-end under current policy params
        # (off-policy REINFORCE), so this is consistent with the cached-results
        # path that already uses the same pattern.
        ws_actions = None
        if warmstart_mapper is not None and epoch == warmstart_epoch:
            ws_actions = warmstart_mapper.get_actions_for_combo(
                combo_path, g, data, extras, device
            )
            if ws_actions is not None:
                print(f"  Warm-start: using pretraining biases "
                      f"(system={warmstart_mapper.system}, "
                      f"run={warmstart_mapper.source_runs[0] if warmstart_mapper.source_runs else '?'})")

        if ws_actions is not None:
            actions = ws_actions
            # Compute logp of warmstart actions under the current policy
            # (for saving to epoch_results.pt; the actual REINFORCE gradient
            # uses evaluate_logp at epoch-end, so this value is informational).
            with torch.no_grad():
                logp, _ = policy.evaluate_logp(
                    data.x, data.edge_index, data.edge_type,
                    data.edge_attr, actions
                )
        else:
            actions, logp, mean, log_std = policy.get_actions(
                data.x, data.edge_index, data.edge_type, data.edge_attr,
                deterministic=False
            )

        # 3. Compute value baseline (for variance reduction)
        with torch.no_grad():
            node_emb = encoder(data.x, data.edge_index, data.edge_type)
            baseline = value_network(node_emb).item()
        
        # 4. Write variables.py with sampled biases
        try:
            write_variables_from_actions(
                combo_dir=str(epoch_dir),
                data=data,
                extras=extras,
                actions=actions.cpu(),
                bias_clip=config.get('training', {}).get('bias_clip', 1000.0)
            )
        except Exception as e:
            print(f"  ERROR: Failed to write variables.py: {e}")
            continue
        
        # 5. Submit simulation via SLURM sbatch
        if config.get('run_sims', False):
            epoch_run_script = epoch_dir / 'run_epoch.sh'
            slurm_output = epoch_dir / f"{combo_path.name}_ep{epoch}.%j.out"
            variables_path = epoch_dir / 'variables.py'
            run_script_content = f"""#!/bin/bash
#SBATCH --job-name={combo_path.name}_ep{epoch}
#SBATCH --output={slurm_output}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p ada6000 --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --time=00:20:00

module load charmm/charmm/c51a1

cd {combo_path}
python3 msld_flat.py --vars-file {variables_path.relative_to(combo_path)} --out-dir {epoch_dir.relative_to(combo_path)} > {epoch_dir.relative_to(combo_path)}/output.out 2>&1
"""
            epoch_run_script.write_text(run_script_content)
            epoch_run_script.chmod(0o755)
            try:
                result = subprocess.run(
                    ['sbatch', str(epoch_run_script)],
                    cwd=str(epoch_dir),
                    capture_output=True, text=True, check=True
                )
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted SLURM job {job_id}")
                job_queue.append((combo_path, epoch_dir, job_id, actions.detach(), logp.detach(), baseline, 0))
            except subprocess.CalledProcessError as e:
                print(f"  sbatch failed: {e.stderr}")
                continue
            except Exception as e:
                print(f"  Simulation submission failed: {e}")
                continue

            # Throttle: wait if at the concurrent job limit
            while len(job_queue) >= max_concurrent:
                time.sleep(5)
                _throttle_done = []
                for _ti, (_jcp, _jed, _jjid, _, _, _, _) in enumerate(job_queue):
                    _r = subprocess.run(
                        ['squeue', '--job', _jjid, '--noheader'],
                        capture_output=True, text=True
                    )
                    if not _r.stdout.strip():
                        _throttle_done.append(_ti)
                for _ti in reversed(_throttle_done):
                    job_queue.pop(_ti)

    # Wait for remaining jobs and process them
    max_retries = 3
    print(f"\nWaiting for {len(job_queue)} remaining jobs...")
    while job_queue:
        completed = []
        for i, (jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, jbaseline, retry_count) in enumerate(job_queue):
            r = subprocess.run(
                ['squeue', '--job', jjob_id, '--noheader'],
                capture_output=True, text=True
            )
            if not r.stdout.strip():
                completed.append(i)
                print(f"  Job {jjob_id} completed")

        for i in reversed(completed):
            jcombo_path, jepoch_dir, jjob_id, jactions, jlogp, jbaseline, retry_count = job_queue.pop(i)

            output_file = jepoch_dir / 'output.out'
            simulation_success = check_simulation_success(output_file)

            if not simulation_success and retry_count < max_retries:
                print(f"  Simulation failed for {jcombo_path.name}, retrying "
                      f"(attempt {retry_count + 2}/{max_retries + 1})...")
                epoch_run_script = jepoch_dir / 'run_epoch.sh'
                try:
                    result = subprocess.run(
                        ['sbatch', str(epoch_run_script)],
                        cwd=str(jepoch_dir),
                        capture_output=True, text=True, check=True
                    )
                    new_job_id = result.stdout.strip().split()[-1]
                    job_queue.append((jcombo_path, jepoch_dir, new_job_id,
                                      jactions, jlogp, jbaseline, retry_count + 1))
                except Exception as e:
                    print(f"  Error resubmitting: {e} — skipping {jcombo_path.name}")
                continue

            if not simulation_success:
                print(f"\nFATAL: Simulation failed after {max_retries + 1} attempts "
                      f"for {jcombo_path.name}")
                print(f"  Check: {output_file}")
                sys.exit(1)

            # Compute reward (explicit kwargs — reward_config may have extra keys)
            reward_config = config.get('reward', {})
            try:
                reward = compute_msld_reward_improved(
                    str(jepoch_dir),
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
            except Exception as e:
                print(f"  Warning: Failed to compute reward for {jcombo_path.name}: {e}")
                reward = -100.0

            epoch_rewards.append(reward)
            print(f"  Combo {jcombo_path.name} reward: {reward:.4f}")

            # Save with raw metrics so reward can be recomputed later
            raw_metrics = parse_simulation_metrics(output_file)
            torch.save({
                'reward': reward,
                'actions': jactions.cpu(),
                'logp': jlogp.cpu(),
                'baseline': jbaseline,
                'reward_config': reward_config,
                'populations': raw_metrics.get('populations', {}),
                'transitions': raw_metrics.get('transitions', {}),
                'ddg_pairs': raw_metrics.get('ddg_pairs', {}),
            }, jepoch_dir / 'epoch_results.pt')

            # Defer REINFORCE + value update to end-of-epoch batch update
            deferred_updates.append((jcombo_path, jepoch_dir, reward, jbaseline))

        if job_queue:
            time.sleep(10)

    # === Epoch-batch REINFORCE update ===
    # Per-edge advantage using per-pair DDG reward and Q-network baseline.
    # Normalise A_pair per-combo before accumulating (prevents combos with many
    # edges from dominating the gradient).
    if deferred_updates:
        reward_config = config.get('reward', {})
        lambda_e = reward_config.get('lambda_entropy', 0.0)

        # floor_mask_min_transitions: combos below this threshold are excluded from
        # the policy gradient (reward is noise-floor, not informative signal).
        min_trans = reward_config.get(
            'floor_mask_min_transitions',
            reward_config.get('min_transitions_per_site', 3)
        )

        # Count informative combos to normalise accumulated gradient magnitude.
        n_policy_updates = 0
        for _, epoch_dir_u, _, _ in deferred_updates:
            try:
                sd = torch.load(Path(epoch_dir_u) / 'epoch_results.pt', weights_only=False)
                trans_vals_u = sd.get('transitions', [])
                tv = list(trans_vals_u.values()) if isinstance(trans_vals_u, dict) else list(trans_vals_u)
                if min(tv) >= min_trans if tv else False:
                    n_policy_updates += 1
            except Exception:
                pass
        if n_policy_updates == 0:
            n_policy_updates = 1

        rl_optimizer.zero_grad()
        for (combo_path_u, epoch_dir_u, reward_u, _) in deferred_updates:
            try:
                _, vdata_u, _ = build_graph_and_data_with_deepset(
                    str(combo_path_u), deepset_model, config, device
                )

                # ── Value network update (diagnostic; uses global scalar reward) ──
                # Detach RGCN output so value backward does not touch encoder weights.
                with torch.no_grad():
                    vnode_emb = encoder(vdata_u.x, vdata_u.edge_index, vdata_u.edge_type)
                value_pred = value_network(vnode_emb.detach())
                val_loss = F.mse_loss(
                    value_pred,
                    torch.tensor([reward_u], dtype=torch.float32, device=device),
                )
                value_optimizer.zero_grad()
                val_loss.backward()
                value_optimizer.step()
                epoch_value_loss += val_loss.item()

                # ── Load saved actions ──
                results_pt = Path(epoch_dir_u) / 'epoch_results.pt'
                saved_data = torch.load(results_pt, weights_only=False)

                # Skip policy gradient for floor-reward combos.
                site_trans = saved_data.get('transitions', [])
                trans_vals = list(site_trans.values()) if isinstance(site_trans, dict) else list(site_trans)
                min_site_trans = min(trans_vals) if trans_vals else 0
                if min_site_trans < min_trans:
                    continue

                saved_actions = saved_data['actions'].to(device)

                # ── Compute per-edge log-prob under current policy (skip connection active) ──
                logp_saved, log_std_saved = policy.evaluate_logp(
                    vdata_u.x, vdata_u.edge_index, vdata_u.edge_type,
                    vdata_u.edge_attr, saved_actions,
                )

                # ── Per-edge reward (DDG binary + population balance) ──
                ddg_pairs = saved_data.get('ddg_pairs', {})
                populations = saved_data.get('populations', [])
                r_pair = compute_pair_reward(
                    vdata_u.edge_index, ddg_pairs, populations
                ).to(device)   # [E]

                # ── Q-network update and per-edge advantage ──
                if q_network is not None and q_optimizer is not None:
                    # Build edge inputs the same way the policy does internally
                    with torch.no_grad():
                        p2_emb = encoder(vdata_u.x, vdata_u.edge_index, vdata_u.edge_type)
                        p1_for_skip = vdata_u.x if policy.p1_dim > 0 else None
                        edge_inp = policy.edge_inputs(
                            p2_emb, vdata_u.edge_index, vdata_u.edge_attr, p1_for_skip
                        )
                    # Q(s, a): critic grades the actual actions the actor submitted
                    q_pred = q_network(edge_inp.detach(), saved_actions.detach())   # [E]
                    q_loss = F.mse_loss(q_pred, r_pair.detach())
                    q_optimizer.zero_grad()
                    q_loss.backward()
                    q_optimizer.step()
                    # Per-edge advantage: subtract Q(s, a) baseline (detached)
                    with torch.no_grad():
                        q_baseline = q_network(edge_inp, saved_actions)   # [E]
                    a_pair = (r_pair - q_baseline).detach()   # [E]
                else:
                    # Fall back to per-edge reward without Q baseline
                    a_pair = r_pair.detach()   # [E]

                # Normalise advantage within this combo
                a_std = a_pair.std()
                if a_std > 1e-6:
                    a_pair = (a_pair - a_pair.mean()) / (a_std + 1e-8)
                else:
                    a_pair = torch.zeros_like(a_pair)

                # ── REINFORCE policy loss (per-edge) ──
                policy_loss = -(logp_saved * a_pair).sum() / n_policy_updates
                if lambda_e > 0:
                    policy_loss = policy_loss - lambda_e * log_std_saved.mean() / n_policy_updates
                if not (torch.isnan(policy_loss) or torch.isinf(policy_loss)):
                    policy_loss.backward()
                    epoch_loss += policy_loss.item() * n_policy_updates

            except Exception as e:
                print(f"  Warning: Could not update network for {combo_path_u.name}: {e}")

        # Single RL optimizer step — only updates edge_mlp; encoder is frozen.
        torch.nn.utils.clip_grad_norm_(policy.edge_mlp.parameters(), max_norm=1.0)
        rl_optimizer.step()

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
            all_combo_info = list_possible_combinations(
                input_dir,
                out_dir,
                max_subs_per_site=cc.get('max_subs_per_site', 10)
            )
            print(f"Found {len(all_combo_info)} possible combinations")

            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / 'manifest.txt').open('w') as f:
                for info in all_combo_info:
                    f.write(info['path'] + '\n')

            combo_metadata_path = out_dir / 'combo_metadata.json'
            with open(combo_metadata_path, 'w') as f:
                json.dump(all_combo_info, f, indent=2)
            print(f"Saved combination metadata to {combo_metadata_path}")
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

    # Print curriculum stage summary if enabled
    _curriculum_cfg = config.get('curriculum', {})
    if _curriculum_cfg.get('enabled', False):
        _stages = _curriculum_cfg.get('stages', [])
        if not _stages:
            print("Warning: Curriculum enabled but no stages defined.")
        else:
            print(f"\n=== Curriculum Learning Enabled ===")
            print(f"Total stages: {len(_stages)}")
            for _i, _s in enumerate(_stages, 1):
                print(f"  Stage {_i}: {_s['name']} - "
                      f"{_s.get('min_subs_per_site', 1)}-{_s.get('max_subs_per_site', '?')} subs/site, "
                      f"{_s.get('min_sites', 1)}-{_s.get('max_sites', '?')} sites, "
                      f"{_s['epochs']} epochs")

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
    
    # Load pretrained AtomBondGNN encoder (GINConv+GlobalAttentionPool, frozen weights)
    deepset_config = config.get('deepset', {})
    encoder_path = deepset_config.get('encoder_path')
    if not encoder_path or not Path(encoder_path).exists():
        raise ValueError(
            f"deepset.encoder_path must be set in config and point to best_encoder.pt. "
            f"Got: {encoder_path}"
        )
    print(f"\nLoading pretrained AtomBondGNN encoder from {encoder_path}...")
    deepset_model = load_pretrained_atombondgnn(encoder_path, freeze_weights=True).to(device)
    
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
        in_dim=sample_data.x.size(1),  # 66D = 64 (DeepSet sum-pool) + 2 (environmental)
        hidden_dims=encoder_config.get('hidden_dims', [64, 64]),
        out_dim=encoder_config.get('out_dim', 32),
        num_relations=sample_data.edge_type.max().item() + 1
    ).to(device)
    
    # EdgePolicy (uses RGCN embeddings + skip-connected P1 embeddings to predict bias coefficients)
    policy = EdgePolicy.from_pyg_data(
        encoder=encoder,
        emb_dim=encoder_config.get('out_dim', 32),
        data=sample_data,
        mlp_hidden=policy_config.get('mlp_hidden', 64),
        mlp_out_dim=len(sample_extras['relation_names']) // 2,  # 4 bias types
        p1_dim=sample_data.x.size(1),  # skip connection: pre-RGCN node features
    ).to(device)
    
    # ValueNetwork (diagnostic baseline — trained on global reward but not used in actor loss)
    value_config = train_config.get('value_network', {})
    value_network = ValueNetwork(
        emb_dim=encoder_config.get('out_dim', 32),
        hidden_dims=value_config.get('hidden_dims', [64, 32])
    ).to(device)

    # QNetwork (per-edge critic for DDG-based credit assignment)
    q_network = QNetwork(
        in_dim=policy.edge_mlp.trunk[0].in_features,
        action_dim=policy.mlp_out_dim,  # 4 bias types: linear, quadratic, skew, end
        hidden_dims=[64, 32],
    ).to(device)

    # Freeze RGCN encoder: only updated by BC pretraining, not RL.
    encoder.requires_grad_(False)

    # Optimizers — RL optimizer touches edge_mlp only; encoder excluded.
    optimizer_config = train_config.get('optimizer', {})
    rl_optimizer = torch.optim.Adam(
        policy.edge_mlp.parameters(),
        lr=optimizer_config.get('lr', 0.001)
    )
    value_lr = value_config.get('lr', optimizer_config.get('lr', 0.001) * 10)
    value_optimizer = torch.optim.Adam(
        value_network.parameters(),
        lr=value_lr
    )
    q_lr = optimizer_config.get('q_lr', optimizer_config.get('lr', 0.001) * 5)
    q_optimizer = torch.optim.Adam(
        q_network.parameters(),
        lr=q_lr
    )
    
    print(f"DeepSet: {sum(p.numel() for p in deepset_model.parameters())} params")
    print(f"Encoder: {sum(p.numel() for p in encoder.parameters())} params (frozen during RL)")
    print(f"Policy edge_mlp: {sum(p.numel() for p in policy.edge_mlp.parameters())} params")
    print(f"Q-Network: {sum(p.numel() for p in q_network.parameters())} params")
    print(f"Value Network: {sum(p.numel() for p in value_network.parameters())} params")
    print(f"Skip connection p1_dim: {policy.p1_dim}  edge_mlp in_dim: {policy.edge_mlp.trunk[0].in_features}")
    
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
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            policy.load_state_dict(checkpoint['policy_state_dict'])
            value_network.load_state_dict(checkpoint['value_state_dict'])
            rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
            value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            if 'q_network_state_dict' in checkpoint:
                q_network.load_state_dict(checkpoint['q_network_state_dict'])
            if 'q_optimizer_state_dict' in checkpoint:
                q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'deepset_state_dict' in checkpoint:
                deepset_model.load_state_dict(checkpoint['deepset_state_dict'])
    elif pretrain_path and Path(pretrain_path).exists():
        print(f"Loading pretrained policy from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
        # Pretrain checkpoint uses 'encoder_state'/'policy_state' keys.
        # AtomBondGNN is already loaded from deepset.encoder_path above (frozen
        # during pretraining, not stored in best_policy.pt).
        if 'encoder_state' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state'])
            print("  Loaded encoder weights from pretrain checkpoint")
        if 'policy_state' in checkpoint:
            policy.load_state_dict(checkpoint['policy_state'])
            print("  Loaded policy weights from pretrain checkpoint")
        # Load Q-critic warmup weights if pretrained_q.pt exists alongside best_policy.pt
        q_pretrain_path = Path(pretrain_path).parent / 'pretrained_q.pt'
        if q_pretrain_path.exists():
            print(f"  Loading Q-critic warmup from {q_pretrain_path}")
            q_ckpt = torch.load(q_pretrain_path, map_location=device, weights_only=False)
            q_network.load_state_dict(q_ckpt['q_state'])
            print("  Loaded Q-critic weights from pretrain Q checkpoint")
        else:
            print("  No pretrained_q.pt found alongside policy checkpoint — Q-critic starts from random init")

    # Step 4b: Load warm-start bias mapper (optional)
    warmstart_mapper = None
    warmstart_config = config.get('warmstart', {})
    ws_map_path = warmstart_config.get('mapping_path')
    warmstart_epoch = int(warmstart_config.get('epoch', 0))
    if ws_map_path and Path(ws_map_path).exists():
        warmstart_mapper = WarmStartMapper(ws_map_path)
        print(
            f"\nLoaded warm-start mapping: {ws_map_path}"
            f"\n  system={warmstart_mapper.system}, "
            f"runs={warmstart_mapper.source_runs}, "
            f"score({warmstart_mapper.score_type})={warmstart_mapper.score:.1f}"
            f"\n  Will apply warm-start biases at epoch {warmstart_epoch}"
        )
    elif ws_map_path:
        print(f"  Warning: warmstart.mapping_path not found: {ws_map_path} — skipping warm start")

    # Step 5: Training loop
    print("\n=== Training ===")
    curriculum_config = config.get('curriculum', {})
    curriculum_enabled = curriculum_config.get('enabled', False)

    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        num_epochs = sum(stage['epochs'] for stage in stages)
        print(f"Curriculum learning enabled: {len(stages)} stages, {num_epochs} total epochs")
    else:
        num_epochs = train_config.get('num_epochs', 100)
        print(f"Training for {num_epochs} epochs")

    all_stats = []
    current_stage_idx = 0
    current_stage_epoch = 0
    active_archive_jobs = []  # Track background archive processes

    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Determine the initial active combo set
    if curriculum_enabled:
        stages = curriculum_config.get('stages', [])
        current_stage = stages[0]
        print(f"\n=== Stage 1/{len(stages)}: {current_stage['name']} ===")
        active_train_combos = filter_combos_by_curriculum(
            train_combos,
            min_sites=current_stage['min_sites'],
            max_sites=current_stage['max_sites'],
            min_subs_per_site=current_stage.get('min_subs_per_site', 1),
            max_subs_per_site=current_stage.get('max_subs_per_site', None)
        )
        print(f"  Filtered to {len(active_train_combos)} combos for this stage")
        max_combos = current_stage.get('max_train_combos',
                      curriculum_config.get('max_train_combos_per_stage'))
        if max_combos is not None and len(active_train_combos) > max_combos:
            rng2 = np.random.RandomState(seed + current_stage_idx)
            active_train_combos = list(rng2.choice(
                active_train_combos, size=max_combos, replace=False))
            print(f"  Capped to {max_combos} random combos")
        # Apply any reward overrides for this stage
        epoch_config = config.copy()
        if current_stage.get('reward_override'):
            epoch_reward = {**config.get('reward', {}), **current_stage['reward_override']}
            epoch_config = {**config, 'reward': epoch_reward}
    else:
        active_train_combos = train_combos
        epoch_config = config

    for epoch in range(start_epoch, num_epochs):
        # Check if we need to advance to next curriculum stage
        if curriculum_enabled:
            stages = curriculum_config.get('stages', [])
            stage_epochs = stages[current_stage_idx]['epochs']
            if current_stage_epoch >= stage_epochs:
                # Check progression criteria before advancing
                progression_type = curriculum_config.get('progression', {}).get('type', 'epoch')
                reward_threshold = curriculum_config.get('progression', {}).get('reward_threshold', 0.0)
                can_advance = False
                if progression_type == 'epoch':
                    can_advance = True
                elif progression_type in ('reward', 'both'):
                    recent_rewards = [s['avg_reward'] for s in all_stats[-5:] if 'avg_reward' in s]
                    avg_recent = np.mean(recent_rewards) if recent_rewards else -999.0
                    can_advance = avg_recent >= reward_threshold
                    if not can_advance:
                        print(f"\nStage {current_stage_idx+1} kept: avg reward "
                              f"{avg_recent:.4f} < threshold {reward_threshold}")
                else:
                    can_advance = True

                if can_advance and current_stage_idx + 1 < len(stages):
                    # Per-stage archiving (runs in background while next stage trains)
                    archive_config = config.get('archive', {})
                    if archive_config.get('enabled', False) and archive_config.get('per_stage', False):
                        prev_stage_name = stages[current_stage_idx]['name']
                        print(f"\n=== Archiving Stage {current_stage_idx+1}: {prev_stage_name} ===")
                        base_archive_dir = Path(archive_config.get('archive_dir', 'archives'))
                        stage_archive_dir = base_archive_dir / f"stage_{current_stage_idx+1}_{prev_stage_name}"
                        stage_archive_dir.mkdir(parents=True, exist_ok=True)
                        print(f"  Archiving {len(active_train_combos)} combinations")
                        archive_script = stage_archive_dir / 'archive_stage.sh'
                        with open(archive_script, 'w') as _af:
                            _af.write("#!/bin/bash\n")
                            _af.write(f"# Archive script for stage {current_stage_idx+1}: {prev_stage_name}\n\n")
                            for _cp in active_train_combos:
                                _cd = Path(_cp)
                                if _cd.exists():
                                    _tf = stage_archive_dir / f"{_cd.name}.tar.gz"
                                    _af.write(f"echo 'Archiving {_cd.name}...'\n")
                                    _af.write(f"tar -czf {_tf} -C {_cd.parent} {_cd.name}\n")
                                    if archive_config.get('remove_after', False):
                                        _af.write(f"rm -rf {_cd}\n")
                                    _af.write(f"echo '  -> {_tf}'\n\n")
                        archive_script.chmod(0o755)
                        log_file = stage_archive_dir / 'archive.log'
                        print(f"  Launching background archive, logging to {log_file}")
                        with open(log_file, 'w') as _log:
                            _proc = subprocess.Popen(
                                ['bash', str(archive_script)],
                                stdout=_log, stderr=subprocess.STDOUT
                            )
                        active_archive_jobs.append((_proc, prev_stage_name, stage_archive_dir))
                        print(f"  Archive job started (PID: {_proc.pid})")

                    current_stage_idx += 1
                    current_stage_epoch = 0
                    current_stage = stages[current_stage_idx]
                    print(f"\n{'='*60}")
                    print(f"=== Stage {current_stage_idx+1}/{len(stages)}: {current_stage['name']} ===")
                    print(f"{'='*60}")
                    active_train_combos = filter_combos_by_curriculum(
                        train_combos,
                        min_sites=current_stage['min_sites'],
                        max_sites=current_stage['max_sites'],
                        min_subs_per_site=current_stage.get('min_subs_per_site', 1),
                        max_subs_per_site=current_stage.get('max_subs_per_site', None)
                    )
                    print(f"  Filtered to {len(active_train_combos)} combos for this stage")
                    max_combos = current_stage.get('max_train_combos',
                                  curriculum_config.get('max_train_combos_per_stage'))
                    if max_combos is not None and len(active_train_combos) > max_combos:
                        rng2 = np.random.RandomState(seed + current_stage_idx)
                        active_train_combos = list(rng2.choice(
                            active_train_combos, size=max_combos, replace=False))
                        print(f"  Capped to {max_combos} random combos")
                    epoch_config = config.copy()
                    if current_stage.get('reward_override'):
                        epoch_reward = {**config.get('reward', {}), **current_stage['reward_override']}
                        epoch_config = {**config, 'reward': epoch_reward}

        stage_label = (f"Stage {current_stage_idx+1}/{len(stages)} "
                       f"({stages[current_stage_idx]['name']}) "
                       f"epoch {current_stage_epoch+1}/{stages[current_stage_idx]['epochs']}"
                       ) if curriculum_enabled else f"epoch {epoch}"
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{num_epochs-1}  [{stage_label}]")
        print(f"{'='*80}")

        stats = train_epoch(
            encoder, policy, value_network, rl_optimizer, value_optimizer,
            deepset_model, active_train_combos, epoch, epoch_config, device,
            q_network=q_network, q_optimizer=q_optimizer,
            warmstart_mapper=warmstart_mapper, warmstart_epoch=warmstart_epoch,
        )
        all_stats.append(stats)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Policy Loss: {stats['loss']:.4f}")
        print(f"  Value Loss: {stats['value_loss']:.4f}")
        print(f"  Avg Reward: {stats['avg_reward']:.4f}")

        if curriculum_enabled:
            current_stage_epoch += 1

        # Save checkpoint
        if output_config.get('save_checkpoints', True):
            if epoch % output_config.get('checkpoint_freq', 5) == 0:
                checkpoint_path = checkpoint_dir / f'checkpoint_{epoch:03d}.pt'
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'policy_state_dict': policy.state_dict(),
                    'value_state_dict': value_network.state_dict(),
                    'q_network_state_dict': q_network.state_dict(),
                    'deepset_state_dict': deepset_model.state_dict(),
                    'rl_optimizer_state_dict': rl_optimizer.state_dict(),
                    'value_optimizer_state_dict': value_optimizer.state_dict(),
                    'q_optimizer_state_dict': q_optimizer.state_dict(),
                    'stats': stats
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n=== Training Complete ===")

    # Wait for any background archive jobs
    if active_archive_jobs:
        print("\n=== Waiting for Background Archive Jobs ===")
        for _proc, _stage_name, _archive_dir in active_archive_jobs:
            print(f"  Waiting for stage '{_stage_name}' archive (PID: {_proc.pid})...")
            try:
                _proc.wait(timeout=300)
                if _proc.returncode == 0:
                    print(f"    Archive '{_stage_name}' completed successfully")
                else:
                    print(f"    Archive '{_stage_name}' failed (exit code: {_proc.returncode})")
                print(f"    Log: {_archive_dir}/archive.log")
            except subprocess.TimeoutExpired:
                print(f"    Archive '{_stage_name}' still running (continuing anyway)")
                _proc.terminate()

    # Final archive for non-per-stage archiving
    archive_config = config.get('archive', {})
    if archive_config.get('enabled', False) and not archive_config.get('per_stage', False):
        print("\n=== Archiving Combinations ===")
        from mllf.file_handling.generate_combinations import archive_combo_dirs
        import shutil
        combo_base = out_dir
        pattern = archive_config.get('pattern', 'comb_*')
        remove_after = archive_config.get('remove_after', False)
        _archive_dir = Path(archive_config.get('archive_dir', str(combo_base / 'archives')))
        _archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"Archiving combinations from {combo_base} matching '{pattern}'")
        archived = archive_combo_dirs(combo_base, pattern=pattern, remove=remove_after)
        if _archive_dir != combo_base:
            for _ap in archived:
                _dp = _archive_dir / _ap.name
                print(f"  Moving {_ap.name} to {_archive_dir}")
                shutil.move(str(_ap), str(_dp))
        print(f"Archived {len(archived)} combination directories")
        if remove_after:
            print("  Removed original directories after archiving")

    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value_network.state_dict(),
        'q_network_state_dict': q_network.state_dict(),
        'deepset_state_dict': deepset_model.state_dict(),
        'rl_optimizer_state_dict': rl_optimizer.state_dict(),
        'value_optimizer_state_dict': value_optimizer.state_dict(),
        'q_optimizer_state_dict': q_optimizer.state_dict(),
        'all_stats': all_stats
    }, final_path)
    print(f"Saved final model: {final_path}")


if __name__ == '__main__':
    main()
