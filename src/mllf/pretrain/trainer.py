"""Behavior-cloning pretraining: the AWR epoch and the full training run."""
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from mllf.cb.pairs import build_directed_pairs
from mllf.cb.policy import UnimolPolicy
from mllf.cb.warning_summary import WarningSummary
from mllf.log import log_to_file
from mllf.pretrain.bayesian_init import (
    estimate_reward_noise_variance,
    initialize_bayesian_heads_from_pretrained,
)
from mllf.pretrain.config import PretrainOptions, UnimolSettings
from mllf.pretrain.features import (
    RunInputs,
    build_pairwise_weights,
    build_run_inputs,
    compute_unimol_embeddings_and_edges,
    extract_targets_from_variables,
)
from mllf.pretrain.filters import select_runs
from mllf.pretrain.rewards import assign_run_rewards, build_pair_advantages


log = logging.getLogger(__name__)

# Every message of a pretraining run, with time stamps, in its output directory.
LOG_FILE = 'pretrain.log'


def _awr_run_loss(w, logp_per, combined_mask, active_mask, w_sum, data_hygiene):
    """One run's AWR behaviour-cloning loss.

    Legacy: a weighted mean over the run's kept targets (divided by the sum of
    weights), so every run counts equally and the weights only rank targets
    within a run -- which is why run-level filters were needed. Data hygiene:
    divided by the run's number of active targets instead, so a run whose
    targets are unmeasured (weight 0) or poorly resolved (low AWR/balance
    weight) contributes proportionally less, and no run filter is needed.
    """
    total = -(w * logp_per)[combined_mask].sum()
    if data_hygiene:
        return total / active_mask.sum().clamp(min=1)
    return total / w_sum


def _edge_confidence(pairwise_weights_cache, run_idx, logp_per, device, data_hygiene):
    """Per-edge (or per-edge, per-bias-type) confidence for one run's BC loss.

    Returns a tensor broadcastable to ``logp_per`` ([E, 1] or [E, D]), or None
    when the run should be skipped: under data hygiene a run with no pairwise
    weights (no usable populations) has no measured targets, where the legacy
    behaviour gives every edge full weight.
    """
    weights = pairwise_weights_cache[run_idx] if pairwise_weights_cache is not None else None
    if weights is None:
        if data_hygiene:
            return None
        return torch.ones((logp_per.size(0), 1), dtype=torch.float32, device=device)
    conf = torch.tensor(weights, dtype=torch.float32, device=device)
    return conf.unsqueeze(-1) if conf.dim() == 1 else conf


def pretrain_epoch(
    policy: nn.Module,
    optimizer: optim.Optimizer,
    runs: List[Dict],
    reward_config: Dict,
    device: torch.device,
    graph_cache: Optional[List] = None,
    groups: Optional[Dict] = None,
    reward_weighted: bool = False,
    awr_temperature: float = 0.5,
    pairwise_weights_cache: Optional[List] = None,
    unimol_config: Optional[Dict] = None,
    consensus_dict: Optional[Dict[str, Optional[set]]] = None,
    pair_advantage_cache: Optional[List] = None,
    system_emphasis: Optional[Dict[str, float]] = None,
    data_hygiene: bool = False,
) -> Dict[str, float]:
    """Run one behavior cloning AWR epoch for UnimolPolicy with optional pairwise weighting.

    Args:
        policy: UnimolPolicy model
        optimizer: Optimizer
        runs: List of run dicts with 'run_dir', targets, etc.
        reward_config: Reward configuration dict
        device: Torch device
        graph_cache: Optional pre-built graph cache (list of tuples)
        groups: Optional run grouping for grouped updates
        reward_weighted: Whether to use AWR weighting
        awr_temperature: Temperature for AWR reweighting
        pairwise_weights_cache: Optional per-run pairwise edge weights (confidence in [0,1]),
                        either [E] or, under data hygiene, [E][4] per bias type
        data_hygiene: If True, runs with no pairwise weights are skipped instead of
                        getting full weight on every edge (see
                        ``compute_pairwise_measured_mask``).
        unimol_config: Optional dict with 'environment_cutoff' and 'use_environment_difference'
        pair_advantage_cache: Optional per-run [E, 4] per-edge, per-dimension AWR
                        advantage (``pair_advantages``: compute_pair_reward(), the reward
                        online training uses, normalised over all runs and gated by the
                        online success test for the run's length). The AWR weight is then
                        per-edge/per-dimension instead of one scalar for the whole run, so
                        pairs with better bias coefficients within an otherwise-successful
                        run get emphasised more than pairs that were poorly resolved. Runs
                        whose entry is None fall back to the whole-run scalar
                        ``_bc_reward`` weight.
        system_emphasis: Optional {system_name_substring: weight} dict to up-weight
                        specific (typically under-represented) systems' runs, matched
                        case-insensitively against each run's 'system_label'. See
                        ``pretrain_with_runs`` for how this combines with reward_weighted.

    Returns:
        Dict with 'loss', 'num_runs', 'num_updates' keys
    """
    if unimol_config is None:
        unimol_config = {}

    env_cutoff = unimol_config.get('environment_cutoff', 8.0)
    use_environment_difference = unimol_config.get('use_environment_difference', True)
    policy.train()

    # Build normalised reward weights (mean = 1.0) so the learning rate is unchanged.
    if reward_weighted:
        raw_w = [max(0.0, run.get("_bc_reward", 0.0)) for run in runs]
        total_w = sum(raw_w)
        if total_w > 1e-10:
            n_runs = len(runs)
            norm_weights = [w * n_runs / total_w for w in raw_w]
        else:
            reward_weighted = False
            norm_weights = None
    else:
        norm_weights = None

    # Per-system emphasis: up-weight runs from named (typically under-
    # represented) systems so they drive proportionally more gradient and get
    # fit more accurately, independent of reward_weighted. Matched by
    # case-insensitive substring against each run's 'system_label' (set in
    # main() from the --pretraining-dir basename). Combined multiplicatively
    # with the reward-based weight above, then renormalised to mean=1.0 so
    # the effective learning rate stays comparable to an unweighted epoch.
    if system_emphasis:
        base_w = norm_weights if norm_weights is not None else [1.0] * len(runs)
        emphasis_w = []
        for run in runs:
            label = run.get('system_label', '').lower()
            w = 1.0
            for pattern, weight in system_emphasis.items():
                if pattern.lower() in label:
                    w = weight
                    break
            emphasis_w.append(w)
        combined = [b * e for b, e in zip(base_w, emphasis_w)]
        total_c = sum(combined)
        if total_c > 1e-10:
            norm_weights = [c * len(runs) / total_c for c in combined]

    epoch_loss = 0.0
    num_updates = 0
    total_runs = 0

    if groups is not None and graph_cache is not None:
        # Per-group gradient accumulation (one step per unique Uni-Mol embedding set)
        for group_key, group_indices in groups.items():
            valid = [
                (i, norm_weights[i] if norm_weights is not None else 1.0)
                for i in group_indices
                if (norm_weights is None or norm_weights[i] >= 1e-8)
                and graph_cache[i] is not None
            ]
            if not valid:
                continue
            n_valid = len(valid)
            optimizer.zero_grad()
            group_loss = 0.0

            for run_idx, rw in valid:
                unimol_emb, edge_idx, targets_list = graph_cache[run_idx]
                targets = torch.tensor(targets_list, dtype=torch.float32, device=device)

                # Move embeddings and edge_index to device
                unimol_emb = unimol_emb.to(device)
                edge_idx = edge_idx.to(device)

                # Forward pass through UnimolPolicy
                mean, log_std = policy._forward_edges(unimol_emb, edge_idx)

                active_mask = targets.abs() > 1e-8
                if not active_mask.any():
                    continue

                # AWR loss: -exp(A/β) · log π(a | s) over active elements.
                std = torch.exp(log_std)
                logp_per = torch.distributions.Normal(mean, std).log_prob(targets)  # [E, D]

                # Per-edge confidence weight (population balance + DDG validity).
                edge_conf = _edge_confidence(pairwise_weights_cache, run_idx, logp_per, device,
                                             data_hygiene)
                if edge_conf is None:
                    continue

                advantage = (pair_advantage_cache[run_idx] if pair_advantage_cache is not None
                             else None)
                if advantage is not None and advantage.shape[0] == logp_per.shape[0]:
                    # Per-edge, per-dimension AWR advantage (pair_advantages):
                    # pairs with better bias coefficients within an
                    # otherwise-successful run get emphasised more than pairs
                    # that were poorly resolved, instead of every pair in the
                    # run sharing one scalar weight.
                    awr_w = torch.clamp(
                        torch.exp(
                            advantage.to(device) /
                            awr_temperature),
                        max=20.0)  # [E, D]
                else:
                    # Fall back to the whole-run scalar reward when per-pair
                    # data is unavailable for this run (e.g. older collected
                    # runs missing ddg_pairs/populations).
                    raw_r = runs[run_idx].get("_bc_reward", 0.0)
                    awr_w = torch.full_like(logp_per, min(math.exp(raw_r / awr_temperature), 20.0))

                w = awr_w * edge_conf  # [E, D]
                combined_mask = active_mask & (edge_conf > 1e-8)
                w_sum = w[combined_mask].sum() if combined_mask.any(
                ) else torch.tensor(0.0, device=device)
                if combined_mask.any() and w_sum > 1e-8:
                    run_loss = rw * _awr_run_loss(w, logp_per, combined_mask, active_mask,
                                                  w_sum, data_hygiene) / n_valid
                else:
                    continue

                if torch.isnan(run_loss) or torch.isinf(run_loss):
                    continue
                run_loss.backward()
                group_loss += run_loss.item()

            if group_loss > 0.0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += group_loss
                num_updates += 1
                total_runs += n_valid
            else:
                optimizer.zero_grad()

        avg_loss = epoch_loss / num_updates if num_updates > 0 else 0.0
        return {'loss': avg_loss, 'num_runs': total_runs, 'num_updates': num_updates}

    # Legacy per-run updates (when graph_cache provided but groups not)
    for run_idx, run in enumerate(runs):
        rw = norm_weights[run_idx] if norm_weights is not None else 1.0
        if rw < 1e-8:
            continue

        if graph_cache is not None:
            cached = graph_cache[run_idx]
            if cached is None:
                continue
            unimol_emb, edge_idx, targets_list = cached
            targets = torch.tensor(targets_list, dtype=torch.float32, device=device)
        else:
            # Fallback: compute embeddings on-the-fly (slow)
            run_dir = Path(run["run_dir"])
            gi_path = run_dir / "graph_info.json"
            try:
                unimol_emb, edge_idx, nsubs_per_site, _ = compute_unimol_embeddings_and_edges(
                    run_dir, gi_path,
                    env_cutoff=env_cutoff,
                    use_environment_difference=use_environment_difference,
                    consensus_dict=consensus_dict,
                    env_input=unimol_config.get('env_input', 'dual'),
                    ligand_other_sites=bool(unimol_config.get('ligand_other_sites', False)),
                    site_size_input=bool(unimol_config.get('site_size_input', False)),
                )
                pairs = build_directed_pairs(nsubs_per_site)
                targets_list = extract_targets_from_variables(run_dir, nsubs_per_site, pairs)
                if not targets_list:
                    continue
                targets = torch.tensor(targets_list, dtype=torch.float32, device=device)
            except Exception as e:
                log.error(f"  Error computing Uni-Mol embeddings for {run_dir.name}: {e}")
                continue

        # Move to device
        unimol_emb = unimol_emb.to(device)
        edge_idx = edge_idx.to(device)

        # Forward pass
        mean, log_std = policy._forward_edges(unimol_emb, edge_idx)

        active_mask = targets.abs() > 1e-8
        if not active_mask.any():
            continue

        # Compute AWR loss
        optimizer.zero_grad()
        std = torch.exp(log_std)
        logp_per = torch.distributions.Normal(mean, std).log_prob(targets)

        # Per-edge confidence weight (population balance + DDG validity).
        edge_conf = _edge_confidence(pairwise_weights_cache, run_idx,
                                     logp_per, device, data_hygiene)
        if edge_conf is None:
            continue

        advantage = pair_advantage_cache[run_idx] if pair_advantage_cache is not None else None
        if advantage is not None and advantage.shape[0] == logp_per.shape[0]:
            awr_w = torch.clamp(
                torch.exp(
                    advantage.to(device) /
                    awr_temperature),
                max=20.0)  # [E, D]
        else:
            raw_r = run.get("_bc_reward", 0.0)
            awr_w = torch.full_like(logp_per, min(math.exp(raw_r / awr_temperature), 20.0))

        w = awr_w * edge_conf  # [E, D]
        combined_mask = active_mask & (edge_conf > 1e-8)
        w_sum = w[combined_mask].sum() if combined_mask.any() else torch.tensor(0.0, device=device)
        if combined_mask.any() and w_sum > 1e-8:
            loss = rw * _awr_run_loss(w, logp_per, combined_mask, active_mask, w_sum, data_hygiene)
        else:
            continue

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_updates += 1
            total_runs += 1

    avg_loss = epoch_loss / num_updates if num_updates > 0 else 0.0
    return {'loss': avg_loss, 'num_runs': total_runs, 'num_updates': num_updates}


RELATION_NAMES = ['linear_ij', 'linear_ji', 'quadratic_ij', 'quadratic_ji',
                  'skew_ij', 'skew_ji', 'end_ij', 'end_ji']


def build_policy(config: Dict, unimol: UnimolSettings, device) -> UnimolPolicy:
    """The deterministic UnimolPolicy described by ``training.unimol`` and ``unimol``."""
    policy_config = config.get('training', {}).get('unimol', {})
    return UnimolPolicy(
        unimol_dim=policy_config.get('unimol_dim', 512),
        mlp_hidden=policy_config.get('mlp_hidden', 64),
        mlp_out_dim=len(RELATION_NAMES) // 2,
        use_dual_embeddings=unimol.use_environment_difference,
        env_input=unimol.env_input,
        site_size_input=unimol.site_size_input,
    ).to(device)


def _print_setup(unimol: UnimolSettings) -> None:
    # Fixed schema: 4 bias types (linear, quadratic, skew, end), 2 directions each.
    log.debug("\nSchema for Uni-Mol pretraining:")
    log.debug("  Bias types: 4 (linear, quadratic, skew, end)")
    log.debug(f"  Relations: {len(RELATION_NAMES)} (2 directions per type)")
    log.debug(f"  Relation names: {RELATION_NAMES}")
    log.debug(f"Environment input: {unimol.env_input}")
    log.debug(f"Ligand other-site reference substituents: {unimol.ligand_other_sites}")
    log.debug(f"Site-size input: {unimol.site_size_input}")
    log.debug(f"Ligand geometry from minimized.pdb for minimized runs: "
              f"{unimol.ligand_from_minimized}")


def _print_model(policy, unimol: UnimolSettings, opts: PretrainOptions, n_runs: int) -> None:
    lr = opts.learning_rate
    log.info(f"\nPolicy: {sum(p.numel() for p in policy.parameters())} parameters, node input "
             f"{unimol.env_input}; learning rate {lr} (cosine annealed to {lr / 100:.6f}), "
             f"early stopping after {opts.patience} epochs without improvement")
    log.debug(f"  Graphs for all {n_runs} runs are built once and reused across epochs")


@dataclass
class _Progress:
    """Early-stopping state (restored on resume)."""
    start_epoch: int = 0
    best_loss: float = float('inf')
    epochs_without_improvement: int = 0


def _resume(path: Path, policy, optimizer, scheduler, output_dir: Path, device) -> _Progress:
    """Restore model, optimizer and LR schedule from a checkpoint_epoch_NNN.pt,
    and early stopping from the run's best_policy.pt."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt['policy_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    progress = _Progress(start_epoch=int(ckpt['epoch']))
    best_path = output_dir / "best_policy.pt"
    if best_path.exists():
        best = torch.load(best_path, map_location='cpu', weights_only=False)
        progress.best_loss = float(best['loss'])
        progress.epochs_without_improvement = progress.start_epoch - int(best['epoch'])
    log.info(f"\nResuming from {path}: epoch {progress.start_epoch} done; "
             f"best loss {progress.best_loss:.4f}, {progress.epochs_without_improvement} epoch(s) "
             "without improvement")
    return progress


def _train_epochs(policy, optimizer, scheduler, runs, inputs: RunInputs, weights_cache,
                  advantages, reward_weighted: bool, unimol: UnimolSettings,
                  opts: PretrainOptions, reward_config: Dict, output_dir: Path, device,
                  progress: _Progress) -> None:
    """The behavior-cloning epochs: best_policy.pt on each lower loss, a
    checkpoint per epoch, early stopping after ``patience`` epochs."""
    # one run per group: pretrain_epoch's per-group gradient accumulation
    groups = {f"run_{i}": [i] for i in range(len(runs)) if inputs.graph_cache[i] is not None}
    for epoch in range(progress.start_epoch, opts.epochs):
        stats = pretrain_epoch(
            policy, optimizer, runs, reward_config, device,
            graph_cache=inputs.graph_cache,
            groups=groups,
            reward_weighted=reward_weighted,
            awr_temperature=opts.awr_temperature,
            pairwise_weights_cache=weights_cache,
            unimol_config=unimol.epoch_config(),
            consensus_dict=inputs.consensus,
            pair_advantage_cache=advantages,
            system_emphasis=opts.system_emphasis,
            data_hygiene=opts.data_hygiene,
        )
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        improved = stats['loss'] < progress.best_loss
        if improved:
            progress.best_loss = stats['loss']
            progress.epochs_without_improvement = 0
            torch.save({'policy_state': policy.state_dict(), 'epoch': epoch + 1,
                        'loss': stats['loss']}, output_dir / "best_policy.pt")
        else:
            progress.epochs_without_improvement += 1
        log.info(f"Epoch {epoch + 1}/{opts.epochs}: AWR loss {stats['loss']:.4f}, "
                 f"{stats['num_runs']} runs ({stats.get('num_updates', stats['num_runs'])} "
                 f"gradient steps), learning rate {lr:.6f}"
                 + ("; saved as best" if improved else ""))

        torch.save({'policy_state': policy.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'stats': stats}, output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")

        if progress.epochs_without_improvement >= opts.patience:
            log.info(f"\nEarly stopping: no improvement for {opts.patience} epochs (best loss: "
                     f"{progress.best_loss:.4f})")
            break


def _save_bayesian_siblings(policy, config: Dict, unimol: UnimolSettings,
                            inputs: RunInputs, weights_cache, opts: PretrainOptions,
                            output_dir: Path, device) -> None:
    """final_policy_bayesian.pt (and best_policy_bayesian.pt): the trained
    trunks with each BayesianLinearHead's posterior mean warm-started from the
    deterministic readout (``initialize_bayesian_heads_from_pretrained``) and
    its noise variance estimated from the behavior-cloning residuals. The
    deterministic checkpoints are not changed."""
    log.info("Building the Bayesian-head checkpoint(s)")
    try:
        hygiene_weights = weights_cache if opts.data_hygiene else None
        noise_vars = estimate_reward_noise_variance(
            policy, inputs.graph_cache, device, weights_cache=hygiene_weights)

        def save(det_policy, out_path, extra_fields):
            bayes_policy = initialize_bayesian_heads_from_pretrained(
                det_policy, bayesian_prior_precision=opts.bayesian_prior_precision,
                graph_cache=inputs.graph_cache, device=device,
                exploration_fraction=opts.exploration_fraction,
                weights_cache=hygiene_weights)
            for d, sigma2 in enumerate(noise_vars):
                bayes_policy.edge_mlp.mlps[d].bayesian_head.set_noise_var(sigma2)
            ckpt = {'policy_state': bayes_policy.state_dict(), 'noise_var': noise_vars}
            ckpt.update(extra_fields)
            torch.save(ckpt, out_path)
            log.info(f"  Saved {out_path}")

        save(policy, output_dir / "final_policy_bayesian.pt", {'epoch': opts.epochs})
        best_path = output_dir / "best_policy.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location=device, weights_only=False)
            best_policy = build_policy(config, unimol, device)
            best_policy.load_state_dict(best['policy_state'])
            save(best_policy, output_dir / "best_policy_bayesian.pt",
                 {'epoch': best.get('epoch'), 'loss': best.get('loss')})
    except Exception as e:
        log.warning(f"  Warning: Bayesian-head post-pass failed: {e}")


def _write_metadata(config: Dict, unimol: UnimolSettings, opts: PretrainOptions, n_runs: int,
                    n_ok: int, best_loss: float, output_dir: Path, device) -> None:
    policy_config = config.get('training', {}).get('unimol', {})
    metadata = {
        'policy_type': 'unimol',
        'unimol_dim': policy_config.get('unimol_dim', 512),
        'mlp_hidden': policy_config.get('mlp_hidden', 64),
        'env_input': unimol.env_input,
        'ligand_other_sites': unimol.ligand_other_sites,
        'site_size_input': unimol.site_size_input,
        'ligand_from_minimized': unimol.ligand_from_minimized,
        'structure_selection': unimol.structure_selection,
        'num_relations': len(RELATION_NAMES),
        'num_pretraining_runs': n_runs,
        'num_runs_with_graphs': n_ok,
        'epochs': opts.epochs,
        'best_loss': best_loss,
        'training_method': 'behavior_cloning',
        'device': str(device),
    }
    with open(output_dir / "pretrain_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def pretrain_with_runs(runs: List[Dict], output_dir: Path, config: Dict,
                       options: Optional[PretrainOptions] = None, **kwargs) -> None:
    """Behavior-clone a UnimolPolicy on collected runs (``runs.load_pretraining_runs``).

    ``config`` is the pretraining YAML (``unimol``, ``training.unimol`` and
    ``reward`` sections); ``options`` (or the same fields as keyword
    arguments) are the command-line settings, see ``PretrainOptions``.

    Writes to ``output_dir``: best_policy.pt, checkpoint_epoch_NNN.pt,
    final_policy.pt, pretrain_metadata.json, pretrain.log (every message),
    pretraining_systems.json, pretraining_representations_used.csv,
    environment_consensus.json (env_input 'dual') and, with
    ``bayesian_heads``, best/final_policy_bayesian.pt.
    """
    if options is not None and kwargs:
        raise TypeError("pass either options or keyword arguments, not both")
    opts = options or PretrainOptions(**kwargs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(opts.device if opts.device is not None
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    if len(runs) == 0:
        log.error("Error: No valid runs found")
        return

    reward_config = config.get('reward', {})
    runs = select_runs(runs, opts, reward_config)
    if runs is None:
        return
    if 'vocabulary' in config:
        log.info("  Note: the 'vocabulary' config block is no longer used and is ignored")

    unimol = UnimolSettings.from_config(config)
    _print_setup(unimol)
    policy = build_policy(config, unimol, device)
    if opts.init_checkpoint is not None:
        init = torch.load(opts.init_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(init['policy_state'])
        log.info(f"\nInitialized policy from {opts.init_checkpoint} "
                 f"(epoch {init.get('epoch')}, loss {init.get('loss')})")
    _print_model(policy, unimol, opts, len(runs))

    # Adam, cosine-annealed from lr to lr/100 over all epochs
    optimizer = optim.Adam(policy.parameters(), lr=opts.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.epochs, eta_min=opts.learning_rate / 100.0)
    progress = _Progress()
    if opts.resume_checkpoint is not None:
        progress = _resume(opts.resume_checkpoint, policy, optimizer, scheduler, output_dir,
                           device)

    # Warnings (one per run and file, often) are summarized once at the end.
    with log_to_file(output_dir / LOG_FILE), WarningSummary(show_template=True, width=120):
        inputs = build_run_inputs(runs, unimol, output_dir)
        weights_cache = build_pairwise_weights(runs, inputs.nsubs_by_system, opts.data_hygiene)
        advantages = (build_pair_advantages(runs, inputs.graph_cache, reward_config)
                      if opts.use_per_pair_awr else None)
        reward_weighted = opts.reward_weighted
        if assign_run_rewards(runs, reward_config) == 0 and reward_weighted:
            log.warning("  Warning: all rewards zero or negative; disabling reward weighting.")
            reward_weighted = False

        log.info(f"\nBehavior cloning: {opts.epochs} epochs on {len(runs)} runs "
                 f"({inputs.n_ok} with valid graphs)"
                 + ("; reward-weighted loss" if reward_weighted else ""))
        _train_epochs(policy, optimizer, scheduler, runs, inputs, weights_cache, advantages,
                      reward_weighted, unimol, opts, reward_config, output_dir, device,
                      progress)
        torch.save({'policy_state': policy.state_dict(), 'epoch': opts.epochs},
                   output_dir / "final_policy.pt")
        if opts.bayesian_heads:
            _save_bayesian_siblings(policy, config, unimol, inputs, weights_cache, opts,
                                    output_dir, device)
        _write_metadata(config, unimol, opts, len(runs), inputs.n_ok, progress.best_loss,
                        output_dir, device)
        log.info(f"\nBehavior cloning complete: best loss {progress.best_loss:.4f}; "
                 f"saved to {output_dir}")
