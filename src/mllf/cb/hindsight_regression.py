"""Full-network online update by regression on hindsight targets.

The NeuralLinear+TS closed-form update (``UnimolPolicy.update_bayesian_posteriors``)
only moves each bias head's last linear layer. This module instead takes
gradient steps on the whole policy -- every head's trunk plus its output
weights -- toward the hindsight targets of the epoch's runs
(``mllf.cb.hindsight``): the biases each run measured it should have used,
each weighted by its own variance. Pretraining runs are replayed in the same
loss (with data hygiene) so that fitting the new pairs does not overwrite what
the model already knows.

After the steps, each Bayesian head's posterior mean is set to the trained
output weights while its precision matrix is kept, so Thompson sampling and
uncertainty-driven selection keep working with the same spread.

Selected with ``bandit.shared_update: regression`` (default ``posterior``, the
closed-form update); settings under ``bandit.regression``.
"""
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from mllf.cb.hindsight import BIAS_DIMS, OFFSET_DIMS

DEFAULTS = {
    'steps': 100,              # gradient steps per epoch
    'lr': 3e-5,
    'batch_online': 32,        # online graphs per step
    'batch_replay': 32,        # pretraining graphs per step
    'replay_weight': 0.5,      # loss = (1 - w) * online + w * replay
    'online_history': 5,       # epochs of online targets kept in the buffer
    'replay_runs': 2000,       # pretraining runs sampled for replay
    'replay_manifest': None,   # pretraining_manifest.json of the pretrained model
    'replay_consensus': None,  # environment_consensus.json of the pretrained model
    'replay_cache': None,      # where the built replay set is saved (torch)
    'embedding_cache_dir': None,
    'structure_type': 'unrelaxed',
    'grad_clip': 1.0,
}


def regression_config(bandit_cfg: Dict) -> Dict:
    cfg = dict(DEFAULTS)
    cfg.update(bandit_cfg.get('regression') or {})
    return cfg


# ── Replay set (pretraining runs) ────────────────────────────────────────────

def _load_consensus(path: Optional[str], prep_dirs: set) -> Optional[Dict]:
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for prep, sites in raw.items():
        if prep not in prep_dirs:
            continue
        out[prep] = {s: (None if atoms is None else {tuple(a) for a in atoms})
                     for s, atoms in sites.items()}
    return out


def build_replay_set(cfg: Dict, seed: int = 0, verbose: bool = True) -> List[Tuple]:
    """Sample pretraining runs and build (x, edge_index, targets [E,4], weights [E,4]).

    Weights are the data-hygiene pairwise weights (balance x measured mask), so
    only measured targets count. Cached to ``cfg['replay_cache']``.
    """
    cache = cfg.get('replay_cache')
    if cache and Path(cache).exists():
        data = torch.load(cache, weights_only=False)
        if verbose:
            print(f"  [regression] replay set loaded from {cache}: {len(data)} runs")
        return data

    from mllf.cb.graph_utils import build_directed_pairs
    from mllf.cb.pretrain_policy import (
        _compute_unimol_embeddings_and_edges,
        _extract_targets_from_variables,
        compute_pairwise_confidence_weights,
        compute_pairwise_measured_mask,
        load_pretraining_runs,
    )
    manifest = cfg.get('replay_manifest')
    if not manifest or not Path(manifest).exists():
        raise FileNotFoundError(f"bandit.regression.replay_manifest not found: {manifest}")
    # Manifest entries are combo/run-group directories, written relative to
    # where pretraining was launched; resolve them against the manifest's
    # ancestors and skip ones that no longer exist.
    raw_dirs = json.load(open(manifest))['directories']
    bases = [Path.cwd()] + list(Path(manifest).resolve().parents)

    def _resolve(d):
        p = Path(d)
        if p.is_absolute():
            return p if p.exists() else None
        for b in bases:
            if (b / p).exists():
                return (b / p).resolve()
        return None
    rng = random.Random(seed)

    # Round-robin over datasets (the directory under pretraining/), one run
    # per combo directory, so no single large dataset dominates the replay.
    by_ds: Dict[str, List[str]] = {}
    for d in raw_dirs:
        parts = Path(d).parts
        ds = parts[parts.index('pretraining') + 1] if 'pretraining' in parts else parts[0]
        by_ds.setdefault(ds, []).append(d)
    for v in by_ds.values():
        rng.shuffle(v)
    chosen = []
    datasets = sorted(by_ds)
    while len(chosen) < cfg['replay_runs'] and any(by_ds.values()):
        for ds in datasets:
            if len(chosen) >= cfg['replay_runs'] or not by_ds[ds]:
                continue
            path = _resolve(by_ds[ds].pop())
            if path is None:
                continue
            try:
                runs = load_pretraining_runs(path)
            except Exception:
                continue
            rng.shuffle(runs)
            for run in runs:
                ns = run['metadata'].get('nsubs_per_site')
                if not ns:
                    continue
                pairs = build_directed_pairs(ns)
                w = compute_pairwise_confidence_weights(run['sim_results'], ns, pairs)
                if not w or not any(x > 0 for x in w):
                    continue
                mask = compute_pairwise_measured_mask(run['sim_results'], pairs, w)
                weights = [[wi * m for m in row] for wi, row in zip(w, mask)]
                chosen.append((run, ns, pairs, weights))
                break
    rng.shuffle(chosen)
    chosen = chosen[:cfg['replay_runs']]
    prep_dirs = {str(Path(r['run_dir']).parent / 'prep') for r, *_ in chosen}
    consensus = _load_consensus(cfg.get('replay_consensus'), prep_dirs)

    data = []
    for n, (run, ns, pairs, weights) in enumerate(chosen):
        run_dir = Path(run['run_dir'])
        try:
            emb, ei, ns2, _ = _compute_unimol_embeddings_and_edges(
                run_dir, run_dir / 'graph_info.json',
                structure_type=cfg['structure_type'], use_environment_difference=True,
                consensus_dict=consensus, cache_dir=cfg.get('embedding_cache_dir'), run_idx=n)
            targets = _extract_targets_from_variables(run_dir, ns2, build_directed_pairs(ns2))
            if not targets or len(targets) != ei.size(1) or len(weights) != ei.size(1):
                continue
            data.append((emb.cpu(), ei.cpu(), torch.tensor(targets, dtype=torch.float32),
                         torch.tensor(weights, dtype=torch.float32)))
        except Exception:
            continue
        if verbose and (n + 1) % 200 == 0:
            print(f"  [regression] replay: built {len(data)}/{n + 1}")
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache)
    if verbose:
        print(f"  [regression] replay set built: {len(data)} runs"
              + (f", saved to {cache}" if cache else ""))
    return data


# ── Online buffer ────────────────────────────────────────────────────────────

def online_item(x: torch.Tensor, edge_index: torch.Tensor, targets: torch.Tensor,
                var: torch.Tensor) -> Optional[Tuple]:
    """(x, edge_index, targets [E,4], weights [E,4]) with weights = 1/var (0 if inf)."""
    w = torch.where(torch.isfinite(var), 1.0 / var.clamp(min=1e-6), torch.zeros_like(var))
    if not (w > 0).any():
        return None
    return (x.detach().cpu(), edge_index.detach().cpu(),
            targets.detach().cpu().float(), w.cpu().float())


# ── Regression step ──────────────────────────────────────────────────────────

def _predict(policy, x, edge_index, ws):
    """Physical-unit means [E, D] from trunks + trainable output weights ``ws``."""
    z = policy.forward_features(x, edge_index, None)          # [E, D, F]
    cols = []
    for d in range(policy.mlp_out_dim):
        zd = z[:, d, :]
        z_aug = torch.cat([zd, torch.ones(zd.shape[0], 1, dtype=zd.dtype, device=zd.device)], -1)
        cols.append(policy._preact_to_action(z_aug @ ws[d], d))
    return torch.stack(cols, dim=-1)


def _weighted_sq(pred, targets, weights, scale):
    """Per-dim weighted mean squared error / scale^2, summed over dims."""
    loss = pred.new_zeros(())
    for d in range(pred.shape[1]):
        w = weights[:, d]
        if w.sum() <= 0:
            continue
        loss = loss + (w * (pred[:, d] - targets[:, d]) ** 2).sum() / w.sum() / scale[d] ** 2
    return loss


def _batch_loss(policy, items, ws, scale, device):
    """Weighted squared error pooled over the batch's targets (each target counts
    by its own weight, not by which graph it came from), per dim / scale^2."""
    if not items:
        return None
    preds, targets, weights = [], [], []
    for x, ei, t, w in items:
        preds.append(_predict(policy, x.to(device), ei.to(device), ws))
        targets.append(t.to(device))
        weights.append(w.to(device))
    return _weighted_sq(torch.cat(preds), torch.cat(targets), torch.cat(weights), scale)


def residual_scale(policy, replay: List[Tuple], device, n: int = 200,
                   seed: int = 0) -> torch.Tensor:
    """Per-dim weighted RMS residual of the current model on replay data: the
    unit each dim's error is measured in, so the four heads count comparably."""
    D = policy.mlp_out_dim
    num = torch.zeros(D)
    den = torch.zeros(D)
    with torch.no_grad():
        ws = [policy.edge_mlp.mlps[d].bayesian_head.mu for d in range(D)]
        # A random sample of the whole pool (not its first n items: the online
        # buffer is ordered oldest-first, so those would not represent it).
        items = random.Random(seed).sample(replay, min(n, len(replay)))
        for x, ei, t, w in items:
            pred = _predict(policy, x.to(device), ei.to(device), ws).cpu()
            num += (w * (pred - t) ** 2).sum(0)
            den += w.sum(0)
    return (num / den.clamp(min=1e-9)).sqrt().clamp(min=0.05)


def regression_step(policy, online: List[Tuple], replay: List[Tuple], cfg: Dict,
                    device: str = 'cpu', seed: int = 0,
                    scale: Optional[torch.Tensor] = None) -> Dict:
    """Gradient steps on trunks + output weights; re-seed the Bayesian heads.

    Each source's error is measured in its own unit: per dim, its weighted RMS
    residual under the model at the start of the step (``residual_scale``).
    Pretraining targets scatter far more around the model than online
    hindsight targets, so one shared unit would let replay swamp the online
    signal; with separate units both terms start near 1 per dim and
    ``replay_weight`` sets their balance. ``scale`` (tests) overrides both.

    Returns stats (losses before/after, steps).
    """
    if not online:
        return {'steps': 0}
    rng = random.Random(seed)
    D = policy.mlp_out_dim
    heads = [policy.edge_mlp.mlps[d].bayesian_head for d in range(D)]
    ws = [torch.nn.Parameter(h.mu.detach().clone()) for h in heads]
    trunk_params = [p for d in range(D) for p in policy.edge_mlp.mlps[d].trunk.parameters()]
    req = [p.requires_grad for p in trunk_params]
    for p in trunk_params:
        p.requires_grad_(True)
    if scale is not None:
        scale_on = scale_rp = scale.to(device)
    else:
        scale_on = residual_scale(policy, online, device, seed=seed).to(device)
        scale_rp = (residual_scale(policy, replay, device, seed=seed) if replay
                    else torch.ones(D)).to(device)
    opt = torch.optim.Adam(trunk_params + ws, lr=float(cfg['lr']))
    rw = float(cfg['replay_weight']) if replay else 0.0

    def _sample(pool, k):
        return rng.sample(pool, min(k, len(pool))) if pool else []

    with torch.no_grad():
        eval_on = _sample(online, 64)
        eval_rp = _sample(replay, 64)
        before = (float(_batch_loss(policy, eval_on, ws, scale_on, device)),
                  float(_batch_loss(policy, eval_rp, ws, scale_rp, device)) if eval_rp else float('nan'))
    policy.train()
    for _ in range(int(cfg['steps'])):
        opt.zero_grad()
        l_on = _batch_loss(policy, _sample(online, int(cfg['batch_online'])), ws, scale_on, device)
        loss = (1.0 - rw) * l_on
        if rw > 0:
            loss = loss + rw * _batch_loss(policy, _sample(replay, int(cfg['batch_replay'])),
                                            ws, scale_rp, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk_params + ws, float(cfg['grad_clip']))
        opt.step()
    with torch.no_grad():
        after = (float(_batch_loss(policy, eval_on, ws, scale_on, device)),
                 float(_batch_loss(policy, eval_rp, ws, scale_rp, device)) if eval_rp else float('nan'))
    for p, r in zip(trunk_params, req):
        p.requires_grad_(r)
    # Each head's posterior mean becomes the trained weights; its precision
    # matrix (calibrated in pretraining, saved in the checkpoint) is kept, so
    # Thompson sampling keeps its spread. Do NOT reset it from the head's
    # _prior_precision attribute: that isn't saved in checkpoints, and after a
    # load it is the config's prior_precision (1.0), which made sampling
    # spreads ~300x too wide.
    with torch.no_grad():
        for h, w in zip(heads, ws):
            mu = w.detach().to(h.mu.dtype)
            h.mu = mu
            h.Lambda_mu = h.Lambda @ mu
    return {'steps': int(cfg['steps']), 'online_before': before[0], 'online_after': after[0],
            'replay_before': before[1], 'replay_after': after[1]}
