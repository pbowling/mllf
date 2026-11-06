"""Predictions of a pretrained model on the runs it was trained on, for
``mllf.analysis.pretrain``.

The runs are the directories of the model's ``pretraining_manifest.json``.
Each run is embedded exactly as pretraining embedded it
(``mllf.pretrain.features.build_run_inputs`` with the model's unimol settings
from ``pretrain_metadata.json``), and the model predicts every directed
pair's four coefficients (linear, quadratic, skew, end), with the posterior
predictive sd for a model with Bayesian heads.

A run's coefficient is scored only where the run's biases were good for it,
as pretraining judges them: the pair passed online training's success test
for a run of its length and, for quadratic, skew and end, the run measured
that coefficient's lambda-profile signal. By default each dataset (a
system's directory of collected runs) contributes its ``max_runs`` runs
with the most such pairs.
"""
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


log = logging.getLogger(__name__)

BIASES = ('linear', 'quadratic', 'skew', 'end')
CANDIDATES_PER_DATASET = 50       # runs read per dataset (at random) before ranking


def model_files(model_dir: Path, model: Optional[str] = None) -> Path:
    """The checkpoint to analyze: ``model``, else best_policy_bayesian.pt,
    else best_policy.pt."""
    if model:
        return Path(model)
    for name in ('best_policy_bayesian.pt', 'best_policy.pt'):
        if (model_dir / name).exists():
            return model_dir / name
    raise FileNotFoundError(f"no best_policy_bayesian.pt or best_policy.pt in {model_dir}")


def manifest_dirs(model_dir: Path) -> List[Path]:
    """The pretraining directories of the model's manifest. Relative entries
    are resolved against the working directory, the model directory and its
    parents (whichever holds the first entry)."""
    from mllf.pretrain.runs import read_manifest
    path = model_dir / 'pretraining_manifest.json'
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; the analysis reads the runs pretraining "
                                "used from it")
    dirs = read_manifest(path)
    if not dirs:
        return []
    bases = [Path.cwd(), model_dir] + list(model_dir.resolve().parents)
    first = Path(dirs[0])
    base = Path('.') if first.is_absolute() else next(
        (b for b in bases if (b / first).is_dir()), None)
    if base is None:
        raise FileNotFoundError(f"{dirs[0]} (first entry of {path}) not found relative to the "
                                "working directory or the model directory")
    # normalized as text: resolving each of tens of thousands of entries on
    # a network filesystem takes minutes
    import os
    base = Path(os.path.abspath(base))
    return [Path(d) if Path(d).is_absolute() else Path(os.path.normpath(base / d))
            for d in dirs]


def model_settings(model_dir: Path, config_path: Optional[str] = None):
    """UnimolSettings of the model: pretrain_metadata.json over the config's
    unimol block (the config supplies the embedding cache)."""
    import yaml

    from mllf.pretrain.config import UnimolSettings
    config = {}
    if config_path:
        config = yaml.safe_load(Path(config_path).read_text()) or {}
    else:
        found = sorted(model_dir.glob('*.yaml'))
        if found:
            config = yaml.safe_load(found[0].read_text()) or {}
            log.info(f"Pretraining config: {found[0]}")
    unimol = dict(config.get('unimol') or {})
    meta_path = model_dir / 'pretrain_metadata.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for key in ('env_input', 'ligand_other_sites', 'site_size_input', 'ligand_from_minimized',
                'structure_selection'):
        if key in meta:
            unimol[key] = meta[key]
    return UnimolSettings.from_config({'unimol': unimol}), {**config, 'unimol': unimol}


def load_model(path: Path, config: Dict, device: str):
    """The checkpoint's UnimolPolicy, in eval mode."""
    import torch

    from mllf.cb.training_setup import build_policy, policy_weights
    ckpt = torch.load(path, map_location=device, weights_only=False)
    weights = policy_weights(ckpt)
    if weights is None:
        raise ValueError(f"{path} holds no policy weights")
    bayes = any('bayesian_head' in k for k in weights)
    cfg = {**config, 'training': {'policy': {'use_bayesian_heads': bayes}}}
    policy = build_policy(cfg, device)
    policy.load_state_dict(weights)
    return policy.eval()


def good_mask(sim_results: Dict, pairs: List, reward_config: Dict) -> np.ndarray:
    """[E, 4] bool: whether each pair's coefficient of each bias type is a
    target to score against (module docstring)."""
    import math

    import torch

    from mllf.cb.pair_weights import MEASURED_SIGNAL
    from mllf.pretrain.rewards import run_pair_success
    mask = np.zeros((len(pairs), 4), dtype=bool)
    if not pairs:
        return mask
    ok = run_pair_success({'sim_results': sim_results},
                          torch.tensor(pairs, dtype=torch.long).T, reward_config)
    if ok is None:
        return mask

    def measured(field, lo, hi):
        vals = sim_results.get(field) or {}
        v = vals.get(f"{lo}_{hi}", vals.get(f"{hi}_{lo}"))
        try:
            return v is not None and math.isfinite(float(v))
        except (TypeError, ValueError):
            return False
    for k, (i, j) in enumerate(pairs):
        if ok[k]:
            lo, hi = min(i, j) + 2, max(i, j) + 2
            mask[k, 0] = True
            for d, field in enumerate(MEASURED_SIGNAL[1:], start=1):
                mask[k, d] = measured(field, lo, hi)
    return mask


def _scored_runs(task):
    """(good targets, run dir, run) of each run of one directory."""
    from mllf.cb.graph_utils import build_directed_pairs
    from mllf.pretrain.runs import load_pretraining_runs
    d, name, reward_config = task
    out = []
    for run in load_pretraining_runs(d):
        nsubs = (run.get('metadata') or {}).get('nsubs_per_site')
        if not nsubs:
            continue
        n_good = int(good_mask(run['sim_results'], build_directed_pairs(nsubs),
                               reward_config).sum())
        if n_good:
            run['system_label'] = name
            out.append((n_good, str(run['run_dir']), run))
    return out


def select_runs(dirs: List[Path], max_runs: Optional[int], reward_config: Dict,
                seed: int = 0, workers: int = 8) -> List[Dict]:
    """Runs of every dataset (system label), at most ``max_runs`` per dataset:
    those with the most good (pair, coefficient) targets, of
    CANDIDATES_PER_DATASET directories read at random (``workers`` threads)."""
    import concurrent.futures
    from mllf.pretrain.runs import system_label
    by_dataset: Dict[str, List[Path]] = defaultdict(list)
    for d in dirs:
        by_dataset[system_label(d)].append(d)
    rng = random.Random(seed)
    chosen = []
    tasks = []
    for name, ds in sorted(by_dataset.items()):
        if max_runs and len(ds) > CANDIDATES_PER_DATASET:
            ds = rng.sample(ds, CANDIDATES_PER_DATASET)
        tasks += [(d, name, reward_config) for d in ds]
    by_name: Dict[str, List] = defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for (_, name, _), found in zip(tasks, ex.map(_scored_runs, tasks)):
            by_name[name].extend(found)
    for name in sorted(by_dataset):
        scored = sorted(by_name[name], key=lambda t: (-t[0], t[1]))
        keep = [r for _, _, r in (scored[:max_runs] if max_runs else scored)]
        chosen.extend(keep)
        log.debug(f"{name}: {len(keep)} of {len(scored)} runs with good targets")
    log.info(f"  {len(chosen)} runs from {len(by_dataset)} datasets")
    return chosen


def _node_labels(run_dir: Path, n: int) -> List[Dict]:
    """[(site, sub, fragment)] of a run's nodes in embedding order."""
    try:
        gi = json.loads((run_dir / 'graph_info.json').read_text())
    except (OSError, ValueError):
        return [{'site': None, 'sub': None, 'fragment': None}] * n
    nodes = sorted((v for v in (gi.get('sites') or {}).values()
                    if v.get('site') is not None),
                   key=lambda v: (v['site'], v['sub']))
    out = [{'site': v['site'], 'sub': v['sub'],
            'fragment': v.get('master') or Path(v.get('filename') or '').stem or None}
           for v in nodes]
    return out if len(out) == n else [{'site': None, 'sub': None, 'fragment': None}] * n


def collect(model_dir: Path, out_cache: Path, model: Optional[str] = None,
            config_path: Optional[str] = None, device: Optional[str] = None,
            max_runs: Optional[int] = 10, workers: int = 8) -> Dict:
    """Records of every selected run (cached in ``out_cache`` by model file,
    manifest and ``max_runs``): {'runs': [...], 'settings': {...}}.

    Each run record: dataset, system, family, environment, run_dir, nsubs,
    pairs [E], nodes (site, sub, fragment), ligand embeddings [N, 512],
    pred / actual / sd / good [E, 4]."""
    import torch

    from mllf.analysis.common import load_cache, save_cache
    from mllf.pretrain.features import build_run_inputs
    model_path = model_files(model_dir, model)
    key = {'model': str(model_path.resolve()), 'mtime': model_path.stat().st_mtime,
           'max_runs': max_runs}
    cached = load_cache(out_cache)
    if cached.get('key') == key:
        log.info(f"Using cached predictions ({out_cache})")
        return cached
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    unimol, config = model_settings(model_dir, config_path)
    reward_config = config.get('reward') or {}
    log.info(f"Model {model_path}; unimol env_input {unimol.env_input}, site size "
             f"{unimol.site_size_input}, other-site references {unimol.ligand_other_sites}, "
             f"{unimol.structure_selection} structures")
    dirs = manifest_dirs(model_dir)
    log.info(f"Selecting runs from {len(dirs)} pretraining directories "
             f"(up to {max_runs or 'all'} per dataset)")
    runs = select_runs(dirs, max_runs, reward_config, workers=workers)
    work = out_cache.parent / 'inputs'
    work.mkdir(parents=True, exist_ok=True)
    inputs = build_run_inputs(runs, unimol, work)
    policy = load_model(model_path, config, device)
    from mllf.analysis.common import environment_of
    records = []
    for run, graph in zip(runs, inputs.graph_cache):
        if graph is None:
            continue
        emb, ei, targets = graph
        if targets is None:
            continue
        pairs = [tuple(int(v) for v in ei[:, k]) for k in range(ei.size(1))]
        actual = np.asarray(targets, float)
        good = good_mask(run['sim_results'], pairs, reward_config) & (np.abs(actual) > 1e-8)
        pred, sd = _predict(policy, emb.to(device), ei.to(device))
        run_dir = Path(run['run_dir'])
        dataset = run['system_label']
        records.append({
            'dataset': dataset, 'system': system_name(dataset), 'family': family_of(dataset),
            'environment': environment_of(dataset,
                                          (run.get('metadata') or {}).get('solvent_state')),
            'run_dir': str(run_dir), 'nsubs': list(run['metadata'].get('nsubs_per_site') or []),
            'pairs': pairs, 'nodes': _node_labels(run_dir, emb.size(0)),
            'ligand': emb[:, :512].cpu().numpy().astype(np.float32),
            # the node input the model sees: ligand embedding, then the second
            # half's phase flags and site size (515-D)
            'node_input': emb[:, :515].cpu().numpy().astype(np.float32),
            'pred': pred, 'actual': actual, 'sd': sd, 'good': good})
    log.info(f"  predictions for {len(records)} runs")
    data = {'key': key, 'runs': records,
            'settings': {'model': str(model_path), 'env_input': unimol.env_input,
                         'site_size_input': unimol.site_size_input,
                         'ligand_other_sites': unimol.ligand_other_sites,
                         'structure_selection': unimol.structure_selection,
                         'max_runs': max_runs, 'directories': len(dirs)}}
    save_cache(out_cache, data)
    return data


def _predict(policy, emb, ei):
    """(mean [E, 4], posterior predictive sd [E, 4] or NaN) in physical units."""
    import torch
    with torch.no_grad():
        mean, _ = policy._forward_edges(emb, ei)
        mean = mean.cpu().numpy()
        sd = np.full_like(mean, np.nan)
        if getattr(policy, 'use_bayesian_heads', False):
            z = policy.forward_features(emb, ei, None)
            for d in range(mean.shape[1]):
                m, v = policy.edge_mlp.mlps[d].bayesian_head.predict(z[:, d, :])
                s = v.clamp_min(0).sqrt()
                sd[:, d] = ((policy._preact_to_action(m + s, d)
                             - policy._preact_to_action(m - s, d)).abs() / 2).cpu().numpy()
    return mean, sd


# names

_GROUP_SUFFIXES = ('_pair_combos_v2', '_triplet_combos_v2', '_quad_combos_v2', '_pair_combos',
                   '_triplet_combos', '_quad_combos', '_combos')
_ENV_TOKENS = {'vacuum': 'vacuum', 'vac': 'vacuum', 'gas': 'vacuum', 'solvent': 'solvent',
               'solv': 'solvent', 'water': 'solvent', 'protein': 'protein', 'prot': 'protein'}


def system_name(dataset: str) -> str:
    """The system of a dataset: group and combo-batch suffixes dropped and the
    environment word normalized (``luis_bace_solvent_group3`` ->
    ``luis_bace_solvent``, ``14benz_solv_pair_combos`` -> ``14benz_solvent``)."""
    import re
    name = re.sub(r'_group[A-Za-z0-9]*$', '', dataset)
    for suf in _GROUP_SUFFIXES:
        if name.endswith(suf):
            name = name[:-len(suf)]
    parts = name.split('_')
    for i, p in enumerate(parts):
        if p.lower() in _ENV_TOKENS:
            return '_'.join(parts[:i] + [_ENV_TOKENS[p.lower()]])
    return name


def family_of(dataset: str) -> str:
    """The molecule series of a dataset: its system name without the environment."""
    parts = system_name(dataset).split('_')
    if parts[-1] in ('vacuum', 'solvent', 'protein') and len(parts) > 1:
        parts = parts[:-1]
    return '_'.join(parts)
