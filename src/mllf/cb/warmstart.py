"""Warm-start bias coefficient initialization from pretraining data.

Builds a mapping of (original_site, original_sub) → bias coefficients from
the best-performing run(s) in a pretraining directory.  At a chosen epoch
(default 0) of RL training these biases are applied directly as the actions
submitted to the simulator, bypassing the stochastic policy sample.

Why this helps
--------------
Epoch 0 typically has ~100 % of combos below the floor-mask threshold
(Policy Loss 0.0000) because the random policy produces ≤1 transition per
site and the reward is pure noise.  The warm-start breaks this cold-start
loop by using pretraining biases that are already calibrated for the system,
guaranteeing at least *some* transitions and real REINFORCE signal from
the very first epoch.

The REINFORCE gradient is still valid because ``evaluate_logp`` is always
called at epoch-end under the *current* policy parameters — the warmstart
actions are treated as off-policy samples, which is the standard
pre-collected data pattern already used for cached (already-run) combos.

Building the map (run once before training):
--------------------------------------------
    from mllf.cb.warmstart import build_warmstart_map, save_warmstart_map
    m = build_warmstart_map('/path/to/pretraining/14benz_solv',
                             nsubs_per_site=[6, 5])
    save_warmstart_map(m, '/path/to/warmstart_14benz_solv.json')

    # or use the CLI script:
    python examples/build_warmstart_map.py \\
        --pretrain-dir /path/to/pretraining/14benz_solv \\
        --nsubs-per-site 6 5 \\
        --output /path/to/warmstart_14benz_solv.json

Using in training:
------------------
    from mllf.cb.warmstart import WarmStartMapper
    mapper = WarmStartMapper('/path/to/warmstart_14benz_solv.json')
    # Pass to train_epoch(..., warmstart_mapper=mapper, warmstart_epoch=0)
"""
from __future__ import annotations

import json
import re
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_bias_from_variables_py(py_path: Path) -> dict:
    """Extract the YAML bias dict from a ``variables.py`` file."""
    text = py_path.read_text(encoding='utf-8')
    m = re.search(r'bias_string\s*=\s*(?:"""|\'\'\')([\s\S]*?)(?:"""|\'\'\')' , text)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


def _node_to_site_sub(node_idx: int, nsubs_per_site: List[int]) -> Tuple[int, int]:
    """Convert 0-based node index → 1-based (site, sub) tuple."""
    offset = 0
    for site_idx, nsubs in enumerate(nsubs_per_site):
        if node_idx < offset + nsubs:
            return (site_idx + 1, node_idx - offset + 1)
        offset += nsubs
    raise IndexError(
        f"node_idx {node_idx} out of range for nsubs_per_site={nsubs_per_site}"
    )


def _site_sub_to_node(site: int, sub: int, nsubs_per_site: List[int]) -> int:
    """Convert 1-based (site, sub) → 0-based node index."""
    return sum(nsubs_per_site[: site - 1]) + (sub - 1)


def _score_run(sim_results: dict, scoring: str = 'total_transitions') -> float:
    """Return a scalar score for one pretraining run."""
    if scoring == 'total_transitions':
        trans = sim_results.get('transitions', {})
        total = 0.0
        for v in trans.values():
            total += sum(v.values()) if isinstance(v, dict) else float(v)
        return total
    if scoring == 'num_populated_blocks':
        return float(sim_results.get('num_populated_blocks', 0))
    raise ValueError(f"Unknown scoring method: {scoring!r}")


# ---------------------------------------------------------------------------
# Public API: build the map
# ---------------------------------------------------------------------------

def build_warmstart_map(
    pretrain_dir: str,
    nsubs_per_site: List[int],
    scoring: str = 'total_transitions',
    top_k: int = 1,
) -> dict:
    """Scan a pretraining directory and build a warm-start bias mapping.

    Finds the best-performing run(s) by ``scoring`` metric, reads the bias
    matrices from ``variables.py``, and stores them indexed by original
    (site, sub) identifiers.

    Args:
        pretrain_dir: Directory containing ``run{N}/`` subdirectories, each
            with ``variables.py`` and ``simulation_results.json``.
        nsubs_per_site: Substituent counts per site in the pretraining system,
            e.g. ``[6, 5]`` for 14benz.
        scoring: Metric used to rank runs.  Options:
            ``'total_transitions'`` (default) or ``'num_populated_blocks'``.
        top_k: How many top runs to average (default 1 = best run only).

    Returns:
        Dict with keys:
            ``'system'``, ``'source_runs'``, ``'score'``, ``'score_type'``,
            ``'nsubs_per_site'``, ``'b'``, ``'c'``, ``'x'``, ``'s'``.
        ``b`` is keyed by ``"{site},{sub}"``; ``c/x/s`` by
        ``"{site1},{sub1},{site2},{sub2}"``.

    Raises:
        FileNotFoundError: If no valid run directories are found.
        ValueError: If the bias data from the top runs is unusable.
    """
    base = Path(pretrain_dir)
    run_scores: List[Tuple[float, Path]] = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        sr_path = run_dir / 'simulation_results.json'
        vp_path = run_dir / 'variables.py'
        if not sr_path.exists() or not vp_path.exists():
            continue
        try:
            sim = json.loads(sr_path.read_text())
            score = _score_run(sim, scoring)
            run_scores.append((score, run_dir))
        except Exception:
            continue

    if not run_scores:
        raise FileNotFoundError(f"No valid run directories found in {pretrain_dir}")

    run_scores.sort(key=lambda t: t[0], reverse=True)
    selected = run_scores[:top_k]

    n_nodes = sum(nsubs_per_site)
    b_acc = [0.0] * n_nodes
    c_acc = [[0.0] * n_nodes for _ in range(n_nodes)]
    x_acc = [[0.0] * n_nodes for _ in range(n_nodes)]
    s_acc = [[0.0] * n_nodes for _ in range(n_nodes)]
    n_valid = 0

    for _, run_dir in selected:
        bias = _load_bias_from_variables_py(run_dir / 'variables.py')
        b_raw = bias.get('b', [])
        c_raw = bias.get('c', [])
        x_raw = bias.get('x', [])
        s_raw = bias.get('s', [])

        # Flatten b (pretraining format may be [[val1, val2, ...]])
        if b_raw and isinstance(b_raw[0], list):
            b_flat = [v for row in b_raw for v in row]
        else:
            b_flat = list(b_raw)

        if len(b_flat) != n_nodes:
            continue
        n_valid += 1

        for i in range(n_nodes):
            b_acc[i] += float(b_flat[i])
        for i in range(n_nodes):
            for j in range(n_nodes):
                try:
                    c_acc[i][j] += float(c_raw[i][j]) if c_raw else 0.0
                    x_acc[i][j] += float(x_raw[i][j]) if x_raw else 0.0
                    s_acc[i][j] += float(s_raw[i][j]) if s_raw else 0.0
                except (IndexError, TypeError):
                    pass

    if n_valid == 0:
        raise ValueError(
            f"No usable bias data found in top-{top_k} runs of {pretrain_dir}"
        )

    b_avg = [v / n_valid for v in b_acc]
    c_avg = [[v / n_valid for v in row] for row in c_acc]
    x_avg = [[v / n_valid for v in row] for row in x_acc]
    s_avg = [[v / n_valid for v in row] for row in s_acc]

    # Key format: "{site},{sub}" for b; "{s1},{b1},{s2},{b2}" for c/x/s
    b_map: Dict[str, float] = {}
    c_map: Dict[str, float] = {}
    x_map: Dict[str, float] = {}
    s_map: Dict[str, float] = {}

    for ni in range(n_nodes):
        si, subi = _node_to_site_sub(ni, nsubs_per_site)
        b_map[f"{si},{subi}"] = b_avg[ni]
        for nj in range(n_nodes):
            if ni == nj:
                continue
            sj, subj = _node_to_site_sub(nj, nsubs_per_site)
            key = f"{si},{subi},{sj},{subj}"
            c_map[key] = c_avg[ni][nj]
            x_map[key] = x_avg[ni][nj]
            s_map[key] = s_avg[ni][nj]

    return {
        'system': base.name,
        'source_runs': [d.name for _, d in selected[:n_valid]],
        'score': float(selected[0][0]),
        'score_type': scoring,
        'nsubs_per_site': list(nsubs_per_site),
        'b': b_map,
        'c': c_map,
        'x': x_map,
        's': s_map,
    }


def save_warmstart_map(map_dict: dict, out_path: str) -> None:
    """Write a warmstart map dict to a JSON file."""
    out = Path(out_path)
    out.write_text(json.dumps(map_dict, indent=2), encoding='utf-8')
    n_nodes = len(map_dict['b'])
    n_pairs = len(map_dict['c'])
    print(
        f"Saved warmstart map → {out}\n"
        f"  system: {map_dict['system']}, runs: {map_dict['source_runs']}, "
        f"score({map_dict['score_type']}): {map_dict['score']:.1f}\n"
        f"  {n_nodes} node biases, {n_pairs} directed-pair biases"
    )


# ---------------------------------------------------------------------------
# Public API: apply the map during training
# ---------------------------------------------------------------------------

class WarmStartMapper:
    """Provides warm-start bias coefficient actions from a pretraining map.

    Load once in ``main()`` and pass to ``train_epoch()`` as
    ``warmstart_mapper``.  ``get_actions_for_combo`` returns a
    ``[E, 4]`` actions tensor that can be submitted directly to the
    simulator instead of sampling from the policy.

    Falls back silently (returns ``None``) if a combo cannot be mapped, so
    the policy is used as a fallback without any code changes in the
    training loop.

    Args:
        map_path: Path to the JSON file produced by :func:`build_warmstart_map`.
    """

    def __init__(self, map_path: str) -> None:
        data = json.loads(Path(map_path).read_text(encoding='utf-8'))
        self._b: Dict[str, float] = data['b']
        self._c: Dict[str, float] = data['c']
        self._x: Dict[str, float] = data['x']
        self._s: Dict[str, float] = data['s']
        self.nsubs_per_site: List[int] = data['nsubs_per_site']
        self.source_runs: List[str] = data.get('source_runs', [])
        self.system: str = data.get('system', '')
        self.score: float = data.get('score', 0.0)
        self.score_type: str = data.get('score_type', 'total_transitions')

    # ------------------------------------------------------------------
    # Private lookup helpers
    # ------------------------------------------------------------------

    def _b_val(self, orig_site: int, orig_sub: int) -> float:
        return self._b.get(f"{orig_site},{orig_sub}", 0.0)

    def _pair_val(self, table: dict,
                  s1: int, b1: int, s2: int, b2: int) -> float:
        return table.get(f"{s1},{b1},{s2},{b2}", 0.0)

    def _c_canonical(self, os1: int, ob1: int, os2: int, ob2: int) -> float:
        """Return the canonical c value, always from the upper-triangle entry.

        In variables.py c is stored upper-triangular (i<j only).  The mapper
        mirrors this: ``c_map["{s_lo},{b_lo},{s_hi},{b_hi}"]`` holds the
        non-zero value and the reversed key holds 0.  We look up whichever
        ordering has the smaller original node index as the first key.
        """
        n1 = _site_sub_to_node(os1, ob1, self.nsubs_per_site)
        n2 = _site_sub_to_node(os2, ob2, self.nsubs_per_site)
        if n1 <= n2:
            return self._pair_val(self._c, os1, ob1, os2, ob2)
        else:
            return self._pair_val(self._c, os2, ob2, os1, ob1)

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def get_actions_for_combo(
        self,
        combo_path: Path,
        g,
        data,
        extras: dict,
        device: str = 'cpu',
    ) -> Optional[torch.Tensor]:
        """Build a warm-start ``[E, 4]`` actions tensor for one combo.

        For each directed edge in ``data.edge_index`` the method:

        1. Reads ``combo_path/mapping.json`` to obtain (new_site, new_sub) →
           (orig_site, orig_sub) for every node in the combo.
        2. Looks up the pretraining bias values indexed by original (site, sub).
        3. Assembles a ``[E, 4]`` tensor with columns
           ``[linear, quadratic, skew, end]`` matching the ``bias_type_index``
           in ``write_variables_from_actions``.

        Sign convention for quadratic
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``write_variables_from_actions`` stores ``-val`` when the edge has a
        backward relation tag.  To recover the canonical pretraining c value
        the method negates for backward edges:

        * forward edge → ``actions[k, 1] = c_canonical``
        * backward edge → ``actions[k, 1] = -c_canonical``

        Linear
        ~~~~~~
        ``write_variables_from_actions`` forces the first sub of each site to
        b=0 unconditionally.  For combos where the combo's reference sub
        (new_sub==1) maps to a **different** original sub from the pretraining
        reference (i.e. one with a non-zero pretraining b), all b values at
        that site must be expressed relative to the new reference.  The mapper
        computes ``b_ref`` per site (the pretraining b of whatever original sub
        maps to combo_sub1) and assigns ``actions[k, 0] = b_orig[dst] - b_ref``.
        When the reference is unchanged (b_ref==0), this is a no-op.

        Returns:
            ``torch.Tensor`` of shape ``[E, 4]`` on ``device``, or ``None``
            if the mapping is incomplete or ``mapping.json`` is missing.
        """
        mapping_path = combo_path / 'mapping.json'
        if not mapping_path.exists():
            return None

        try:
            mapping_entries = json.loads(mapping_path.read_text())
        except Exception:
            return None

        # (new_site, new_sub) → (orig_site, orig_sub) — deduplicate by new key
        new_to_orig: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for entry in mapping_entries:
            ns = entry.get('new_site')
            nb = entry.get('new_sub')
            os = entry.get('original_site')
            ob = entry.get('original_sub')
            if None in (ns, nb, os, ob):
                continue
            new_to_orig[(int(ns), int(nb))] = (int(os), int(ob))

        if not new_to_orig:
            return None

        # node_idx → (orig_site, orig_sub);  also track new_site and
        # per-(new)site reference b so linear actions can be expressed
        # relative to whichever original sub maps to combo sub 1.
        n_nodes = data.x.size(0)
        node_to_orig: Dict[int, Tuple[int, int]] = {}
        node_to_new_site: Dict[int, int] = {}
        site_b_ref: Dict[int, float] = {}  # new_site → pretraining b of combo's ref sub
        for i in range(n_nodes):
            try:
                meta = g.get_node_info(i) if hasattr(g, 'get_node_info') else {}
                new_site = int(meta.get('site', -1))
                new_sub  = int(meta.get('sub',  -1))
                orig = new_to_orig.get((new_site, new_sub))
                if orig is None:
                    return None  # at least one node has no mapping
                node_to_orig[i] = orig
                node_to_new_site[i] = new_site
                if new_sub == 1:  # reference sub for this site
                    os_ref, ob_ref = orig
                    site_b_ref[new_site] = self._b_val(os_ref, ob_ref)
            except Exception:
                return None

        # relation metadata
        rel_names = extras.get('relation_names', [])
        base_relation_map = extras.get('base_relation_map', {})
        rel_to_base: Dict[str, str] = {}
        for base, (fwd, bwd) in base_relation_map.items():
            rel_to_base[fwd] = base
            rel_to_base[bwd] = base

        E = data.edge_index.size(1)
        actions = torch.zeros(E, 4, dtype=torch.float32)

        # action dim → bias_type_index used by write_variables_from_actions
        # [0]=linear, [1]=quadratic, [2]=skew, [3]=end
        try:
            for k in range(E):
                src_new = int(data.edge_index[0, k].item())
                dst_new = int(data.edge_index[1, k].item())
                os_src, ob_src = node_to_orig[src_new]
                os_dst, ob_dst = node_to_orig[dst_new]

                rel_idx  = int(data.edge_type[k].item())
                rel_name = rel_names[rel_idx] if rel_idx < len(rel_names) else ''
                is_bwd   = rel_name.endswith('_bwd')

                # [0] linear: dst's b relative to this combo's reference sub.
                # When combo_sub1 maps to an original sub with non-zero
                # pretraining b (b_ref), all other subs at that site must be
                # shifted by -b_ref so values are relative to the new reference.
                # When the reference is unchanged (b_ref==0), this is a no-op.
                new_site_dst = node_to_new_site.get(dst_new, -1)
                b_ref = site_b_ref.get(new_site_dst, 0.0)
                actions[k, 0] = self._b_val(os_dst, ob_dst) - b_ref

                # [1] quadratic: always use canonical (upper-triangle) c value;
                # negate for backward edges so write_variables stores the right sign.
                c_val = self._c_canonical(os_src, ob_src, os_dst, ob_dst)
                actions[k, 1] = -c_val if is_bwd else c_val

                # [2] skew: directional — src→dst value directly
                actions[k, 2] = self._pair_val(
                    self._x, os_src, ob_src, os_dst, ob_dst
                )

                # [3] end: directional — src→dst value directly
                actions[k, 3] = self._pair_val(
                    self._s, os_src, ob_src, os_dst, ob_dst
                )

        except (KeyError, IndexError):
            return None

        return actions.to(device)
