"""Final bias export: fixed biases for production runs.

Once a system finishes its last curriculum stage, its production biases are
the shared model's deterministic prediction plus each pair's offset posterior
mean (no Thompson or offset sampling), with b fitted per site from every
within-site edge (``write_variables_from_actions`` / ``fit_site_b``, weighted
by each pair's linear-offset variance).

By default the export uses the system's own input directory, i.e. every
substituent at every site (all perturbations in one simulation); pass
``combo_dir`` for a subset (e.g. an ALF-sized group). Cross-site couplings are
not predicted and stay 0.

Writes ``<out_dir>/variables.py`` plus ``export_info.json`` (node -> original
site/sub, how many pairs had offsets, b-fit misfit per site).

Standalone use, from a finished run::

    python -m mllf.cb.export_biases config.yaml --model final_model.pt \
        --bandit-state checkpoints/bandit_state.json --out final_biases

Confidence labels (``label_final_biases``): each exported group directory gets
``CONFIDENCE.txt`` / ``confidence.json`` listing whether the group passed its
group test and, for every pair in the group, whether it is resolved, was run
but not resolved, or was never run (with visits, good runs and the b/c offset
uncertainty). A group that never passed is renamed ``<group>_unconverged``.
The training driver labels each system when it finishes; for a finished run::

    python -m mllf.cb.export_biases config.yaml --label checkpoints_online_A
"""
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from mllf.cb import system_stages as ss
from mllf.cb.bandit_config import BanditConfig
from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.group_correction import apply_group_values, correction_config
from mllf.cb.hindsight import (
    BIAS_DIMS,
    OFFSET_KEY,
    OFFSET_PRIOR_SD,
    apply_pair_offsets,
    cap_quadratic,
    combo_node_labels,
    linear_edge_variance,
    pair_offset_posterior,
)
from mllf.file_handling.combo_sets import build_combo_sets
from mllf.file_handling.write_bias_coeff import write_variables_from_actions


log = logging.getLogger(__name__)


def input_dir_labels(combo_dir) -> List[Tuple[int, int]]:
    """(site, sub) per node for a directory whose substituent files carry their
    original numbers (a system's input dir): sorted by site, then sub, the
    node order ``build_graph_and_data`` uses."""
    d = Path(combo_dir)
    pdb_dir = d / 'prep' if (d / 'prep').exists() else d
    found = set()
    for p in pdb_dir.glob('site*_sub*.pdb'):
        m = re.search(r'site(\d+)_sub(\d+)', p.name)
        if m:
            found.add((int(m.group(1)), int(m.group(2))))
    return sorted(found)


def export_final_biases(policy, system_cfg: Dict, config: Dict, bandit_state: Optional[Dict],
                        out_dir, device: str = 'cpu', combo_dir=None,
                        dims=BIAS_DIMS) -> Path:
    """Write production biases for one system (see module docstring).

    Args:
        policy: the trained UnimolPolicy (Bayesian heads).
        system_cfg: the system's ``systems:`` entry (name, input_dir, solvent_state).
        config: the workflow config (unimol, bandit, training.bias_clip).
        bandit_state: per-pair state with offsets (``bandit_state.json``), or None
            for the shared model alone.
        out_dir: directory to write variables.py and export_info.json into.
        combo_dir: substituent set to export; default the system's input_dir.
    Returns the path of the written variables.py.
    """
    combo_dir = Path(combo_dir or system_cfg['input_dir'])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    system = system_cfg.get('name', 'unknown')
    bandit = BanditConfig.from_config(config)

    data, extras = build_graph_and_data(str(combo_dir), system_cfg, config, device)
    labels = combo_node_labels(combo_dir) or input_dir_labels(combo_dir)
    policy.eval()
    with torch.no_grad():
        mean, _, _ = policy.get_actions_thompson(
            data.x, data.edge_index, data.edge_type, deterministic=True)
    mean = mean.detach()
    actions, n_shifted = apply_pair_offsets(
        mean, mean, bandit_state, system, labels, data.edge_index,
        prior_sd=bandit.offset_prior_sd, sample=False, dims=dims)
    actions, _ = cap_quadratic(actions, mean, bandit.c_cap)
    if correction_config(bandit.raw) is not None:
        # the group's corrected x / s (bandit.group_correction), if any
        actions, _ = apply_group_values(actions, bandit_state, system, combo_dir.name,
                                        labels, data.edge_index)
    lin_var = linear_edge_variance(bandit_state, system, labels, data.edge_index,
                                   prior_sd=bandit.offset_prior_sd)
    write_variables_from_actions(
        str(out_dir), data, extras, actions.cpu(),
        bias_clip=config.get('training', {}).get('bias_clip', 1000.0),
        linear_var=lin_var)

    b_fit_path = out_dir / 'b_fit.json'
    info = {
        'system': system,
        'combo_dir': str(combo_dir),
        'nodes': [{'node': n, 'site': s, 'sub': u} for n, (s, u) in enumerate(labels)],
        'nsubs_per_site': extras.get('nsubs_per_site'),
        'edges_with_offsets': int(n_shifted),
        'n_edges': int(data.edge_index.size(1)),
        'b_fit': json.loads(b_fit_path.read_text()) if b_fit_path.exists() else {},
        'cross_site_couplings': 'not predicted (0)',
    }
    (out_dir / 'export_info.json').write_text(json.dumps(info, indent=1))
    return out_dir / 'variables.py'


UNCONVERGED_SUFFIX = '_unconverged'


def group_confidence(system: str, group_combo, ref_keys: set, bandit_state: Optional[Dict],
                     stage: str, adv: Dict,
                     prior_sd: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Per within-site pair of a group: reference or not, status (resolved /
    run, not resolved / never run), visits and good runs in ``stage``, and the
    b / c offset posterior sd (the prior sd when never measured)."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    psd = {**OFFSET_PRIOR_SD, **(prior_sd or {})}
    rows = []
    for k in ss.combo_pair_keys(system, group_combo):
        e = store.get(k) or {}
        st = (e.get('stages') or {}).get(stage) or {}
        resolved = ss.pair_resolved(e, stage, int(adv.get('successes', 2)),
                                    adv.get('max_offset_sd'), prior_sd)
        _, site, lo, hi = k.split('|')
        rows.append({
            'pair': f"{site} {lo}-{hi}", 'reference': k in ref_keys,
            'status': 'resolved' if resolved else ('run, not resolved' if st.get('visits')
                                                   else 'never run'),
            'visits': int(st.get('visits', 0)), 'good_runs': int(st.get('successes', 0)),
            'b_offset_sd': round(pair_offset_posterior(0.0, e.get('linear') or [],
                                                       psd['linear'])[1] ** 0.5, 2),
            'c_offset_sd': round(pair_offset_posterior(0.0, e.get('quadratic') or [],
                                                       psd['quadratic'])[1] ** 0.5, 2),
        })
    return rows


def label_group_export(group_dir, system: str, group_combo, passed_run: Optional[str],
                       rows: List[Dict]) -> Path:
    """Write CONFIDENCE.txt / confidence.json into an exported group directory
    and rename it ``<group>_unconverged`` if the group never passed. Returns
    the (possibly renamed) directory. Idempotent."""
    group_dir = Path(group_dir)
    name = Path(group_combo).name
    base = group_dir.parent / name
    target = base if passed_run else group_dir.parent / f"{name}{UNCONVERGED_SUFFIX}"
    for d in (base, group_dir.parent / f"{name}{UNCONVERGED_SUFFIX}"):
        if d.exists() and d != target:
            if target.exists():
                shutil.rmtree(target)
            d.rename(target)
    target.mkdir(parents=True, exist_ok=True)
    n_res = sum(r['status'] == 'resolved' for r in rows)
    info = {'system': system, 'group': name, 'group_passed': bool(passed_run),
            'passing_run': passed_run, 'pairs_resolved': n_res, 'pairs': len(rows),
            'pair_confidence': rows}
    (target / 'confidence.json').write_text(json.dumps(info, indent=1))
    lines = [f"{system} / {name}",
             (f"Group PASSED (every substituent sampled) in {passed_run}" if passed_run else
              "Group did NOT pass: no run sampled every substituent; biases are the model + "
              "pair-offset estimate"),
             f"Pairs resolved: {n_res}/{len(rows)}", "",
             f"{'pair':16s} {'ref':4s} {'status':20s} {'visits':>6s} {'good':>5s} {'b_sd':>6s} "
             f"{'c_sd':>6s}"]
    for r in sorted(rows, key=lambda r: (
            r['status'] != 'resolved', not r['reference'], r['pair'])):
        lines.append(f"{r['pair']:16s} {'yes' if r['reference'] else '':4s} {r['status']:20s} "
                     f"{r['visits']:6d} {r['good_runs']:5d} {r['b_offset_sd']:6.2f} "
                     f"{r['c_offset_sd']:6.2f}")
    (target / 'CONFIDENCE.txt').write_text('\n'.join(lines) + '\n')
    return target


def label_final_biases(checkpoint_dir, config: Dict, systems: Optional[List[str]] = None) -> int:
    """Label every exported group of a run (``final_biases/<system>/<group>``)
    from its system_stages.json and bandit_state.json (see
    ``label_group_export``). Returns the number of groups labelled."""
    ck = Path(checkpoint_dir)
    stages_state = json.loads((ck / 'system_stages.json').read_text())
    bandit = (json.loads((ck / 'bandit_state.json').read_text())
              if (ck / 'bandit_state.json').exists() else {})
    stage_cfg = next((st for st in (config.get('curriculum') or {}).get('stages', [])
                      if st.get('group_set')), None)
    if stage_cfg is None:
        return 0
    n = 0
    for sys_cfg in config.get('systems', []):
        name = sys_cfg.get('name')
        if (systems and name not in systems) or name not in stages_state:
            continue
        sets, _ = build_combo_sets(sys_cfg, config.get('alf_root'))
        rset = stage_cfg.get('reference_set')
        ref_keys = {k for c in sets.get(rset, sets.get(stage_cfg.get('combo_set'), []))
                    for k in ss.combo_pair_keys(name, c)}
        passed = stages_state[name].get('groups_done') or {}
        for g in sets.get(stage_cfg['group_set'], []):
            gname = Path(g).name
            exported = ck / 'final_biases' / name
            if not ((exported / gname).exists()
                    or (exported / f"{gname}{UNCONVERGED_SUFFIX}").exists()):
                continue
            rows = group_confidence(name, g, ref_keys, bandit, stage_cfg['name'], stage_cfg.get(
                'advance') or {}, BanditConfig.from_config(config).offset_prior_sd)
            label_group_export(exported / gname, name, g, passed.get(gname), rows)
            n += 1
    return n


def main(argv=None):
    """Same as ``mllf export``."""
    from mllf.cli.export import main as cli_main
    return cli_main(argv)


if __name__ == '__main__':
    main()
