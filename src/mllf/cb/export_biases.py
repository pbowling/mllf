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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from mllf.cb.graph_build import build_graph_and_data
from mllf.cb.hindsight import (
    BIAS_DIMS,
    OFFSET_KEY,
    OFFSET_PRIOR_SD,
    apply_pair_offsets,
    combo_node_labels,
    linear_edge_variance,
    pair_offset_posterior,
)
from mllf.cli.workflow import write_variables_from_actions


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
    bandit_cfg = config.get('bandit', {})

    data, extras = build_graph_and_data(str(combo_dir), system_cfg, config, device)
    labels = combo_node_labels(combo_dir) or input_dir_labels(combo_dir)
    policy.eval()
    with torch.no_grad():
        mean, _, _ = policy.get_actions_thompson(
            data.x, data.edge_index, data.edge_type, deterministic=True)
    mean = mean.detach()
    actions, n_shifted = apply_pair_offsets(
        mean, mean, bandit_state, system, labels, data.edge_index,
        prior_sd=bandit_cfg.get('offset_prior_sd'), sample=False, dims=dims)
    lin_var = linear_edge_variance(bandit_state, system, labels, data.edge_index,
                                   prior_sd=bandit_cfg.get('offset_prior_sd'))
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
                     stage: str, adv: Dict, prior_sd: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Per within-site pair of a group: reference or not, status (resolved /
    run, not resolved / never run), visits and good runs in ``stage``, and the
    b / c offset posterior sd (the prior sd when never measured)."""
    from mllf.cb import system_stages as ss
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    psd = {**OFFSET_PRIOR_SD, **(prior_sd or {})}
    rows = []
    for k in ss.combo_pair_keys(system, group_combo):
        e = store.get(k) or {}
        st = (e.get('stages') or {}).get(stage) or {}
        resolved = ss.pair_resolved(e, stage, int(adv.get('successes', 2)), adv.get('max_offset_sd'), prior_sd)
        _, site, lo, hi = k.split('|')
        rows.append({
            'pair': f"{site} {lo}-{hi}", 'reference': k in ref_keys,
            'status': 'resolved' if resolved else ('run, not resolved' if st.get('visits') else 'never run'),
            'visits': int(st.get('visits', 0)), 'good_runs': int(st.get('successes', 0)),
            'b_offset_sd': round(pair_offset_posterior(0.0, e.get('linear') or [], psd['linear'])[1] ** 0.5, 2),
            'c_offset_sd': round(pair_offset_posterior(0.0, e.get('quadratic') or [], psd['quadratic'])[1] ** 0.5, 2),
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
                import shutil
                shutil.rmtree(target)
            d.rename(target)
    target.mkdir(parents=True, exist_ok=True)
    n_res = sum(r['status'] == 'resolved' for r in rows)
    info = {'system': system, 'group': name, 'group_passed': bool(passed_run),
            'passing_run': passed_run, 'pairs_resolved': n_res, 'pairs': len(rows), 'pair_confidence': rows}
    (target / 'confidence.json').write_text(json.dumps(info, indent=1))
    lines = [f"{system} / {name}",
             (f"Group PASSED (every substituent sampled) in {passed_run}" if passed_run else
              "Group did NOT pass: no run sampled every substituent; biases are the model + pair-offset estimate"),
             f"Pairs resolved: {n_res}/{len(rows)}", "",
             f"{'pair':16s} {'ref':4s} {'status':20s} {'visits':>6s} {'good':>5s} {'b_sd':>6s} {'c_sd':>6s}"]
    for r in sorted(rows, key=lambda r: (r['status'] != 'resolved', not r['reference'], r['pair'])):
        lines.append(f"{r['pair']:16s} {'yes' if r['reference'] else '':4s} {r['status']:20s} "
                     f"{r['visits']:6d} {r['good_runs']:5d} {r['b_offset_sd']:6.2f} {r['c_offset_sd']:6.2f}")
    (target / 'CONFIDENCE.txt').write_text('\n'.join(lines) + '\n')
    return target


def label_final_biases(checkpoint_dir, config: Dict, systems: Optional[List[str]] = None) -> int:
    """Label every exported group of a run (``final_biases/<system>/<group>``)
    from its system_stages.json and bandit_state.json (see
    ``label_group_export``). Returns the number of groups labelled."""
    from mllf.cb import system_stages as ss
    from mllf.file_handling.combo_sets import build_combo_sets
    ck = Path(checkpoint_dir)
    stages_state = json.loads((ck / 'system_stages.json').read_text())
    bandit = json.loads((ck / 'bandit_state.json').read_text()) if (ck / 'bandit_state.json').exists() else {}
    stage_cfg = next((st for st in (config.get('curriculum') or {}).get('stages', []) if st.get('group_set')),
                     None)
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
            if not ((exported / gname).exists() or (exported / f"{gname}{UNCONVERGED_SUFFIX}").exists()):
                continue
            rows = group_confidence(name, g, ref_keys, bandit, stage_cfg['name'], stage_cfg.get('advance') or {},
                                    (config.get('bandit') or {}).get('offset_prior_sd'))
            label_group_export(exported / gname, name, g, passed.get(gname), rows)
            n += 1
    return n


def main() -> None:
    import argparse
    import yaml
    from mllf.cb.policy import UnimolPolicy
    from mllf.cb.workflow_utils import load_bandit_state

    p = argparse.ArgumentParser(description='Export production biases from a trained policy')
    p.add_argument('config', help='workflow YAML (systems, unimol, bandit, training)')
    p.add_argument('--model', default=None, help='checkpoint with policy_state_dict or policy_state')
    p.add_argument('--bandit-state', default=None, help='bandit_state.json with per-pair offsets')
    p.add_argument('--systems', nargs='*', default=None, help='system names (default: all in config)')
    p.add_argument('--out', default='final_biases', help='output root (one subdirectory per system)')
    p.add_argument('--label', default=None, metavar='CHECKPOINT_DIR',
                   help='only label an existing run\'s exported groups (confidence files, '
                        '_unconverged names) and exit')
    args = p.parse_args()

    config = yaml.safe_load(open(args.config))
    if args.label:
        n = label_final_biases(args.label, config, args.systems)
        print(f"labelled {n} exported group(s) under {Path(args.label) / 'final_biases'}")
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pol_cfg = config.get('training', {}).get('policy', {})
    policy = UnimolPolicy(
        unimol_dim=pol_cfg.get('unimol_dim', 512), mlp_hidden=pol_cfg.get('mlp_hidden', 64),
        mlp_out_dim=4, use_dual_embeddings=True,
        use_bayesian_heads=pol_cfg.get('use_bayesian_heads', True),
        bayesian_prior_precision=config.get('bandit', {}).get('prior_precision', 1.0),
    ).to(device)
    if not args.model:
        p.error('--model is required unless --label is given')
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt.get('policy_state_dict') or ckpt['policy_state'])
    state = load_bandit_state(Path(args.bandit_state)) if args.bandit_state else None
    for sys_cfg in config.get('systems', []):
        if args.systems and sys_cfg.get('name') not in args.systems:
            continue
        path = export_final_biases(policy, sys_cfg, config, state,
                                   Path(args.out) / sys_cfg['name'], device=device)
        print(f"{sys_cfg['name']}: {path}")


if __name__ == '__main__':
    main()
