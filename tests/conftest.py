"""Shared fixtures.

``fake_cluster`` runs the online-training code (``train_epoch``,
``run_training``) without SLURM, CHARMM or Uni-Mol: graph/embedding
building, combo-directory creation, job submission, ``squeue``/``sbatch``,
output parsing and the reward are replaced by small deterministic fakes;
everything else (proposals, per-pair state, updates, stages, logging,
checkpoints) is the real code.
"""
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

MSLD_TEMPLATE = "timestep = 0.002\nesteps = 1000\nnsteps = 4000\n"


def _seed(*parts) -> int:
    return int(hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _labels(d: Path):
    entries = json.loads((Path(d) / 'mapping.json').read_text())
    nodes = {(e['new_site'], e['new_sub']): (e['original_site'], e['original_sub'])
             for e in entries}
    return [nodes[k] for k in sorted(nodes)]


def write_mapping(d: Path, sites) -> None:
    entries = [{'new_site': ns, 'new_sub': nu, 'original_site': site, 'original_sub': sub}
               for ns, (site, subs) in enumerate(sites, start=1)
               for nu, sub in enumerate(subs, start=1)]
    (Path(d) / 'mapping.json').write_text(json.dumps(entries))


def sites_from_name(name: str):
    out = []
    for part in name.split('__'):
        m = re.search(r'site(\d+)_subs_([\d_]+)', part)
        if m:
            out.append((int(m.group(1)),
                        [int(x) for x in m.group(2).strip('_').split('_') if x]))
    return out


class FakeCluster:
    """The fakes, and what they saw."""

    def __init__(self):
        self.fail_first = ()     # combo-name suffixes whose first attempt fails
        self.fail_always = ()    # combo-name suffixes that always fail
        self.attempts = {}
        self.submitted = []
        self.job = 1000

    # graph / combo dirs -------------------------------------------------
    def build_graph_and_data(self, combo_dir, system_config, global_config, device='cpu'):
        from mllf.cb.pairs import build_directed_pairs
        labels = _labels(Path(combo_dir))
        by_site = {}
        for s, u in labels:
            by_site.setdefault(s, []).append(u)
        nsubs = [len(by_site[s]) for s in sorted(by_site)]
        x = torch.stack([torch.randn(1024, generator=torch.Generator().manual_seed(
            _seed(system_config.get('name'), s, u))) for s, u in labels])
        pairs = build_directed_pairs(nsubs)
        data = type('Data', (), {})()
        data.x = x.to(device)
        data.edge_index = torch.LongTensor([[p[0] for p in pairs], [p[1] for p in pairs]])
        data.edge_type = None
        data.site_index = torch.LongTensor([i for i, n in enumerate(nsubs) for _ in range(n)])
        extras = {'relation_names': ['linear', 'quadratic', 'skew', 'end'],
                  'base_relation_map': {k: k for k in ('linear', 'quadratic', 'skew', 'end')},
                  'nsubs_per_site': nsubs}
        return data, extras

    def ensure_combo_dir_exists(self, combo_path, system_config, global_config, verbose=False,
                                _lookup=None):
        target = Path(system_config['out_dir']) / Path(combo_path).name
        if not target.exists():
            target.mkdir(parents=True)
            write_mapping(target, sites_from_name(target.name))
            shutil.copy(Path(system_config['input_dir']) / 'msld_flat.py', target / 'msld_flat.py')
        return target

    # simulation ---------------------------------------------------------
    def _write_output(self, epoch_dir: Path) -> None:
        combo = epoch_dir.parent
        text = (combo / 'msld_flat.py').read_text()
        e = int(re.search(r"esteps\s*=\s*(\d+)", text).group(1))
        n = int(re.search(r"nsteps\s*=\s*(\d+)", text).group(1))
        k = str(epoch_dir)
        self.attempts[k] = self.attempts.get(k, 0) + 1
        fail = (bool(self.fail_always) and combo.name.endswith(self.fail_always)
                or bool(self.fail_first) and combo.name.endswith(self.fail_first)
                and self.attempts[k] == 1)
        body = ''.join(f"   NSTEP  = {s:8d} NSAVC  = 10\n  TIME STEP = 4.09E-02 AKMA 2.0E-03 PS\n"
                       for s in (e, n))
        (epoch_dir / 'output.out').write_text(('FAIL\n' if fail else 'OK\n') + body)

    def submit_epoch_simulation(self, combo_path, epoch_dir, config=None, solvent_state='solv'):
        self._write_output(Path(epoch_dir))
        (Path(epoch_dir) / 'run_epoch.sh').write_text('#!/bin/bash\n')
        self.submitted.append(str(epoch_dir))
        self.job += 1
        return str(self.job)

    def run(self, cmd, *args, **kwargs):
        if cmd[:1] == ['squeue']:
            return subprocess.CompletedProcess(cmd, 0, stdout='', stderr='')
        if cmd[:1] == ['sbatch']:
            self._write_output(Path(kwargs.get('cwd') or Path(cmd[1]).parent))
            self.job += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=f'Submitted batch job {self.job}',
                                               stderr='')
        raise AssertionError(f'unexpected command {cmd}')

    @staticmethod
    def check_simulation_success(output_file):
        p = Path(output_file)
        return p.exists() and p.read_text().startswith('OK')

    def _rng(self, epoch_dir: Path):
        return random.Random(_seed(epoch_dir.parent.name, epoch_dir.name,
                                   self.attempts.get(str(epoch_dir), 0)))

    def compute_msld_reward_improved(self, epoch_dir, **kwargs):
        return round(self._rng(Path(epoch_dir)).uniform(-3, 3), 6)

    def parse_simulation_metrics(self, output_file):
        epoch_dir = Path(output_file).parent
        labels = _labels(epoch_dir.parent)
        rng = self._rng(epoch_dir)
        n = len(labels)
        sites = sorted({s for s, _ in labels})
        node_site = [sites.index(s) for s, _ in labels]
        m = {'populations': [rng.randint(10, 400) for _ in range(n)],
             'population_sites': node_site,
             'transitions': [rng.randint(5, 40) for _ in sites], 'ddg_pairs': {},
             'fraction_physical': 0.8, 'barrier_pairs': {}, 'skew_pairs': {}, 'end_pairs': {},
             'transition_pairs': {}, 'endpoint_pairs': {}, 'site_holdoff': []}
        for i in range(n):
            for j in range(i + 1, n):
                if node_site[i] == node_site[j]:
                    k = f"{i + 2}_{j + 2}"
                    m['ddg_pairs'][k] = round(rng.uniform(-4, 4), 4)
                    for f in ('barrier_pairs', 'skew_pairs', 'end_pairs'):
                        m[f][k] = round(rng.uniform(-2, 6), 4)
                    m['transition_pairs'][k] = rng.randint(3, 25)
                    edge = rng.randint(100, 500)
                    m['endpoint_pairs'][k] = {'edge': edge, 'at_lo': edge // 4, 'near_lo': 0,
                                              'near_hi': 0, 'at_hi': edge // 4}
        return m


_FAKED = ('build_graph_and_data', 'ensure_combo_dir_exists', 'submit_epoch_simulation',
          'check_simulation_success', 'compute_msld_reward_improved', 'parse_simulation_metrics')


@pytest.fixture
def fake_cluster(monkeypatch):
    """Patch the cluster-facing functions everywhere the training code
    imported them; returns the FakeCluster (set fail_first / fail_always)."""
    import mllf.cb.training_loop  # noqa: F401 -- load every module the loop uses
    fake = FakeCluster()
    for mod in list(sys.modules.values()):
        if getattr(mod, '__name__', '').startswith('mllf'):
            for name in _FAKED:
                if hasattr(mod, name):
                    monkeypatch.setattr(mod, name, getattr(fake, name))
    monkeypatch.setattr(subprocess, 'run', fake.run)
    monkeypatch.setattr('time.sleep', lambda s: None)
    return fake


def make_system(root: Path, name: str, sites):
    """A system input dir (msld_flat.py, mapping.json, a manifest of every
    within-site pair) and its ``systems:`` entry; returns (entry, combos)."""
    idir = root / 'inputs' / name
    odir = root / 'out' / name
    idir.mkdir(parents=True)
    (idir / 'msld_flat.py').write_text(MSLD_TEMPLATE)
    write_mapping(idir, sites)
    combos = [str(odir / f"pair_site{site}_subs_{subs[a]}_{subs[b]}")
              for site, subs in sites for a in range(len(subs)) for b in range(a + 1, len(subs))]
    (idir / 'manifest.txt').write_text('\n'.join(combos) + '\n')
    entry = {'name': name, 'input_dir': str(idir), 'out_dir': str(odir), 'solvent_state': 'solv'}
    return entry, combos
