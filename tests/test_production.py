"""Production replicas and their end-state free energies."""
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from mllf import production as prod
from mllf.analysis import estimators as est
from mllf.file_handling.read_lambda import read_lmd


KT = est.KB * 298.15


def _rec(payload: bytes) -> bytes:
    return struct.pack('<i', len(payload)) + payload + struct.pack('<i', len(payload))


def write_lmd(path: Path, L: np.ndarray, b, nsavl: int = 10, nfile=None) -> None:
    """A lambda file as CHARMM writes it (environment block first)."""
    n, nb = L.shape
    icntrl = np.zeros(20, '<i4')
    icntrl[0], icntrl[2], icntrl[6], icntrl[10] = nfile or n, nsavl, nb + 1, 2
    parts = [_rec(b'MSLD' + icntrl.tobytes()),
             _rec(np.array([0.002 / 0.04888821], '<f4').tobytes()),
             _rec(struct.pack('<i', 1) + b' ' * 80),
             _rec(struct.pack('<i', 0)), _rec(b''),
             _rec(np.ones(nb + 1, '<i4').tobytes()),
             _rec(np.array([298.15], '<f4').tobytes()),
             _rec(np.concatenate([[0.0], b]).astype('<f4').tobytes())]
    for row in L:
        parts.append(_rec(np.concatenate([[1.0], row]).astype('<f4').tobytes()))
        parts.append(_rec(np.zeros(nb, '<f4').tobytes()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(parts))


def variables_text(b, c=None, x=None, s=None) -> str:
    nb = len(b)
    z = np.zeros((nb, nb)).tolist()
    data = {'b': [list(map(float, b))], 'c': c or z, 'x': x or z, 's': s or z}
    return f"bias_string = '''\n{yaml.safe_dump(data)}'''\n"


def sample_frames(rng, g, b_set, nsubs, n, gap=0.1):
    """Frames whose end states follow exp(-(G - E_end)/kT) (E_end = sum of
    b at one-hot lambdas), with a fraction ``gap`` of frames between end
    states at one site."""
    combos = est.combinations(nsubs)
    terms = est.bias_terms({'b': b_set}, nsubs)
    logp = -(g - est.endpoint_energy(combos, terms, nsubs)) / KT
    p = np.exp(logp - logp.max())
    p /= p.sum()
    pick = combos[rng.choice(len(combos), n, p=p)]
    L = np.zeros((n, sum(nsubs)), np.float32)
    o = est.offsets(nsubs)
    for i in range(len(nsubs)):
        L[np.arange(n), o[i] + pick[:, i] - 1] = 1.0
    mid = rng.random(n) < gap
    L[mid, :nsubs[0]] = 1.0 / nsubs[0]
    return L


def test_lambda_file_is_read_with_its_header(tmp_path):
    L = np.random.default_rng(0).random((7, 3)).astype(np.float32)
    write_lmd(tmp_path / 'a.lmd', L, [0.0, -3.0, 2.5])
    F, hdr = read_lmd(tmp_path / 'a.lmd')
    assert F.shape == (7, 3) and np.allclose(F, L)
    assert hdr['nblocks'] == 3 and hdr['dt_ps'] == pytest.approx(0.02, rel=1e-5)
    assert np.allclose(hdr['fixed_bias'], [0.0, -3.0, 2.5])


def test_lambda_file_cut_short_gives_the_frames_written(tmp_path):
    L = np.ones((5, 2), np.float32)
    write_lmd(tmp_path / 'a.lmd', L, [0, 0], nfile=100)
    path = tmp_path / 'a.lmd'
    path.write_bytes(path.read_bytes()[:-10])
    F, hdr = read_lmd(path)
    assert len(F) == 4 and hdr['frames_in_header'] == 100


def test_bias_energy_applies_only_the_terms_the_simulation_writes():
    terms = est.bias_terms({'b': [0, 1.5], 'c': [[9, 2.0], [7.0, 9]],
                            'x': [[5, 0.5], [0.25, 5]], 's': [[3, 0.2], [0.1, 3]]}, [2])
    lam = np.array([[0.3, 0.7]])
    expect = (1.5 * 0.7 + 2.0 * 0.3 * 0.7
              + 0.5 * (1 - np.exp(-5.56 * 0.3)) * 0.7 + 0.25 * (1 - np.exp(-5.56 * 0.7)) * 0.3
              + 0.2 * 0.3 / 0.317 * 0.7 + 0.1 * 0.7 / 0.717 * 0.3)
    assert est.bias_energy(lam, terms)[0] == pytest.approx(expect)
    # at an end state of a single site only b is left
    ends = est.endpoint_energy(np.array([[1], [2]]), terms, [2])
    assert np.allclose(ends, [0.0, 1.5])


def test_one_site_wham_and_potts_recover_the_free_energies():
    rng = np.random.default_rng(1)
    nsubs = [3]
    g = np.array([0.0, -4.0, 6.0])
    b = g + np.array([0.0, 0.3, -0.2])           # nearly flat biased landscape
    terms = est.bias_terms({'b': b}, nsubs)
    reps = [{'L': sample_frames(rng, g, b, nsubs, 20000), 'terms': terms} for _ in range(4)]
    res = est.analyze_replicas(reps, nsubs, 298.15, n_boot=10)
    assert res['reference'] == 0
    assert np.allclose(res['wham'], g, atol=0.08)
    assert np.allclose(res['potts'], res['wham'], atol=1e-3)     # one site: the same estimate
    assert np.all(res['wham_sd'][1:] > 0) and np.all(res['wham_sd'][1:] < 0.1)


def test_two_bias_sets_are_combined_by_wham():
    rng = np.random.default_rng(2)
    nsubs = [2]
    g = np.array([0.0, 3.0])
    reps = []
    for b in ([0.0, 3.5], [0.0, 2.0]):
        reps.append({'L': sample_frames(rng, g, np.array(b), nsubs, 30000, gap=0.0),
                     'terms': est.bias_terms({'b': b}, nsubs)})
    res = est.analyze_replicas(reps, nsubs, 298.15, n_boot=0)
    assert res['bias_sets'] == 2 and res['potts'] is None
    assert res['wham'][1] == pytest.approx(3.0, abs=0.06)


def test_potts_estimates_a_combination_never_sampled_together():
    """Two independent sites: G of (2, 2) is the sum of the single changes,
    and the Potts model gives it from (1, 2) and (2, 1) alone."""
    rng = np.random.default_rng(3)
    nsubs = [2, 2]
    g = np.array([0.0, 1.0, 2.0, 3.0])                 # (1,1) (1,2) (2,1) (2,2)
    b = [0.0, 1.8, 0.0, 1.2]
    terms = est.bias_terms({'b': b}, nsubs)
    reps = [{'L': sample_frames(rng, g, np.array(b), nsubs, 40000, gap=0.0), 'terms': terms}
            for _ in range(3)]
    res = est.analyze_replicas(reps, nsubs, 298.15, n_boot=5)
    assert np.allclose(res['wham'], g, atol=0.08)
    assert np.allclose(res['potts'], g, atol=0.08)
    # with (2, 2) never sampled, its pair is unsampled and neither gives a value
    for r in reps:
        st = est.frame_states(r['L'], nsubs)
        r['L'] = r['L'][~np.all(st == 2, axis=1)]
    res = est.analyze_replicas(reps, nsubs, 298.15, n_boot=0)
    assert np.isnan(res['wham'][3]) and np.isnan(res['potts'][3])


def test_potts_estimates_a_rarely_sampled_combination_from_its_pairs():
    """Three independent sites with substituent 2 at 6% of frames at each:
    (2, 2, 2) is seen a few times per replica, its pairs ~20x as often, so
    the Potts value is close to the truth and less noisy than the histogram."""
    rng = np.random.default_rng(5)
    nsubs = [2, 2, 2]
    combos = est.combinations(nsubs)
    dg = np.array([1.0, 2.0, 0.5])
    g = ((combos - 1) @ dg).astype(float)
    shift = KT * np.log(0.06 / 0.94)
    b = [v for d in dg for v in (0.0, d + shift)]
    terms = est.bias_terms({'b': b}, nsubs)
    reps = [{'L': sample_frames(rng, g, np.array(b), nsubs, 5000, gap=0.0), 'terms': terms}
            for _ in range(6)]
    res = est.analyze_replicas(reps, nsubs, 298.15, n_boot=30)
    last = len(combos) - 1
    assert res['counts'][last] < 30
    assert res['potts'][last] == pytest.approx(3.5, abs=0.2)
    assert res['potts_sd'][last] < res['wham_sd'][last]


def test_sampling_counts_fractions_entries_and_transitions():
    L = np.array([[1, 0], [1, 0], [0.5, 0.5], [0, 1], [1, 0], [0, 1]], np.float32)
    s = est.sampling(L, [2], dt_ps=1000.0)            # 1 ns per frame
    assert s['fraction'][0] == pytest.approx([3 / 6, 2 / 6])
    assert s['transitions_per_ns'][0] == pytest.approx(3 / 6)
    assert s['entries_per_ns'][0] == pytest.approx([2 / 6, 2 / 6])
    assert s['physical'] == pytest.approx(5 / 6)


# Setting up and analyzing a production

def _combo(tmp_path: Path, nsubs=(3,)) -> Path:
    combo = tmp_path / 'group01_site1_subs_4_5_6'
    prep = combo / 'prep'
    prep.mkdir(parents=True)
    for i, n in enumerate(nsubs):
        for j in range(n):
            (prep / f'site{i + 1}_sub{j + 1}_pres.rtf').write_text('ATOM X\n')
    (prep / 'minimized.crd').write_text('*\n')
    (combo / 'msld_flat.py').write_text('timestep = 0.002\nesteps = 25000\nnsteps = 50000\n')
    (combo / 'info.py').write_text("info = {}\ninfo['temp'] = 298.15\n")
    (combo / 'mapping.json').write_text(
        '[' + ','.join(f'{{"new_site": 1, "new_sub": {j + 1}, "original_sub": {j + 4}}}'
                       for j in range(nsubs[0])) + ']')
    return combo


def test_production_setup_writes_self_contained_replicas(tmp_path, monkeypatch):
    combo = _combo(tmp_path)
    var = tmp_path / 'variables.py'
    var.write_text(variables_text([0.0, -1.0, 2.0]))
    submitted = []
    monkeypatch.setattr(prod, '_sbatch', lambda script, cwd, dep=None: (
        submitted.append((script.name, dep)) or str(100 + len(submitted))))
    monkeypatch.setattr(prod, '_queued', lambda: set())
    config = {'combo': str(combo), 'variables': str(var), 'out_dir': str(tmp_path / 'prod'),
              'replicas': 2, 'ns': 2.0, 'slurm': {'partition': 'gpu'}}
    out = prod.submit(config)
    d = tmp_path / 'prod' / 'replica_01'
    assert (d / 'prep' / 'site1_sub3_pres.rtf').exists() and (d / 'msld_flat.py').exists()
    assert (d / 'mapping.json').exists()
    text = (d / 'variables.py').read_text()
    assert 'esteps = 25000' in text and 'nsteps = 1000000' in text
    assert "info['nsubs'] = [3]" in (d / 'info.py').read_text()
    assert '--partition=gpu' in (d / 'run.sh').read_text()
    assert submitted[-1] == ('analysis.sh', 'afterany:101:102')
    assert out['analysis'] == '103'
    # a second run leaves queued replicas alone
    monkeypatch.setattr(prod, '_queued', lambda: {'101', '102'})
    submitted.clear()
    out = prod.submit(config)
    assert out['replicas'] == {'replica_01': 'queued', 'replica_02': 'queued'}
    assert [s for s, _ in submitted] == ['analysis.sh']


def test_production_inputs_are_checked(tmp_path):
    combo = _combo(tmp_path)
    var = tmp_path / 'variables.py'
    var.write_text(variables_text([0.0, -1.0]))
    with pytest.raises(ValueError, match='b has 2 entries'):
        prod.resolve({'combo': str(combo), 'variables': str(var), 'out_dir': str(tmp_path / 'p')})
    with pytest.raises(ValueError, match='variables is not set'):
        prod.resolve({'combo': str(combo), 'out_dir': str(tmp_path / 'p')})


def test_production_analysis_writes_free_energies_and_report(tmp_path, monkeypatch):
    pytest.importorskip('pandas')
    pytest.importorskip('matplotlib')
    pytest.importorskip('scipy')
    from mllf.analysis.production import analyze_production
    combo = _combo(tmp_path)
    b = [0.0, -3.7, 5.8]
    var = tmp_path / 'variables.py'
    var.write_text(variables_text(b))
    monkeypatch.setattr(prod, '_sbatch', lambda *a, **k: '1')
    monkeypatch.setattr(prod, '_queued', lambda: set())
    out = tmp_path / 'prod'
    prod.submit({'combo': str(combo), 'variables': str(var), 'out_dir': str(out),
                 'replicas': 3, 'ns': 1.0, 'analysis': {'run': False}})
    rng = np.random.default_rng(4)
    g = np.array([0.0, -4.0, 6.0])
    for k in (1, 2, 3):
        L = sample_frames(rng, g, np.array(b), [3], 50000)
        write_lmd(out / f'replica_0{k}' / 'res' / 'x_flat.lmd', L, b)
    report = analyze_production(out, n_boot=10).read_text()
    import pandas as pd
    df = pd.read_csv(out / 'analysis' / 'tables' / 'free_energies.csv')
    assert list(df['site1_system']) == [4, 5, 6]
    assert np.allclose(df['G_wham'], g, atol=0.1)
    assert 'Relative to 4' in report and 'replica_03' in report
    for fig in ('populations', 'lambda_trajectories', 'profiles'):
        assert (out / 'analysis' / 'figures' / f'{fig}.png').exists()


def test_analysis_command_puts_a_source_tree_on_the_path(monkeypatch, tmp_path):
    import site

    import mllf
    src = Path(mllf.__file__).resolve().parents[1]
    monkeypatch.setattr(site, 'getsitepackages', lambda: [str(tmp_path)])
    monkeypatch.setattr(site, 'getusersitepackages', lambda: str(tmp_path / 'user'))
    lines = prod.analysis_command(tmp_path / 'out')
    assert len(lines) == 2 and lines[0].startswith(f'export PYTHONPATH={src}')
    assert lines[1].endswith('-m mllf production analyze ' + str(tmp_path / 'out'))
    # installed: the interpreter finds this copy without PYTHONPATH
    monkeypatch.setattr(site, 'getsitepackages', lambda: [str(src)])
    assert prod.analysis_command(tmp_path / 'out') == lines[1:]
