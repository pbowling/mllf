"""Solvent legs take their Uni-Mol environment from the solvent, not a protein
file left in their prep (e.g. the luis_* solvent systems ship proa.pdb)."""

from pathlib import Path

import numpy as np
import pytest

from mllf.cb import unimol_representation as ur


def _pdb(path: Path, atoms, segid):
    """atoms: [(name, resname, resnum, x, y, z, element)]"""
    lines = [f"ATOM  {k + 1:5d} {n:<4s} {r:<4s} {i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00      {segid:<4s}{e:>2s}"
             for k, (n, r, i, x, y, z, e) in enumerate(atoms)]
    path.write_text("\n".join(lines + ["END"]) + "\n")


def _prep(root: Path, system: str) -> Path:
    prep = root / system / 'comb_0001_site1_subs_1_2' / 'prep'
    prep.mkdir(parents=True)
    _pdb(prep / 'core.pdb', [('C1', 'LIG', 1, 0.0, 0.0, 0.0, 'C')], 'LIG')
    _pdb(prep / 'proa.pdb', [('CA', 'GLY', 10, 1.0, 0.0, 0.0, 'C'),
                            ('N', 'GLY', 10, 0.0, 1.5, 0.0, 'N')], 'PROA')
    _pdb(prep / 'solvent.pdb', [('OH2', 'TIP3', 1, 3.0, 0.0, 0.0, 'O'),
                               ('OH2', 'TIP3', 2, 0.0, 3.0, 0.0, 'O'),
                               ('OH2', 'TIP3', 3, 0.0, 0.0, 3.0, 'O')], 'SOLV')
    return prep


@pytest.mark.parametrize('system, solvent', [
    ('luis_bace_solvent', True), ('faah_solv', True), ('abl_water', True),
    ('indolizine_solv', True), ('luis_bace_protein', False), ('indolizine_prot', False),
    ('1benz_vac', False),
])
def test_solvent_phase_from_system_name(tmp_path, system, solvent):
    prep = tmp_path / system / 'pair01_site1_subs_1_2' / 'prep'
    assert ur._is_solvent_phase_prep(prep) is solvent
    assert ur._is_solvent_phase_prep(tmp_path / system / 'prep') is solvent


def test_solvent_leg_environment_skips_protein_file(tmp_path):
    prep = _prep(tmp_path, 'luis_bace_solvent')
    consensus = ur.build_environment_consensus(str(prep / 'core.pdb'), prep, env_cutoff=8.0)
    assert len(consensus) == 3                       # the three waters, not the 2 protein atoms
    coords, elements = ur._load_environment_pdb(
        prep, sub_coords=np.zeros((1, 3)), ligand_cutoff=8.0, skip_minimized=True)
    assert sorted(elements) == ['O', 'O', 'O']


def test_protein_leg_still_uses_protein_file(tmp_path):
    prep = _prep(tmp_path, 'luis_bace_protein')
    consensus = ur.build_environment_consensus(str(prep / 'core.pdb'), prep, env_cutoff=8.0)
    assert len(consensus) == 2
    coords, elements = ur._load_environment_pdb(
        prep, sub_coords=np.zeros((1, 3)), ligand_cutoff=8.0, skip_minimized=True)
    assert sorted(elements) == ['C', 'N']


def test_cache_key_changes_only_for_affected_solvent_legs(tmp_path):
    affected = _prep(tmp_path, 'luis_bace_solvent')
    clean = _prep(tmp_path, '1benz_solvent')
    (clean / 'proa.pdb').unlink()
    protein = _prep(tmp_path, 'luis_bace_protein')
    args = dict(env_cutoff=8.0, include_other_sites=False)

    def key(prep, marked):
        k = ur._generate_embedding_cache_key(str(prep / 'site1_sub1_frag.pdb'), str(prep / 'core.pdb'),
                                             str(prep), **args)
        orig = ur._solvent_leg_with_protein_file
        ur._solvent_leg_with_protein_file = lambda p: marked
        try:
            unmarked = ur._generate_embedding_cache_key(str(prep / 'site1_sub1_frag.pdb'),
                                                        str(prep / 'core.pdb'), str(prep), **args)
        finally:
            ur._solvent_leg_with_protein_file = orig
        return k, unmarked

    k, k_marked = key(affected, True)
    assert k == k_marked                             # affected solvent leg carries the marker
    for prep in (clean, protein):
        k, k_unmarked = key(prep, False)
        assert k == k_unmarked                       # everything else keeps its old key
