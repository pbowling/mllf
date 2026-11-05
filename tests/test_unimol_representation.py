"""The on-disk Uni-Mol embedding cache: keys, layout and lookup."""
import contextlib
import json

import numpy as np
import pytest

from mllf.cb.unimol_representation import (
    _generate_embedding_cache_key,
    _system_label_from_prep_dir,
    load_embedding,
    save_embedding,
)


BASE = dict(sub_pdb='/x/sys/prep/site1_sub1_frag.pdb', core_pdb='/x/sys/prep/core.pdb',
            prep_dir='/x/sys/prep')


def _key(**over):
    args = dict(BASE, env_cutoff=5.0, include_other_sites=False)
    args.update(over)
    return _generate_embedding_cache_key(**args)


def test_key_is_a_deterministic_16_character_hash():
    assert _key() == _key() and len(_key()) == 16


@pytest.mark.parametrize('change', [
    {'env_cutoff': 8.0}, {'include_other_sites': True},
    {'custom_search_paths': ['/a', '/b']}, {'variant': 'ligand_refs'},
    {'consensus_atoms': {('1', 'A', 'CA')}},
])
def test_key_changes_with_each_input(change):
    assert _key(**change) != _key()


def test_no_custom_paths_and_an_empty_list_share_a_key():
    assert _key(custom_search_paths=None) == _key(custom_search_paths=[])


@pytest.mark.parametrize('prep, label', [
    ('/x/pretraining/14benz_solv/prep', '14benz_solv'),
    ('/x/pretraining/12fuzed_vac/comb_1744603_site1_subs_18_19/prep',
     '12fuzed_vac__comb_1744603_site1_subs_18_19'),
])
def test_cache_is_organized_by_system(prep, label):
    assert _system_label_from_prep_dir(prep) == label


def test_save_writes_layout_metadata_and_values(tmp_path):
    emb = np.arange(512, dtype=np.float32)
    path = save_embedding(emb, tmp_path, env_cutoff=5.0, **BASE)
    assert path.relative_to(tmp_path).parts[:3] == ('embeddings', 'sys', 'site1_sub1')
    meta = json.loads(path.with_name(path.stem + '_metadata.json').read_text())
    assert meta['embedding_shape'] == [512] and meta['env_cutoff'] == 5.0
    assert np.array_equal(load_embedding(tmp_path, env_cutoff=5.0, **BASE), emb)


@pytest.mark.parametrize('overwrite, kept', [(False, 1.0), (True, 2.0)])
def test_save_overwrites_only_on_request(tmp_path, overwrite, kept):
    save_embedding(np.ones(512, np.float32), tmp_path, **BASE)
    with pytest.warns(UserWarning) if not overwrite else contextlib.nullcontext():
        path = save_embedding(np.full(512, 2.0, np.float32), tmp_path, overwrite=overwrite,
                              **BASE)
    assert np.load(path)[0] == kept


def test_two_systems_with_the_same_substituent_name_do_not_collide(tmp_path):
    a = save_embedding(np.zeros(512, np.float32), tmp_path,
                       sub_pdb='/x/sysA/comb_0001/prep/site1_sub3_frag.pdb',
                       core_pdb='/x/sysA/comb_0001/prep/core.pdb',
                       prep_dir='/x/sysA/comb_0001/prep')
    b = save_embedding(np.ones(512, np.float32), tmp_path,
                       sub_pdb='/x/sysB/comb_0007/prep/site1_sub3_frag.pdb',
                       core_pdb='/x/sysB/comb_0007/prep/core.pdb',
                       prep_dir='/x/sysB/comb_0007/prep')
    assert a.parent != b.parent


def test_load_misses_on_a_different_key_or_cache(tmp_path):
    save_embedding(np.ones(512, np.float32), tmp_path, env_cutoff=5.0, **BASE)
    assert load_embedding(tmp_path, env_cutoff=8.0, **BASE) is None
    assert load_embedding(tmp_path / 'none', **BASE) is None


def test_load_finds_entries_of_the_older_flat_layout(tmp_path):
    flat = tmp_path / 'embeddings' / 'site1_sub1'
    flat.mkdir(parents=True)
    np.save(flat / f'{_key()}.npy', np.ones(512, np.float32))
    assert load_embedding(tmp_path, env_cutoff=5.0, **BASE) is not None


def test_ligand_entry_is_found_whatever_its_environment_settings(tmp_path):
    """The ligand half does not depend on the environment atoms; an entry
    cached by a run that built them is reused by one that does not."""
    emb = np.full(512, 3.0, np.float32)
    save_embedding(emb, tmp_path, env_cutoff=8.0, consensus_atoms={('1', 'W', 'OH2')},
                   variant='ligand', **BASE)
    assert np.array_equal(load_embedding(tmp_path, env_cutoff=8.0, variant='ligand', **BASE),
                          emb)
    # but not across variants
    assert load_embedding(tmp_path, env_cutoff=8.0, variant='ligand_refs', **BASE) is None
