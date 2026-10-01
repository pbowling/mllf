"""Tests for mllf.file_handling.manifests (migrated from training/workflow.py).

Covers:
  - iter_json_array_streaming  (chunked JSON-array decode)
  - count_combinations_fast    (arithmetic combo count vs. list_possible_combinations)
  - build_system_manifest      (load-existing-manifest path)
  - create_and_load_system_manifest (generate manifest + combo_metadata.json)
  - build_combo_dir_lookup     (missing-combo metadata pre-scan)
  - ensure_combo_dir_exists    (existing / lazy-create / not-found paths)
"""
import json
from pathlib import Path

import pytest

from mllf.file_handling.generate_combinations import list_possible_combinations
from mllf.file_handling.manifests import (
    build_combo_dir_lookup,
    build_system_manifest,
    count_combinations_fast,
    create_and_load_system_manifest,
    ensure_combo_dir_exists,
    iter_json_array_streaming,
)


def _make_site_sub_fixture(base: Path, sites: dict) -> Path:
    """Create a minimal site/sub fragment fixture under base/prep.

    Args:
        sites: {site_id: n_subs}
    """
    prep_dir = base / 'prep'
    prep_dir.mkdir(parents=True, exist_ok=True)
    for site, n in sites.items():
        for sub in range(1, n + 1):
            (prep_dir / f'site{site}_sub{sub}_pres.rtf').write_text(f'PRES p{site}_{sub} 0.0\n')
            (prep_dir / f'site{site}_sub{sub}_frag.pdb').write_text(f'ATOM {sub}\n')
    (prep_dir / 'full_ligand.rtf').write_text('RTF\n')
    (prep_dir / 'full_ligand.pdb').write_text('ATOM 1\n')
    return base


# ---------------------------------------------------------------------------
# iter_json_array_streaming
# ---------------------------------------------------------------------------

def test_iter_json_array_streaming_matches_json_load(tmp_path):
    data = [{'path': f'combo_{i}', 'sites': [1], 'subs': [i]} for i in range(25)]
    p = tmp_path / 'combo_metadata.json'
    p.write_text(json.dumps(data))

    # Small chunk_size forces the reader across many chunk boundaries,
    # including mid-object and mid-array-separator splits.
    streamed = list(iter_json_array_streaming(p, chunk_size=7))
    assert streamed == data


def test_iter_json_array_streaming_empty_array(tmp_path):
    p = tmp_path / 'empty.json'
    p.write_text('[]')
    assert list(iter_json_array_streaming(p, chunk_size=4)) == []


# ---------------------------------------------------------------------------
# count_combinations_fast
# ---------------------------------------------------------------------------

def test_count_combinations_fast_matches_list_possible_combinations(tmp_path):
    input_dir = _make_site_sub_fixture(tmp_path / 'sys_a', {1: 4, 2: 3})
    out_dir = tmp_path / 'sys_a_out'

    fast_count = count_combinations_fast({'input_dir': str(input_dir)}, max_subs_per_site=10)
    real_combos = list_possible_combinations(input_dir, out_dir, max_subs_per_site=10)

    assert fast_count == len(real_combos)
    assert fast_count > 0


def test_count_combinations_fast_excludes_single_sub_sites(tmp_path):
    # A site with only 1 substituent can't form a combination and must be
    # excluded from the arithmetic count entirely (list_possible_combinations
    # would raise for this input; count_combinations_fast must not).
    input_dir = _make_site_sub_fixture(tmp_path / 'sys_b', {1: 1})
    assert count_combinations_fast({'input_dir': str(input_dir)}) == 0


def test_count_combinations_fast_no_site_sub_files(tmp_path):
    empty_dir = tmp_path / 'empty_sys'
    empty_dir.mkdir()
    assert count_combinations_fast({'input_dir': str(empty_dir)}) == 0


# ---------------------------------------------------------------------------
# build_system_manifest / create_and_load_system_manifest
# ---------------------------------------------------------------------------

def test_build_system_manifest_missing_raises(tmp_path):
    input_dir = tmp_path / 'no_manifest'
    input_dir.mkdir()
    sys_entry = {'name': 'sys', 'input_dir': str(input_dir), 'out_dir': str(tmp_path / 'out')}
    with pytest.raises(FileNotFoundError):
        build_system_manifest(sys_entry)


def test_create_and_load_system_manifest_writes_manifest_and_metadata(tmp_path):
    input_dir = _make_site_sub_fixture(tmp_path / 'sys_c', {1: 3})
    out_dir = tmp_path / 'sys_c_out'
    sys_entry = {'name': 'sys_c', 'input_dir': str(input_dir), 'out_dir': str(out_dir)}

    result = create_and_load_system_manifest(sys_entry, max_subs_per_site=10)

    assert (input_dir / 'manifest.txt').exists()
    assert (input_dir / 'combo_metadata.json').exists()
    assert result['sys_entry'] is sys_entry
    assert len(result['combos']) == count_combinations_fast(sys_entry)
    assert any('Found' in line for line in result['log'])


def test_create_and_load_system_manifest_respects_max_combos(tmp_path):
    input_dir = _make_site_sub_fixture(tmp_path / 'sys_d', {1: 5})
    out_dir = tmp_path / 'sys_d_out'
    sys_entry = {'name': 'sys_d', 'input_dir': str(input_dir), 'out_dir': str(out_dir)}

    full = create_and_load_system_manifest(sys_entry, max_subs_per_site=10)
    assert len(full['combos']) > 3

    # Re-load (manifest.txt already exists) via build_system_manifest, capped.
    capped = build_system_manifest(sys_entry, max_combos=3, seed=42)
    assert len(capped['combos']) == 3
    # Deterministic given the same seed.
    capped_again = build_system_manifest(sys_entry, max_combos=3, seed=42)
    assert capped['combos'] == capped_again['combos']


# ---------------------------------------------------------------------------
# build_combo_dir_lookup
# ---------------------------------------------------------------------------

def test_build_combo_dir_lookup_finds_only_missing_entries(tmp_path):
    input_dir = _make_site_sub_fixture(tmp_path / 'sys_e', {1: 3})
    out_dir = tmp_path / 'sys_e_out'
    sys_entry = {'name': 'sys_e', 'input_dir': str(input_dir), 'out_dir': str(out_dir)}
    result = create_and_load_system_manifest(sys_entry)
    combos = result['combos']
    combo_system_map = {c: sys_entry for c in combos}

    # None of the combo dirs have actually been created on disk yet.
    lookup = build_combo_dir_lookup(combos, combo_system_map)
    assert set(lookup.keys()) == set(combos)
    for c in combos:
        assert lookup[c]['path'] == c

    # Create one directory for real; it should drop out of the "missing" set.
    Path(combos[0]).mkdir(parents=True)
    lookup2 = build_combo_dir_lookup(combos, combo_system_map)
    assert combos[0] not in lookup2
    assert set(lookup2.keys()) == set(combos[1:])


def test_build_combo_dir_lookup_empty_when_all_exist(tmp_path):
    d = tmp_path / 'already_here'
    d.mkdir()
    combo_system_map = {str(d): {'input_dir': str(tmp_path)}}
    assert build_combo_dir_lookup([str(d)], combo_system_map) == {}


# ---------------------------------------------------------------------------
# ensure_combo_dir_exists
# ---------------------------------------------------------------------------

def test_ensure_combo_dir_exists_returns_existing_path_unchanged(tmp_path):
    existing = tmp_path / 'comb_already_made'
    existing.mkdir()
    result = ensure_combo_dir_exists(existing, system_config={}, global_config={})
    assert result == existing


def test_ensure_combo_dir_exists_creates_lazily_from_lookup(tmp_path):
    input_dir = _make_site_sub_fixture(tmp_path / 'sys_f', {1: 3})
    out_dir = tmp_path / 'sys_f_out'
    system_config = {'name': 'sys_f', 'input_dir': str(input_dir), 'out_dir': str(out_dir)}

    real_combos = list_possible_combinations(input_dir, out_dir, max_subs_per_site=10)
    assert real_combos, "fixture should produce at least one combination"
    combo_info = real_combos[0]
    combo_path = Path(combo_info['path'])
    assert not combo_path.exists()

    lookup = {combo_info['path']: combo_info}
    created = ensure_combo_dir_exists(combo_path, system_config, {}, _lookup=lookup)

    assert created == combo_path
    assert combo_path.is_dir()
    assert (combo_path / 'prep').is_dir()
    assert (combo_path / 'mapping.json').exists()


def test_ensure_combo_dir_exists_raises_when_info_not_found(tmp_path):
    input_dir = tmp_path / 'sys_g'
    input_dir.mkdir()
    (input_dir / 'combo_metadata.json').write_text('[]')
    system_config = {'input_dir': str(input_dir), 'out_dir': str(tmp_path / 'sys_g_out')}
    missing_combo = tmp_path / 'sys_g_out' / 'comb_not_real'

    with pytest.raises(ValueError):
        ensure_combo_dir_exists(missing_combo, system_config, {})


def test_ensure_combo_dir_exists_raises_when_metadata_missing(tmp_path):
    input_dir = tmp_path / 'sys_h'
    input_dir.mkdir()
    system_config = {'input_dir': str(input_dir), 'out_dir': str(tmp_path / 'sys_h_out')}
    missing_combo = tmp_path / 'sys_h_out' / 'comb_not_real'

    with pytest.raises(FileNotFoundError):
        ensure_combo_dir_exists(missing_combo, system_config, {})
