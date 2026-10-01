"""Tests for mllf.cb.curriculum (migrated from training/workflow.py).

Covers:
  - build_system_index
  - filter_combos_by_curriculum
  - filter_combos_by_transitions
  - apply_stage_combo_filter
  - sample_combos_stratified
"""
from pathlib import Path

import torch

from mllf.cb.curriculum import (
    apply_stage_combo_filter,
    build_system_index,
    filter_combos_by_curriculum,
    filter_combos_by_transitions,
    sample_combos_stratified,
)


# ---------------------------------------------------------------------------
# build_system_index
# ---------------------------------------------------------------------------

def test_build_system_index_reserves_unknown_and_orders_by_config():
    idx = build_system_index([{'name': 'sysA'}, {'name': 'sysB'}])
    assert idx == {'__unknown__': 0, 'sysA': 1, 'sysB': 2}


def test_build_system_index_falls_back_to_positional_name():
    idx = build_system_index([{}, {'name': 'named'}])
    assert idx['system_0'] == 1
    assert idx['named'] == 2


# ---------------------------------------------------------------------------
# filter_combos_by_curriculum
# ---------------------------------------------------------------------------

def _combo_names():
    return [
        Path('comb_0001_site1_subs_2_3'),                       # 1 site, 2 subs
        Path('comb_0002_site1_subs_1_2_3'),                     # 1 site, 3 subs
        Path('comb_0003_site1_subs_1_2__site2_subs_3_4'),       # 2 sites, 2 subs each
        Path('not_a_combo_dir'),                                # no site/subs match -> skipped
    ]


def test_filter_combos_by_curriculum_site_and_sub_bounds():
    combos = _combo_names()
    single_site_pairs = filter_combos_by_curriculum(
        combos, min_sites=1, max_sites=1, min_subs_per_site=2, max_subs_per_site=2,
    )
    assert [c.name for c in single_site_pairs] == ['comb_0001_site1_subs_2_3']

    cross_site = filter_combos_by_curriculum(
        combos, min_sites=2, max_sites=2, min_subs_per_site=1, max_subs_per_site=None,
    )
    assert [c.name for c in cross_site] == ['comb_0003_site1_subs_1_2__site2_subs_3_4']


def test_filter_combos_by_curriculum_whitelist():
    combos = _combo_names()
    result = filter_combos_by_curriculum(
        combos, min_sites=1, max_sites=2,
        whitelist=['comb_0002_site1_subs_1_2_3'],
    )
    assert [c.name for c in result] == ['comb_0002_site1_subs_1_2_3']


# ---------------------------------------------------------------------------
# filter_combos_by_transitions
# ---------------------------------------------------------------------------

def _make_run_with_transitions(combo_dir: Path, run_idx: int, transitions: dict) -> None:
    run_dir = combo_dir / f'run_{run_idx:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'transitions': transitions}, run_dir / 'epoch_results.pt')


def test_filter_combos_by_transitions_qualifies_on_min_across_recent_runs(tmp_path):
    good = tmp_path / 'comb_good'
    bad = tmp_path / 'comb_bad'
    good.mkdir()
    bad.mkdir()
    _make_run_with_transitions(good, 0, {'site0': 12, 'site1': 15})
    _make_run_with_transitions(bad, 0, {'site0': 2, 'site1': 15})

    qualifying = filter_combos_by_transitions([good, bad], min_transitions=10)
    assert qualifying == [good]


def test_filter_combos_by_transitions_respects_lookback(tmp_path):
    combo = tmp_path / 'comb_lookback'
    combo.mkdir()
    # Old run (outside lookback window) qualifies; two recent runs don't.
    _make_run_with_transitions(combo, 0, {'site0': 20})
    _make_run_with_transitions(combo, 1, {'site0': 1})
    _make_run_with_transitions(combo, 2, {'site0': 1})

    assert filter_combos_by_transitions([combo], min_transitions=10, lookback=2) == []
    assert filter_combos_by_transitions([combo], min_transitions=10, lookback=3) == [combo]


def test_filter_combos_by_transitions_no_runs_does_not_qualify(tmp_path):
    empty_combo = tmp_path / 'comb_empty'
    empty_combo.mkdir()
    assert filter_combos_by_transitions([empty_combo], min_transitions=1) == []


# ---------------------------------------------------------------------------
# apply_stage_combo_filter
# ---------------------------------------------------------------------------

def test_apply_stage_combo_filter_combines_curriculum_and_transition_filters(tmp_path):
    combo_a = tmp_path / 'comb_0001_site1_subs_1_2'
    combo_b = tmp_path / 'comb_0002_site1_subs_1_2_3'
    combo_a.mkdir()
    combo_b.mkdir()
    _make_run_with_transitions(combo_a, 0, {'site0': 20})  # qualifies

    stage = {
        'min_sites': 1, 'max_sites': 1,
        'min_subs_per_site': 2, 'max_subs_per_site': 2,
        'min_transitions_for_stage': 10,
        'transition_filter_lookback': 3,
    }
    result = apply_stage_combo_filter(
        [combo_a, combo_b], stage, curriculum_config={}, stage_idx=0, seed=0,
        apply_transition_filter=True,
    )
    assert result == [combo_a]


def test_apply_stage_combo_filter_skips_transition_filter_when_disabled(tmp_path):
    combo_a = tmp_path / 'comb_0001_site1_subs_1_2'
    combo_a.mkdir()  # no runs at all -- would fail the transition filter

    stage = {
        'min_sites': 1, 'max_sites': 1,
        'min_subs_per_site': 2, 'max_subs_per_site': 2,
        'min_transitions_for_stage': 10,
    }
    result = apply_stage_combo_filter(
        [combo_a], stage, curriculum_config={}, stage_idx=0, seed=0,
        apply_transition_filter=False,
    )
    assert result == [combo_a]


# ---------------------------------------------------------------------------
# sample_combos_stratified
# ---------------------------------------------------------------------------

def test_sample_combos_stratified_respects_min_per_system_and_total():
    pools = {
        'small_sys': [f'small_{i}' for i in range(3)],
        'big_sys': [f'big_{i}' for i in range(100)],
    }
    sampled = sample_combos_stratified(pools, n_total=20, min_per_system=3, seed=0)

    assert len(sampled) <= 20
    n_small = sum(1 for c in sampled if c.startswith('small_'))
    n_big = sum(1 for c in sampled if c.startswith('big_'))
    assert n_small == 3  # capped at the small system's pool size
    assert n_big >= 3
    assert n_small + n_big == len(sampled)


def test_sample_combos_stratified_is_deterministic_given_seed():
    pools = {'sys_a': [f'a_{i}' for i in range(50)], 'sys_b': [f'b_{i}' for i in range(50)]}
    first = sample_combos_stratified(pools, n_total=20, min_per_system=5, seed=7)
    second = sample_combos_stratified(pools, n_total=20, min_per_system=5, seed=7)
    assert first == second


def test_sample_combos_stratified_reduces_min_per_system_if_infeasible():
    pools = {'a': list(range(10)), 'b': list(range(10)), 'c': list(range(10))}
    # min_per_system=10 * 3 systems = 30 > n_total=6, so it must reduce
    # min_per_system rather than raise or exceed n_total.
    sampled = sample_combos_stratified(pools, n_total=6, min_per_system=10, seed=1)
    assert len(sampled) <= 6
