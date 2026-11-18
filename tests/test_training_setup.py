"""Training setup helpers (mllf.cb.training_setup)."""
import json
import warnings

from mllf.cb.training_setup import choose_validation
from mllf.cb.warning_summary import WarningSummary


def _choose(tmp_path, config, start_epoch=0, deploy=False, active=None):
    active = active if active is not None else [f'/o/{s}/pair_site1_subs_1_{k}'
                                                for s in 'AB' for k in range(2, 6)]
    cmap = {c: {'name': c.split('/')[2]} for c in active}
    return choose_validation(config, deploy=deploy, start_epoch=start_epoch, num_epochs=3,
                             active=list(active), train_combos=list(active), combo_system_map=cmap,
                             combo_sets_by_system={}, curriculum_cfg={}, curriculum_enabled=False,
                             seed=0, path=tmp_path / 'validation_combos.json')


def test_validation_per_system_is_drawn_persisted_and_restored(tmp_path):
    val, rest = _choose(tmp_path, {'validation_per_system': 2})
    assert len(val) == 4 and not set(val) & set(rest) and len(rest) == 4
    assert {c.split('/')[2] for c in val} == {'A', 'B'}
    assert json.loads((tmp_path / 'validation_combos.json').read_text()) == val
    # a resumed run restores the same set
    val2, rest2 = _choose(tmp_path, {'validation_per_system': 2}, start_epoch=1)
    assert val2 == val and rest2 == rest


def test_no_validation_in_deploy_mode_or_with_zero_per_system(tmp_path):
    for cfg, deploy in (({'validation_per_system': 2}, True),
                        ({'validation_per_system': 0}, False)):
        val, rest = _choose(tmp_path, cfg, deploy=deploy)
        assert val == [] and len(rest) == 8


def test_warning_summary_deduplicates_and_restores_the_handler(capsys):
    original = warnings.showwarning
    with WarningSummary():
        for p in ('/a/x.pt', '/b/y.pt'):
            warnings.warn(f"cache at {p} skipped")
    assert warnings.showwarning is original
    out = capsys.readouterr().out
    assert '2 total occurrences, 1 unique types' in out and '[    2x] UserWarning' in out
