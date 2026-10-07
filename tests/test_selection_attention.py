"""select_combos_by_uncertainty with a pair-attention term."""
from types import SimpleNamespace

from mllf.cb.workflow_utils import select_combos_by_uncertainty

VARIANCE = {'a': 3.0, 'b': 2.0, 'c': 1.0}


def _select(attention, weight):
    return select_combos_by_uncertainty(
        list(VARIANCE), policy=None, combo_system_map={}, config={}, n_select=1,
        bandit_state={}, device='cpu',
        build_graph_fn=lambda combo, sys_cfg, cfg, dev: (SimpleNamespace(name=combo), None),
        score_fn=lambda policy, data, device: VARIANCE[data.name],
        attention_fn=lambda combo, data: attention.get(combo, 0.0),
        attention_weight=weight,
    )


def test_without_attention_highest_variance_wins():
    assert _select({'c': 1.0}, weight=0.0) == ['a']


def test_stuck_pair_outranks_higher_variance():
    # percentiles: a 1.0, b 0.67, c 0.33; c's last run failed (+1 x 0.5 -> 0.83)
    assert _select({'c': 1.0}, weight=0.5) == ['a']
    assert _select({'c': 1.0}, weight=1.0) == ['c']
    assert _select({'b': 1.0}, weight=0.5) == ['b']
