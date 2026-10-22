"""Tests for mllf.cb.epoch_log: the concise epoch_summary.log block."""
import math

from mllf.cb.epoch_log import format_epoch_block, parse_epoch_block

ROWS = [
    {'system': '12benz_solvent', 'stage': 'pairs_groups', 'stage_epoch': 13, 'stage_max': 20,
     'progress': (0, 3, 'groups'), 'ref': (4, 9), 'retired': 1, 'ns': 24.81},
    {'system': '1_FAAH_vac', 'stage': 'pairs_groups', 'stage_epoch': 7, 'stage_max': 20,
     'progress': (7, 7, 'groups'), 'ref': (30, 31), 'retired': 0, 'ns': 24.2},
    {'system': 'x_vac', 'stage': 'warmup', 'stage_epoch': 1, 'stage_max': 2,
     'progress': None, 'ref': None, 'retired': None, 'ns': 0.0},
]


def _block(**kw):
    args = dict(
        epoch=12, runs={'groups': 6, 'reference': 245, 'reversed': 30, 'other': 81},
        failed=['12benz_solvent/group01_site1_subs_1_2_3_4_5'],
        pair_short={'both': 43, 'total': 327}, pair_full={'both': 9, 'total': 101},
        groups_run=6, groups_passed=['12fuzed_vac/group05'],
        groups_reopened=[('14benz_vac/group01', 12)],
        validation={'both': 7, 'total': 48}, gate='model accepted, new best saved',
        regression={'online_before': 2.943, 'online_after': 2.694, 'replay_before': 4.888,
                     'replay_after': 4.457},
        hindsight={'median_step': {'linear': 2.31, 'quadratic': 4.33, 'skew': 0.94, 'end': None},
                   'b_grown': 53, 'b_bisected': 88, 'b_unlocked': 4, 'c_up': 57, 'c_down': 26,
                   'c_unlocked': 2, 'pinned': 13, 'stranded': 34, 'held_off': 33},
        systems=ROWS,
        stage_changes=[{'system': '1_FAAH_vac', 'stage': 'pairs_groups', 'next': None,
                        'reason': 'criterion'}],
        ns_train=503.2, ns_validation=112.0)
    args.update(kw)
    return format_epoch_block(**args)


def test_block_reads_as_expected():
    text = _block()
    lines = text.splitlines()
    assert lines[0] == ('Epoch 12 Summary: 332 runs = 6 groups + 245 reference pairs (30 reversed)'
                        ' + 81 other pairs')
    assert '  Pair runs good: short 43/327 (13.1%), full-length 9/101 (8.9%)' in lines
    assert '  Groups: 6 run, 1 passed: 12fuzed_vac/group05' in lines
    assert '  Stage changes: 1_FAAH_vac pairs_groups -> done (converged)' in lines
    assert lines[-1] == '  Simulated ns: training 503.2, validation 112.0 (not counted)'


def test_block_round_trip():
    nb = parse_epoch_block(_block().split('\n', 1)[1])
    assert nb['short'] == (43, 327) and nb['full'] == (9, 101)
    assert nb['validation'] == (7, 48) and nb['gate'] == 'best'
    assert nb['regression'] == (2.943, 2.694, 4.888, 4.457)
    assert nb['median_step']['linear'] == 2.31 and math.isnan(nb['median_step']['end'])
    assert [r['system'] for r in nb['systems']] == ['12benz_solvent', '1_FAAH_vac', 'x_vac']
    r = nb['systems'][0]
    assert r['progress'] == (0, 3, 'groups passed') and r['ref'] == (4, 9) and r['retired'] == 1
    assert r['stage_epoch'] == 13 and r['stage_max'] == 20 and r['ns'] == 24.8
    assert nb['systems'][2]['progress'] is None and nb['systems'][2]['retired'] is None
    assert nb['changes'] == {'1_FAAH_vac': ('done', 'criterion')}


def test_empty_parts_are_left_out():
    text = _block(failed=[], groups_run=0, groups_passed=[], groups_reopened=[], validation=None,
                  stage_changes=[], pair_full=None)
    assert 'Failed' not in text and 'Groups' not in text and 'Validation' not in text
    assert 'Stage changes' not in text and 'full-length' not in text
