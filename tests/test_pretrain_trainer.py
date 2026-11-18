"""Pretraining's epoch loop: per-head losses, the validation split, per-head model
selection, early stopping, and 515-D phase nodes."""
import pytest
import torch

from mllf.cb.policy import PHASE_NODE_WIDTH, UnimolPolicy
from mllf.pretrain import trainer as tr
from mllf.pretrain.config import PretrainOptions, UnimolSettings
from mllf.pretrain.features import RunInputs


def _policy():
    torch.manual_seed(0)
    return UnimolPolicy(unimol_dim=512, use_dual_embeddings=True, env_input='phase',
                        site_size_input=True)


def _graph(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, PHASE_NODE_WIDTH, generator=g)
    ei = torch.tensor([[0, 1], [1, 0]])
    targets = [[1.0, 5.0, 0.5, 0.2], [-1.0, 5.0, -0.5, -0.2]]
    return x, ei, targets


def _runs(n, combos=None):
    return [{'run_dir': f'/data/ds/{(combos or {}).get(i, f"comb_{i:03d}")}/run_{i:03d}',
             'sim_results': {}, 'metadata': {}} for i in range(n)]


def test_validation_split_holds_out_whole_combos():
    runs = _runs(40, {i: f'comb_{i // 4:03d}' for i in range(40)})     # 10 combos x 4 runs
    held = tr.split_validation(runs, 0.2)
    combos = {runs[i]['run_dir'].rsplit('/', 2)[1] for i in held}
    assert len(combos) == 2 and len(held) == 8
    assert tr.split_validation(runs, 0.2) == held                        # seeded
    assert tr.split_validation(runs, 0.0) == []


def test_head_losses_sum_to_the_run_loss():
    torch.manual_seed(1)
    w, logp = torch.rand(6, 4), torch.randn(6, 4)
    active = torch.rand(6, 4) > 0.3
    mask = active & (torch.rand(6, 4) > 0.2)
    for hygiene in (True, False):
        heads = tr._awr_head_losses(w, logp, mask, active, w[mask].sum(), hygiene)
        assert heads.shape == (4,)
        assert torch.isclose(heads.sum(), tr._awr_run_loss(w, logp, mask, active,
                                                           w[mask].sum(), hygiene))


def _epoch(policy, graphs, groups, train, lr=1e-3):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    return tr.pretrain_epoch(policy, opt, _runs(len(graphs)), {}, torch.device('cpu'),
                             graph_cache=graphs, groups=groups,
                             pairwise_weights_cache=[[[1.0] * 4] * 2] * len(graphs),
                             data_hygiene=True, train=train)


def test_tightly_fit_runs_still_count_and_validation_takes_no_step():
    pol = _policy()
    graphs = [_graph(i) for i in range(3)]
    groups = {f'run_{i}': [i] for i in range(3)}
    # a tight std makes the NLL of a well-fit run negative: it must still count
    with torch.no_grad():
        for mlp in pol.edge_mlp.mlps:
            mlp.readout.bias[1] = -5.0
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    val = _epoch(pol, graphs, groups, train=False)
    assert all(torch.equal(before[k], v) for k, v in pol.state_dict().items())
    assert val['num_updates'] == 3
    assert abs(sum(val['head_loss']) - val['loss']) < 1e-4
    stats = _epoch(pol, graphs, groups, train=True)
    assert stats['num_updates'] == 3 and len(stats['head_loss']) == 4
    assert any(not torch.equal(before[k], v) for k, v in pol.state_dict().items())


# the scripted epochs take no optimizer step, so the scheduler warns about the order
@pytest.mark.filterwarnings('ignore:Detected call of:UserWarning')
def test_each_head_keeps_its_best_epoch_and_training_stops_when_none_improve(
        tmp_path, monkeypatch):
    # scripted validation loss per head: linear best at epoch 2, quadratic at 5,
    # skew at 1, end at 3; nothing improves after epoch 5
    script = {1: [5, 5, 1, 5], 2: [1, 5, 2, 5], 3: [2, 4, 2, 1], 4: [3, 3, 3, 2],
              5: [3, 2, 3, 2]}
    state = {'epoch': 0}

    def fake_epoch(policy, optimizer, runs, reward_config, device, groups=None, train=True,
                   **kw):
        if train:
            state['epoch'] += 1
            with torch.no_grad():   # weights record the epoch they were trained in
                for p in policy.parameters():
                    p.fill_(float(state['epoch']))
            return {'loss': 1.0, 'num_runs': len(groups), 'num_updates': len(groups),
                    'head_loss': [1.0] * 4}
        losses = [float(v) for v in script.get(state['epoch'], [9, 9, 9, 9])]
        return {'loss': sum(losses), 'num_runs': len(groups), 'num_updates': len(groups),
                'head_loss': losses}

    monkeypatch.setattr(tr, 'pretrain_epoch', fake_epoch)
    pol = _policy()
    runs = _runs(10)
    inputs = RunInputs(graph_cache=[_graph(i) for i in range(10)], n_ok=10,
                       nsubs_by_system={}, consensus=None)
    opts = PretrainOptions(epochs=100, patience=3, val_fraction=0.2)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    progress = tr._Progress()
    tr._train_epochs(pol, opt, sched, runs, inputs, None, None, False,
                     UnimolSettings.from_config({}), opts, {}, tmp_path, torch.device('cpu'),
                     progress, val_idx=tr.split_validation(runs, 0.2))
    assert progress.best_head_epoch == [2, 5, 1, 3]
    assert progress.last_epoch == 8                      # epochs 6-8 without improvement
    best = torch.load(tmp_path / 'best_policy.pt', weights_only=False)
    assert best['head_epoch'] == [2, 5, 1, 3] and best['selected_on'] == 'validation'
    for d, ep in enumerate([2, 5, 1, 3]):
        head = [v for k, v in best['policy_state'].items() if k.startswith(f'edge_mlp.mlps.{d}.')]
        assert head and all(torch.all(v == ep) for v in head)
    # resuming restores each head's best epoch and the epochs without improvement
    ck = tmp_path / 'checkpoint_epoch_008.pt'
    resumed = tr._resume(ck, _policy(), torch.optim.Adam(_policy().parameters()),
                         torch.optim.lr_scheduler.CosineAnnealingLR(
                             torch.optim.Adam(_policy().parameters()), T_max=100),
                         tmp_path, 'cpu')
    assert resumed.best_head_epoch == [2, 5, 1, 3] and resumed.epochs_without_improvement == 3
