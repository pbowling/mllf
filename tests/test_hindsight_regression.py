"""mllf.cb.hindsight_regression: full-network update toward hindsight targets."""
import math

import torch

from mllf.cb.hindsight_regression import online_item, regression_config, regression_step
from mllf.cb.policy import UnimolPolicy


def _policy():
    torch.manual_seed(0)
    pol = UnimolPolicy(unimol_dim=512, mlp_hidden=64, mlp_out_dim=4, use_dual_embeddings=True,
                       use_bayesian_heads=True)
    for d in range(4):
        h = pol.edge_mlp.mlps[d].bayesian_head
        h.seed_mean(torch.randn(h.feat_dim) * 0.01, 0.0, prior_precision=50.0)
    return pol


def _graph(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 1024, generator=g)
    ei = torch.tensor([[0, 1], [1, 0]])
    return x, ei


def test_online_item_weights_are_inverse_variance():
    x, ei = _graph(0)
    t = torch.zeros(2, 4)
    v = torch.tensor([[0.25, math.inf, 1.0, math.inf]] * 2, dtype=torch.float64)
    item = online_item(x, ei, t, v)
    assert torch.allclose(item[3][0], torch.tensor([4.0, 0.0, 1.0, 0.0]))
    assert online_item(x, ei, t, torch.full((2, 4), math.inf, dtype=torch.float64)) is None


def test_regression_moves_full_network_toward_targets_and_reseeds_heads():
    pol = _policy()
    trunk0 = [p.detach().clone() for p in pol.edge_mlp.mlps[0].trunk.parameters()]
    mu0 = pol.edge_mlp.mlps[0].bayesian_head.mu.clone()
    online = []
    for s in range(6):
        x, ei = _graph(s)
        t = torch.tensor([[5.0, 20.0, 1.0, 2.0], [-5.0, 20.0, 1.0, 2.0]])
        online.append(online_item(x, ei, t, torch.full((2, 4), 0.25, dtype=torch.float64)))
    cfg = regression_config({'regression': {'steps': 60, 'lr': 1e-3, 'replay_weight': 0.0}})
    st = regression_step(pol, online, [], cfg, scale=torch.ones(4))
    assert st['online_after'] < 0.5 * st['online_before']
    # trunk changed (not only the last layer) and the head was re-seeded
    assert any(not torch.allclose(a, b) for a, b in
               zip(trunk0, pol.edge_mlp.mlps[0].trunk.parameters()))
    h = pol.edge_mlp.mlps[0].bayesian_head
    assert not torch.allclose(h.mu, mu0)
    assert torch.allclose(h.Lambda, 50.0 * torch.eye(h.Lambda.shape[0]))
    assert torch.allclose(h.Lambda_mu, h.Lambda @ h.mu)


def test_replay_term_limits_drift_on_replay_data():
    pol_a, pol_b = _policy(), _policy()
    online = [online_item(*_graph(s), torch.tensor([[30.0, 60.0, 3.0, 5.0],
                                                    [-30.0, 60.0, 3.0, 5.0]]),
                          torch.full((2, 4), 0.25, dtype=torch.float64)) for s in range(4)]
    replay = []
    with torch.no_grad():
        for s in range(10, 18):
            x, ei = _graph(s)
            mean, _, _ = pol_a.get_actions_thompson(x, ei, None, deterministic=True)
            replay.append((x, ei, mean.float(), torch.ones(2, 4)))   # replay = current predictions
    base = {'steps': 80, 'lr': 1e-3}
    st_no = regression_step(pol_a, online, replay,
                            regression_config({'regression': {**base, 'replay_weight': 0.0}}),
                            scale=torch.ones(4))
    st_rp = regression_step(pol_b, online, replay,
                            regression_config({'regression': {**base, 'replay_weight': 0.7}}),
                            scale=torch.ones(4))
    assert st_rp['replay_after'] < st_no['replay_after']


def test_no_online_data_is_a_noop():
    pol = _policy()
    mu0 = pol.edge_mlp.mlps[1].bayesian_head.mu.clone()
    assert regression_step(pol, [], [], regression_config({}))['steps'] == 0
    assert torch.equal(pol.edge_mlp.mlps[1].bayesian_head.mu, mu0)


def test_precision_and_sampling_spread_survive_checkpoint_load(tmp_path):
    # The trainer loads the pretrained model into a policy built with the
    # config's prior_precision (1.0); the calibrated precision lives only in
    # the saved Lambda. Regression must keep it, not reset it.
    src = _policy()                                      # heads seeded with precision 50
    path = tmp_path / 'ckpt.pt'
    torch.save({'policy_state': src.state_dict()}, path)
    pol = UnimolPolicy(unimol_dim=512, mlp_hidden=64, mlp_out_dim=4, use_dual_embeddings=True,
                       use_bayesian_heads=True, bayesian_prior_precision=1.0)
    pol.load_state_dict(torch.load(path, weights_only=False)['policy_state'], strict=False)
    x, ei = _graph(3)

    def spread(p):
        torch.manual_seed(1)
        with torch.no_grad():
            s = torch.stack([p.get_actions_thompson(x, ei, None, exploration_scale=0.4)[0]
                             for _ in range(200)])
        return s.std(0)[0]

    before = spread(pol)
    lam0 = pol.edge_mlp.mlps[0].bayesian_head.Lambda.clone()
    online = [online_item(*_graph(s), torch.tensor([[5.0, 20.0, 1.0, 2.0], [-5.0, 20.0, 1.0, 2.0]]),
                          torch.full((2, 4), 0.25, dtype=torch.float64)) for s in range(4)]
    regression_step(pol, online, [], regression_config({'regression': {'steps': 20, 'lr': 1e-3,
                                                                        'replay_weight': 0.0}}))
    h = pol.edge_mlp.mlps[0].bayesian_head
    assert torch.allclose(h.Lambda, lam0)
    assert torch.allclose(h.Lambda_mu, h.Lambda @ h.mu)
    after = spread(pol)
    assert (after < 3.0 * before + 1e-3).all()          # no blow-up of Thompson sampling
