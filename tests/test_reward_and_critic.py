"""Tests for compute_pair_reward and QNetwork (per-pair credit assignment)."""
import math
import pytest
import torch

from mllf.cb.workflow_utils import compute_pair_reward
from mllf.cb.value_net import QNetwork


# ---------------------------------------------------------------------------
# compute_pair_reward
# ---------------------------------------------------------------------------

class TestComputePairReward:
    """Tests for workflow_utils.compute_pair_reward."""

    def _ei(self, pairs):
        """Build [2, E] edge_index from list of (src, dst) pairs."""
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        return torch.stack([src, dst], dim=0)

    # ------- basic reward values -------

    def test_finite_ddg_gives_positive_reward(self):
        """A finite DDG entry should yield reward >= 1.0."""
        edge_index = self._ei([(0, 1)])        # nodes 0,1 → blocks 2,3 → key "2_3"
        ddg_pairs = {"2_3": -2.5}
        populations = [50, 50]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards.shape == (1,)
        assert rewards[0].item() >= 1.0

    def test_none_ddg_gives_minus_one(self):
        """A None DDG entry should yield -1.0."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": None}
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].item() == pytest.approx(-1.0)

    def test_nan_ddg_gives_minus_one(self):
        """A NaN DDG should yield -1.0."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": float("nan")}
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].item() == pytest.approx(-1.0)

    def test_inf_ddg_gives_minus_one(self):
        """An Inf DDG should yield -1.0."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": float("inf")}
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].item() == pytest.approx(-1.0)

    def test_missing_key_gives_minus_one(self):
        """A key absent from ddg_pairs should yield -1.0."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {}   # no entry at all
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].item() == pytest.approx(-1.0)

    # ------- minority fraction -------

    def test_equal_populations_gives_max_bonus(self):
        """Equal populations → minority_fraction = 0.5 → reward = 1.5."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": 0.0}
        populations = [100, 100]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].item() == pytest.approx(1.5, abs=1e-5)

    def test_unequal_populations_gives_intermediate_bonus(self):
        """Unequal populations → minority_fraction < 0.5 → 1.0 <= reward < 1.5."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": 0.0}
        populations = [90, 10]   # minority_frac = 10/100 = 0.1

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert 1.0 <= rewards[0].item() < 1.5
        assert rewards[0].item() == pytest.approx(1.0 + 10 / 100, abs=1e-4)

    def test_zero_populations_no_division_error(self):
        """Both populations zero should not raise; reward should be 1.0 + 0."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": 0.0}
        populations = [0, 0]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].item() == pytest.approx(1.0, abs=1e-4)

    # ------- key ordering (lo_hi convention) -------

    def test_key_is_lo_hi_regardless_of_edge_direction(self):
        """Key is always min_block_hi_block; reversed edge should match same dict entry."""
        edge_fwd = self._ei([(0, 1)])   # blocks 2_3
        edge_bwd = self._ei([(1, 0)])   # still blocks 2_3 (lo=2, hi=3)
        ddg_pairs = {"2_3": 1.0}
        populations = [50, 50]

        r_fwd = compute_pair_reward(edge_fwd, ddg_pairs, populations)
        r_bwd = compute_pair_reward(edge_bwd, ddg_pairs, populations)
        assert r_fwd[0].item() == pytest.approx(r_bwd[0].item(), abs=1e-5)

    # ------- custom block_offset -------

    def test_custom_block_offset(self):
        """block_offset=1 shifts block IDs by 1 instead of default 2."""
        edge_index = self._ei([(0, 1)])   # nodes 0,1 → blocks 1,2 → key "1_2"
        ddg_pairs = {"1_2": 0.0}
        populations = [50, 50]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations, block_offset=1)
        assert rewards[0].item() == pytest.approx(1.5, abs=1e-5)

    # ------- multi-edge batch -------

    def test_multiple_edges(self):
        """Multiple edges should each get independent rewards."""
        edge_index = self._ei([(0, 1), (0, 2), (1, 2)])
        # blocks: (2,3), (2,4), (3,4)
        ddg_pairs = {"2_3": 0.0, "2_4": None, "3_4": float("nan")}
        populations = [50, 50, 50]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards.shape == (3,)
        assert rewards[0].item() == pytest.approx(1.5, abs=1e-5)
        assert rewards[1].item() == pytest.approx(-1.0)
        assert rewards[2].item() == pytest.approx(-1.0)

    def test_output_dtype_and_device(self):
        """Output should be float32 CPU tensor."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": 0.5}
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards.dtype == torch.float32
        assert rewards.device.type == "cpu"

    def test_reward_range(self):
        """All rewards must lie in [-1.0, +1.5]."""
        edge_index = self._ei([(0, 1), (0, 2), (1, 2), (2, 3)])
        ddg_pairs = {"2_3": 1.0, "2_4": None, "3_4": float("inf")}
        populations = [30, 70, 50, 20]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards.min().item() >= -1.0 - 1e-6
        assert rewards.max().item() <= 1.5 + 1e-6


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------

class TestQNetwork:
    """Tests for value_net.QNetwork per-edge Q-value critic."""

    def test_default_hidden_dims(self):
        """Default hidden_dims=[64, 32]; MLP input = in_dim + action_dim."""
        q = QNetwork(in_dim=200, action_dim=4)
        layers = [m for m in q.mlp if isinstance(m, torch.nn.Linear)]
        assert layers[0].in_features == 204   # 200 state + 4 actions
        assert layers[0].out_features == 64
        assert layers[1].in_features == 64
        assert layers[1].out_features == 32
        assert layers[2].in_features == 32
        assert layers[2].out_features == 1

    def test_custom_hidden_dims(self):
        q = QNetwork(in_dim=100, action_dim=4, hidden_dims=[128, 64, 32])
        layers = [m for m in q.mlp if isinstance(m, torch.nn.Linear)]
        assert len(layers) == 4           # 3 hidden + 1 output
        assert layers[0].in_features == 104  # 100 state + 4 actions
        assert layers[-1].out_features == 1

    def test_output_shape(self):
        """forward() returns scalar per edge: [E]."""
        E, D, A = 12, 200, 4
        q = QNetwork(in_dim=D, action_dim=A)
        edge_inputs = torch.randn(E, D)
        actions = torch.randn(E, A)
        out = q(edge_inputs, actions)
        assert out.shape == (E,), f"Expected ({E},), got {out.shape}"

    def test_single_edge(self):
        q = QNetwork(in_dim=50, action_dim=4)
        edge_inputs = torch.randn(1, 50)
        actions = torch.randn(1, 4)
        out = q(edge_inputs, actions)
        assert out.shape == (1,)

    def test_no_nan_in_output(self):
        q = QNetwork(in_dim=64, action_dim=4)
        edge_inputs = torch.randn(8, 64)
        actions = torch.randn(8, 4)
        out = q(edge_inputs, actions)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        """Gradients should flow back through Q-network parameters."""
        E, D, A = 8, 64, 4
        q = QNetwork(in_dim=D, action_dim=A)
        edge_inputs = torch.randn(E, D)
        actions = torch.randn(E, A)
        out = q(edge_inputs, actions)
        loss = out.mean()
        loss.backward()
        for name, param in q.named_parameters():
            assert param.grad is not None, f"Param {name} has no gradient"

    def test_no_gradient_through_detached_q(self):
        """When Q is detached (for advantage), its params get no grad from actor loss."""
        E, D, A = 6, 32, 4
        q = QNetwork(in_dim=D, action_dim=A)
        edge_inputs = torch.randn(E, D)
        actions = torch.randn(E, A)

        q_values = q(edge_inputs, actions).detach()   # detach as in REINFORCE advantage
        assert q_values.grad_fn is None, "Detached q_values should have no grad_fn"

        # Simulate actor loss: logp (which has a grad_fn) weighted by advantage
        # advantage = R - Q.detach() — no graph through Q
        logp = torch.randn(E, requires_grad=True)   # stand-in for policy logp
        advantage = torch.rand(E) - q_values        # advantage is leaf (no grad_fn through Q)
        actor_loss = -(logp * advantage.detach()).sum()
        actor_loss.backward()

        # Q-network parameters should have no grad because they weren't
        # in the computation graph of actor_loss.
        for name, param in q.named_parameters():
            assert param.grad is None, (
                f"Q-network param {name} should have no grad when not in actor graph"
            )

    def test_deterministic_output(self):
        """Same input should produce same output (no dropout in QNetwork)."""
        q = QNetwork(in_dim=32, action_dim=4)
        q.eval()
        x = torch.randn(5, 32)
        a = torch.randn(5, 4)
        with torch.no_grad():
            o1 = q(x, a)
            o2 = q(x, a)
        assert torch.allclose(o1, o2)

    def test_advantage_computation(self):
        """A = R - Q.detach() should be correct shape and finite."""
        E, D, A = 10, 64, 4
        q = QNetwork(in_dim=D, action_dim=A)
        edge_inputs = torch.randn(E, D)
        actions = torch.randn(E, A)
        rewards = torch.rand(E) * 2 - 1   # in [-1, 1]

        with torch.no_grad():
            q_vals = q(edge_inputs, actions)

        advantage = rewards - q_vals
        assert advantage.shape == (E,)
        assert not torch.isnan(advantage).any()
        assert not torch.isinf(advantage).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
