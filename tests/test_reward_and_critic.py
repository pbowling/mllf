"""Tests for compute_pair_reward and QNetwork (per-pair credit assignment)."""
import math
import pytest
import torch

from mllf.cb.workflow_utils import compute_pair_reward


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
        """A finite DDG entry should yield positive rewards across all dims."""
        edge_index = self._ei([(0, 1)])        # nodes 0,1 → blocks 2,3 → key "2_3"
        ddg_pairs = {"2_3": -2.5}
        populations = [50, 50]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards.shape == (1, 4)
        # All dims are non-negative when the pair was visited
        assert (rewards[0] >= 0.0).all()

    def test_none_ddg_gives_minus_one(self):
        """A None DDG entry should yield -1.0 in all dims."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": None}
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])

    def test_nan_ddg_gives_minus_one(self):
        """A NaN DDG should yield -1.0 in all dims."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": float("nan")}
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])

    def test_inf_ddg_gives_minus_one(self):
        """An Inf DDG should yield -1.0 in all dims."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": float("inf")}
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])

    def test_missing_key_gives_minus_one(self):
        """A key absent from ddg_pairs should yield -1.0 in all dims."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {}   # no entry at all
        populations = [10, 10]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards[0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])

    # ------- minority fraction -------

    def test_equal_populations_gives_max_linear_signal(self):
        """Equal populations → minority_frac = 0.5 → linear dim (0) = 0.5 (maximum)."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": 0.0}
        populations = [100, 100]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        # dim 0 = minority_frac = 0.5 (max possible)
        assert rewards[0, 0].item() == pytest.approx(0.5, abs=1e-5)
        # quadratic dim (1) = (pair_visited=1.0 + trans_quality=0.0) / 2 = 0.5
        assert rewards[0, 1].item() == pytest.approx(0.5, abs=1e-5)

    def test_unequal_populations_gives_intermediate_bonus(self):
        """Unequal populations → minority_frac < 0.5 → linear dim in (0, 0.5)."""
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": 0.0}
        populations = [90, 10]   # minority_frac = 10/100 = 0.1

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        # dim 0 (linear) = minority_frac ∈ (0, 0.5)
        assert 0.0 <= rewards[0, 0].item() < 0.5
        assert rewards[0, 0].item() == pytest.approx(10 / 100, abs=1e-4)

    def test_zero_populations_no_division_error(self):
        """Both populations zero should not raise; signals should be penalized (-1.0).
        
        When both substituents have zero population, this indicates a sampling failure
        (neither substituent was ever explored by REMD), so all dimensions are penalized
        with -1.0.
        """
        edge_index = self._ei([(0, 1)])
        ddg_pairs = {"2_3": 0.0}
        populations = [0, 0]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        # Failure case: both pops are 0 → all dims get -1.0 penalty
        assert rewards[0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])

    # ------- key ordering (lo_hi convention) -------

    def test_key_is_lo_hi_regardless_of_edge_direction(self):
        """Key is always min_block_hi_block; reversed edge should match same dict entry."""
        edge_fwd = self._ei([(0, 1)])   # blocks 2_3
        edge_bwd = self._ei([(1, 0)])   # still blocks 2_3 (lo=2, hi=3)
        ddg_pairs = {"2_3": 1.0}
        populations = [50, 50]

        r_fwd = compute_pair_reward(edge_fwd, ddg_pairs, populations)
        r_bwd = compute_pair_reward(edge_bwd, ddg_pairs, populations)
        assert r_fwd[0].tolist() == pytest.approx(r_bwd[0].tolist(), abs=1e-5)

    # ------- custom block_offset -------

    def test_custom_block_offset(self):
        """block_offset=1 shifts block IDs by 1 instead of default 2."""
        edge_index = self._ei([(0, 1)])   # nodes 0,1 → blocks 1,2 → key "1_2"
        ddg_pairs = {"1_2": 0.0}
        populations = [50, 50]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations, block_offset=1)
        # dim 0 = minority_frac = 0.5 (equal populations → max linear signal)
        assert rewards[0, 0].item() == pytest.approx(0.5, abs=1e-5)

    # ------- multi-edge batch -------

    def test_multiple_edges(self):
        """Multiple edges should each get independent rewards."""
        edge_index = self._ei([(0, 1), (0, 2), (1, 2)])
        # blocks: (2,3), (2,4), (3,4)
        ddg_pairs = {"2_3": 0.0, "2_4": None, "3_4": float("nan")}
        populations = [50, 50, 50]

        rewards = compute_pair_reward(edge_index, ddg_pairs, populations)
        assert rewards.shape == (3, 4)
        # Edge 0 (visited, equal populations): dim 0 = 0.5, dim 1 = 0.5
        assert rewards[0, 0].item() == pytest.approx(0.5, abs=1e-5)
        assert rewards[0, 1].item() == pytest.approx(0.5, abs=1e-5)
        # Edges 1 and 2 (not visited): all dims = -1.0
        assert rewards[1].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])
        assert rewards[2].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])

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

    # ------- quadratic dim: midpoint-barrier signal -------

    def test_barrier_at_target_gives_max_quadratic(self):
        """A barrier exactly at the target scores 1.0 on the quadratic dim."""
        edge_index = self._ei([(0, 1)])
        rewards = compute_pair_reward(edge_index, {"2_3": -1.0}, [50, 50],
                                      barrier_pairs={"2_3": 1.5}, barrier_target=1.5)
        assert rewards[0, 1].item() == pytest.approx(1.0)

    def test_barrier_quality_is_two_sided(self):
        """Over-flattened and too-high barriers both score below the target."""
        edge_index = self._ei([(0, 1)])
        q = {b: compute_pair_reward(edge_index, {"2_3": -1.0}, [50, 50],
                                    barrier_pairs={"2_3": b}, barrier_target=1.5,
                                    barrier_width=1.0)[0, 1].item()
             for b in (-1.0, 1.5, 4.0)}
        assert q[1.5] > q[-1.0] and q[1.5] > q[4.0]
        # symmetric Gaussian: equal distance on either side scores the same
        assert q[-1.0] == pytest.approx(q[4.0])
        assert all(0.5 <= v <= 1.0 for v in q.values())

    def test_barrier_ignores_transition_count(self):
        """With a barrier available, the quadratic dim no longer depends on transitions."""
        edge_index = self._ei([(0, 1)])
        lo = compute_pair_reward(edge_index, {"2_3": -1.0}, [50, 50], total_transitions=2,
                                 barrier_pairs={"2_3": 2.0})
        hi = compute_pair_reward(edge_index, {"2_3": -1.0}, [50, 50], total_transitions=200,
                                 barrier_pairs={"2_3": 2.0})
        assert lo[0, 1].item() == pytest.approx(hi[0, 1].item())

    def test_missing_barrier_falls_back_to_transitions(self):
        """No barrier (absent key, None, or no dict) → legacy transition-based signal."""
        edge_index = self._ei([(0, 1)])
        legacy = compute_pair_reward(edge_index, {"2_3": -1.0}, [50, 50], total_transitions=25)
        for bp in (None, {}, {"2_3": None}):
            r = compute_pair_reward(edge_index, {"2_3": -1.0}, [50, 50], total_transitions=25,
                                    barrier_pairs=bp)
            assert torch.equal(r, legacy)
        assert legacy[0, 1].item() == pytest.approx(0.75)   # (1 + 25/50) / 2

    def test_barrier_does_not_change_other_dims(self):
        """Only the quadratic dim uses the barrier; linear/skew/end are unchanged."""
        edge_index = self._ei([(0, 1)])
        base = compute_pair_reward(edge_index, {"2_3": -1.0}, [30, 70], total_transitions=25)
        with_b = compute_pair_reward(edge_index, {"2_3": -1.0}, [30, 70], total_transitions=25,
                                     barrier_pairs={"2_3": 5.0})
        assert torch.equal(base[0, [0, 2, 3]], with_b[0, [0, 2, 3]])

    def test_unvisited_pair_still_minus_one_with_barrier(self):
        """A barrier value does not rescue a pair that never crossed."""
        edge_index = self._ei([(0, 1)])
        r = compute_pair_reward(edge_index, {"2_3": None}, [50, 50], barrier_pairs={"2_3": 1.5})
        assert r[0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])


    # ------- skew / end dims: profile-asymmetry signals -------

    def test_zero_asymmetry_gives_max_skew_and_end(self):
        edge_index = self._ei([(0, 1)])
        r = compute_pair_reward(edge_index, {"2_3": -1.0}, [50, 50],
                                skew_pairs={"2_3": 0.0}, end_pairs={"2_3": 0.0})
        assert r[0, 2].item() == pytest.approx(1.0)
        assert r[0, 3].item() == pytest.approx(1.0)

    def test_asymmetry_quality_depends_on_magnitude_only(self):
        edge_index = self._ei([(0, 1)])
        q = [compute_pair_reward(edge_index, {"2_3": -1.0}, [50, 50], skew_pairs={"2_3": v},
                                 end_pairs={"2_3": v}, skew_width=0.8, end_width=0.6)[0]
             for v in (-1.0, 1.0, 3.0)]
        assert q[0][2].item() == pytest.approx(q[1][2].item())
        assert q[0][3].item() == pytest.approx(q[1][3].item())
        assert q[2][2].item() < q[1][2].item() < 1.0
        assert all(0.5 <= t[d].item() <= 1.0 for t in q for d in (2, 3))

    def test_missing_asymmetry_falls_back_to_legacy_skew_end(self):
        edge_index = self._ei([(0, 1)])
        legacy = compute_pair_reward(edge_index, {"2_3": -1.0}, [30, 70], total_transitions=25,
                                     fraction_physical=0.2)
        for sp, ep in ((None, None), ({}, {}), ({"2_3": None}, {"2_3": None})):
            r = compute_pair_reward(edge_index, {"2_3": -1.0}, [30, 70], total_transitions=25,
                                    fraction_physical=0.2, skew_pairs=sp, end_pairs=ep)
            assert torch.equal(r, legacy)

    def test_asymmetry_does_not_change_linear_or_quadratic(self):
        edge_index = self._ei([(0, 1)])
        base = compute_pair_reward(edge_index, {"2_3": -1.0}, [30, 70], total_transitions=25,
                                   barrier_pairs={"2_3": 1.0})
        with_a = compute_pair_reward(edge_index, {"2_3": -1.0}, [30, 70], total_transitions=25,
                                     barrier_pairs={"2_3": 1.0},
                                     skew_pairs={"2_3": 2.0}, end_pairs={"2_3": -2.0})
        assert torch.equal(base[0, [0, 1]], with_a[0, [0, 1]])

    def test_unvisited_pair_still_minus_one_with_asymmetry(self):
        edge_index = self._ei([(0, 1)])
        r = compute_pair_reward(edge_index, {"2_3": None}, [50, 50],
                                skew_pairs={"2_3": 0.0}, end_pairs={"2_3": 0.0})
        assert r[0].tolist() == pytest.approx([-1.0, -1.0, -1.0, -1.0])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
