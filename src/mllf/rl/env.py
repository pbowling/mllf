"""A minimal custom Gym environment for testing A2C training.

This environment uses a simple discrete action space and a small observation
vector. It's deterministic and intended only as a scaffold to wire up
Stable Baselines3 training scripts.
"""
from typing import Tuple, Optional, Callable, Any

import numpy as np

try:
    import gym
    from gym import spaces
except Exception:  # pragma: no cover - allow environments with gymnasium
    import gymnasium as gym
    from gymnasium import spaces

from .graph import Graph
from mllf.file_handling.read_output import parse_single_population, parse_transitions_and_rates
from mllf.file_handling.write_bias_coeff import write_bias_inp_from_graph


class GraphEnv(gym.Env):
    """Environment where the observation is the flattened edge coefficients of a graph.

    Observation: Box with length = n_edges * 4 (linear, quadratic, skew, end)
    Action: Box with same length representing additive updates to the edge coefficients.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, num_nodes: int = 3, max_steps: int = 50, coeff_limit: float = 10.0,
                 output_text: Optional[str] = None, target_lambda: float = 0.99, simulation_runner: Optional[Callable[[Any], Any]] = None,
                 out_inp_template: Optional[str] = None):
        super().__init__()
        self.max_steps = max_steps
        self.num_nodes = num_nodes
        self.graph = Graph(num_nodes)
        self._step_count = 0
        n_edges = num_nodes * (num_nodes - 1) // 2
        self.obs_dim = n_edges * 4
        self.coeff_limit = coeff_limit
        # optional external metrics loaded from an output text using read_output parser
        self._external_metrics = False
        self._external_pops = None
        self._external_rates = None
        self.target_lambda = target_lambda
        # optional per-step simulation runner: callable(current_coeff_vector) -> output_text | {output_text, new_coeffs}
        self.simulation_runner = simulation_runner
        # optional template for writing .inp files per step; may contain {step}
        # e.g. 'examples/rl/variables{step}.inp'
        self.out_inp_template = out_inp_template
        if output_text is not None:
            try:
                self.update_metrics_from_output_text(output_text)
                self._external_metrics = True
            except Exception:
                # if parsing fails, fallback to internal proxies
                self._external_metrics = False
        low = -coeff_limit * np.ones(self.obs_dim, dtype=np.float32)
        high = coeff_limit * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Actions are continuous additive updates to the coefficients
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # placeholders to track previous means for reward deltas
        self._prev_mean_population = None
        self._prev_mean_rate = None

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        # reset graph to zero coefficients
        self.graph = Graph(self.num_nodes)
        # initialize previous metrics to the current values so first step has zero delta
        if self._external_metrics and (self._external_pops is not None):
            pops = self._external_pops
            rates = self._external_rates if self._external_rates is not None else np.zeros_like(pops)
        else:
            pops = self._compute_site_populations()
            rates = self._compute_transition_rates()
        self._prev_mean_population = float(np.mean(pops))
        self._prev_mean_rate = float(np.mean(rates))
        return self.graph.as_vector()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=float).flatten()
        if action.size != self.obs_dim:
            raise ValueError(f"Action size {action.size} does not match expected {self.obs_dim}")
        # apply additive update scaled by a small factor
        current = self.graph.as_vector()
        new = np.clip(current + 0.1 * action, -self.coeff_limit, self.coeff_limit)
        self.graph.from_vector(new)
        self._step_count += 1
        done = self._step_count >= self.max_steps
        # Optionally run an external simulation that produces an output text and/or new coefficients
        if self.simulation_runner is not None:
            try:
                # optionally write .inp file before running the simulation
                if self.out_inp_template is not None:
                    try:
                        inp_path = self.out_inp_template.format(step=self._step_count)
                        write_bias_inp_from_graph(self.graph, inp_path)
                    except Exception:
                        pass
                runner_out = self.simulation_runner(self.graph.as_vector())
                if isinstance(runner_out, dict):
                    out_text = runner_out.get("output_text")
                    new_coeffs = runner_out.get("new_coeffs")
                else:
                    out_text = runner_out
                    new_coeffs = None
                if isinstance(out_text, str) and out_text:
                    # update external metrics from the output text
                    try:
                        self.update_metrics_from_output_text(out_text)
                    except Exception:
                        # parsing failure: keep previous or proxy
                        pass
                if new_coeffs is not None:
                    # accept new coefficient vector for the next state
                    try:
                        self.graph.from_vector(new_coeffs)
                    except Exception:
                        pass
            except Exception:
                # runner error: ignore and continue with proxies
                pass

        # compute site-level metrics (prefer external parsed metrics if available)
        if self._external_metrics and (self._external_pops is not None):
            pops = self._external_pops
            rates = self._external_rates if self._external_rates is not None else np.zeros_like(pops)
        else:
            pops = self._compute_site_populations()
            rates = self._compute_transition_rates()

        mean_pop = float(np.mean(pops))
        mean_rate = float(np.mean(rates))

        # reward on increases in mean population and mean transition rate
        # use small weights to keep magnitudes reasonable
        pop_weight = 1.0
        rate_weight = 1.0
        prev_pop = self._prev_mean_population if self._prev_mean_population is not None else mean_pop
        prev_rate = self._prev_mean_rate if self._prev_mean_rate is not None else mean_rate

        reward = pop_weight * (mean_pop - prev_pop) + rate_weight * (mean_rate - prev_rate)

        # update previous metrics
        self._prev_mean_population = mean_pop
        self._prev_mean_rate = mean_rate

        # termination:
        if self._external_metrics and (self._external_pops is not None):
            # external parsed single populations are integer counts -> require >= 1
            if np.all(pops >= 1.0):
                done = True
        else:
            # fallback proxy: when all site populations are non-zero (above tiny epsilon)
            eps = 1e-9
            if np.all(np.abs(pops) > eps):
                done = True

        info = {"site_populations": pops, "site_rates": rates}
        return new, float(reward), bool(done), info

    def update_metrics_from_output_text(self, text: str):
        """Parse an output text and populate `_external_pops` and `_external_rates` arrays.

        Uses `parse_single_population` and `parse_transitions_and_rates` from the
        `mllf.file_handling.read_output` module. Extracts values corresponding to
        `self.target_lambda` and maps site ids to indices 0..num_nodes-1.
        """
        pops_map = parse_single_population(text)
        transitions_map, rates_map = parse_transitions_and_rates(text)

        # determine site ids and map to indices
        # collect site ids from pops_map (block -> {.., site: s}) and from rates_map keys
        site_ids = set()
        for blk, val in pops_map.items():
            site = val.get("site")
            if site is not None:
                site_ids.add(int(site))
        for s in transitions_map.keys():
            site_ids.add(int(s))
        for s in rates_map.keys():
            site_ids.add(int(s))

        if not site_ids:
            raise ValueError("No site information found in output text")

        max_site = max(site_ids)
        if max_site > self.num_nodes:
            # if parsed output has more sites than env nodes, adjust the mapping up to num_nodes
            # prefer truncation rather than error
            pass

        # build arrays length self.num_nodes, default 0
        pops_arr = np.zeros(self.num_nodes, dtype=float)
        rates_arr = np.zeros(self.num_nodes, dtype=float)

        # Aggregate populations per site: average block counts for the chosen lambda
        # Find the closest lambda key available in the parsed dict for each block
        for blk, val in pops_map.items():
            site = val.get("site")
            if site is None:
                continue
            site_idx = int(site) - 1
            if site_idx < 0 or site_idx >= self.num_nodes:
                continue
            counts = val.get("counts", {})
            # choose the lambda key nearest to target_lambda
            if not counts:
                continue
            # find lambda key closest to target_lambda
            lambda_keys = list(counts.keys())
            chosen = min(lambda_keys, key=lambda x: abs(x - self.target_lambda))
            pops_arr[site_idx] += float(counts[chosen])

        # convert sums to averages by counting blocks per site
        blocks_per_site = [0] * self.num_nodes
        for blk, val in pops_map.items():
            site = val.get("site")
            if site is None:
                continue
            site_idx = int(site) - 1
            if 0 <= site_idx < self.num_nodes:
                blocks_per_site[site_idx] += 1
        for i in range(self.num_nodes):
            if blocks_per_site[i] > 0:
                pops_arr[i] = pops_arr[i] / blocks_per_site[i]

        # Extract rates per site using chosen lambda
        for s, rates_dict in rates_map.items():
            site_idx = int(s) - 1
            if 0 <= site_idx < self.num_nodes:
                # find closest lambda key
                lambda_keys = list(rates_dict.keys())
                if not lambda_keys:
                    continue
                chosen = min(lambda_keys, key=lambda x: abs(x - self.target_lambda))
                rates_arr[site_idx] = float(rates_dict[chosen])

        self._external_pops = pops_arr
        self._external_rates = rates_arr
        self._external_metrics = True

    def _compute_site_populations(self) -> np.ndarray:
        """Compute a per-site population proxy from incident edge coefficients.

        For each node i we sum a linear combination of incident edge coefficients
        and apply a softplus to produce a non-negative population-like value.
        """
        n = self.num_nodes
        pops = np.zeros(n, dtype=float)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                e = self.graph.get_edge(a, b)
                # combine coefficients: linear contributes most, skew gives asymmetry
                s += (e.linear + 0.1 * e.skew - 0.05 * e.quadratic + 0.01 * e.end)
            # softplus to ensure non-negative and smooth
            pops[i] = np.log1p(np.exp(s))
        return pops

    def _compute_transition_rates(self) -> np.ndarray:
        """Compute a per-site transition-rate proxy from incident edge coefficients.

        Rate proxy uses magnitude of skew and linear components across incident edges.
        """
        n = self.num_nodes
        rates = np.zeros(n, dtype=float)
        for i in range(n):
            vals = []
            for j in range(n):
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                e = self.graph.get_edge(a, b)
                vals.append(abs(e.skew) + 0.5 * abs(e.linear))
            rates[i] = float(np.mean(vals)) if vals else 0.0
        return rates

    def render(self, mode="human"):
        print(f"Step {self._step_count}: graph_vector={self.graph.as_vector()}\n")

    def close(self):
        return None


# Keep the SimpleCustomEnv name for compatibility but make it a thin wrapper
class SimpleCustomEnv(GraphEnv):
    def __init__(self, max_steps: int = 50):
        # default to 3 nodes for the compatibility env (obs_dim = 3 edges * 4 = 12)
        super().__init__(num_nodes=3, max_steps=max_steps)

