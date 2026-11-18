"""Free energies of end states from MSLD production runs.

A frame's state at a site is the substituent with lambda >= the cutoff
(0.99 by default), or 0 when no substituent is that close to its endpoint.
A combination is one substituent per site; its free energy is that of the
unbiased ensemble relative to a reference combination.

Bias energy. The simulation adds ``-E(lambda)`` with

    E = sum_i b_i l_i + sum_{i<j} c_ij l_i l_j
        + sum_{i!=j} x_ij (1 - exp(-5.56 l_i)) l_j
        + sum_{i!=j} s_ij l_i l_j / (l_i + 0.017)

(``msld_flat.py`` writes these as ``ldin`` and ``ldbv`` terms), so a
substituent with larger b is favoured. At an end state only the linear term
and the terms between substituents at different sites remain, and the
free energy of a combination sigma follows from its biased population p:

    G(sigma) = E(sigma) - kT ln p(sigma)

Estimators.

* WHAM: the replicas' frames are pooled. When the replicas ran with
  different biases, each bias set's normalization f_k comes from binless
  WHAM over every frame, and
  ``G = -kT ln sum_k n_k(sigma) + kT ln sum_k N_k exp(f_k + E_k(sigma)/kT)``
  (n_k: frames in sigma, N_k: all frames of bias set k). With one bias set
  this is the pooled histogram. Only sampled combinations get a value.
* Potts: a Potts model (a field h for each state of each site, a coupling J
  for each pair of states at two sites, state 0 included) fitted to the
  frames' states by pseudo-likelihood maximization gives
  ``G = E(sigma) - kT (sum_i h_i + sum_{i<j} J_ij)``. Its value combines the
  sampling of every combination sharing a substituent or a pair of
  substituents, so it is less noisy than the histogram for many sites, and
  it estimates combinations seen rarely. A combination is left out when one
  of its substituents, or one of its pairs between sites, was never sampled.

Uncertainty: the standard deviation over bootstrap samples of the replicas.
"""
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


KB = 0.0019872041            # kcal/mol/K
X_EXP = 5.56                 # 1 / 0.18
S_ALPHA = 0.017


# Bias energy

def bias_terms(bias: Dict, nsubs: Sequence[int]) -> Dict[str, np.ndarray]:
    """b, c, x and s as the simulation applies them: c above the diagonal
    only (``msld_flat.py`` writes ``c[i, j]`` for ``i < j``), x and s without
    their diagonal. ``bias``: the variables file's b, c, x, s."""
    nb = int(sum(nsubs))
    b = np.asarray(bias.get('b', np.zeros(nb)), float).reshape(-1)
    out = {'b': b}
    for key in ('c', 'x', 's'):
        m = np.asarray(bias.get(key, np.zeros((nb, nb))), float).reshape(nb, nb).copy()
        if key == 'c':
            m = np.triu(m, 1)
        else:
            np.fill_diagonal(m, 0.0)
        out[key] = m
    if b.size != nb:
        raise ValueError(f"b has {b.size} entries for {nb} substituents")
    return out


def bias_energy(L: np.ndarray, terms: Dict[str, np.ndarray]) -> np.ndarray:
    """E of each frame (rows of L, kcal/mol); the simulation applied -E."""
    L = np.asarray(L, float)
    e = L @ terms['b']
    e += np.einsum('ni,ij,nj->n', L, terms['c'], L)
    e += np.einsum('ni,ij,nj->n', 1.0 - np.exp(-X_EXP * L), terms['x'], L)
    e += np.einsum('ni,ij,nj->n', L / (L + S_ALPHA), terms['s'], L)
    return e


def same_bias(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> bool:
    return all(np.allclose(a[k], b[k], atol=1e-6) for k in ('b', 'c', 'x', 's'))


# States and combinations

def offsets(nsubs: Sequence[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(nsubs)[:-1]]).astype(int)


def frame_states(L: np.ndarray, nsubs: Sequence[int], cutoff: float = 0.99) -> np.ndarray:
    """(frames, sites) int: 1-based substituent at or above the cutoff, else 0."""
    out = np.zeros((len(L), len(nsubs)), np.int16)
    for i, (o, n) in enumerate(zip(offsets(nsubs), nsubs)):
        block = L[:, o:o + n]
        top = block.argmax(axis=1)
        hit = block[np.arange(len(L)), top] >= cutoff
        out[hit, i] = top[hit] + 1
    return out


def combinations(nsubs: Sequence[int]) -> np.ndarray:
    """Every combination, (n, sites), 1-based, the last site varying fastest."""
    return np.array(list(product(*[range(1, n + 1) for n in nsubs])), dtype=np.int16).reshape(
        -1, len(nsubs))


def combo_keys(states: np.ndarray, nsubs: Sequence[int]) -> np.ndarray:
    """One integer per row of states (0 = gap included), mixed radix n_i + 1."""
    key = np.zeros(len(states), np.int64)
    for i, n in enumerate(nsubs):
        key = key * (n + 1) + states[:, i]
    return key


def endpoint_energy(combos: np.ndarray, terms: Dict[str, np.ndarray],
                    nsubs: Sequence[int]) -> np.ndarray:
    """E of each combination at its end state (one-hot lambdas)."""
    L = np.zeros((len(combos), int(sum(nsubs))))
    o = offsets(nsubs)
    for i in range(len(nsubs)):
        L[np.arange(len(combos)), o[i] + combos[:, i] - 1] = 1.0
    return bias_energy(L, terms)


# WHAM

def _lse(a: np.ndarray, axis=None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def wham_offsets(energies: np.ndarray, n_frames: np.ndarray, kT: float,
                 tol: float = 1e-10, max_iter: int = 20000) -> np.ndarray:
    """Binless WHAM: the dimensionless normalization f_k of each bias set.

    energies: (K, N) E_k of every pooled frame under each bias set k;
    n_frames: (K,) frames sampled under each. f[0] = 0."""
    K = len(n_frames)
    f = np.zeros(K)
    if K == 1:
        return f
    u = -energies / kT                            # reduced applied bias
    ln_n = np.log(np.asarray(n_frames, float))
    for _ in range(max_iter):
        ln_denom = _lse(ln_n[:, None] + f[:, None] - u, axis=0)
        f_new = -_lse(-u - ln_denom[None, :], axis=1)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            return f_new
        f = f_new
    return f


def wham_free_energy(counts: np.ndarray, n_frames: np.ndarray, e_end: np.ndarray,
                     f: np.ndarray, kT: float) -> np.ndarray:
    """G of each combination. counts: (K, C) frames in each combination per
    bias set; e_end: (K, C) its end-state E under each; NaN where unsampled."""
    total = counts.sum(axis=0)
    ln_denom = _lse(np.log(n_frames)[:, None] + f[:, None] + e_end / kT, axis=0)
    with np.errstate(divide='ignore'):
        g = -kT * np.log(total) + kT * ln_denom
    g[total == 0] = np.nan
    return g


# Potts model (pseudo-likelihood)

class Potts:
    """Fields h[i] (n_i + 1 states, state 0 = no substituent at the cutoff)
    and couplings J[(i, j)] (i < j) in units of kT: log P(sigma) = sum_i
    h_i(sigma_i) + sum_{i<j} J_ij(sigma_i, sigma_j) + const."""

    def __init__(self, nsubs: Sequence[int]):
        self.nsubs = list(nsubs)
        self.q = [n + 1 for n in nsubs]
        self.pairs = [(i, j) for i in range(len(nsubs)) for j in range(i + 1, len(nsubs))]
        sizes = self.q + [self.q[i] * self.q[j] for i, j in self.pairs]
        self.slices = np.concatenate([[0], np.cumsum(sizes)])
        self.theta = np.zeros(int(self.slices[-1]))

    def h(self, i: int, theta=None) -> np.ndarray:
        t = self.theta if theta is None else theta
        return t[self.slices[i]:self.slices[i + 1]]

    def J(self, k: int, theta=None) -> np.ndarray:
        t = self.theta if theta is None else theta
        i, j = self.pairs[k]
        n = len(self.q)
        return t[self.slices[n + k]:self.slices[n + k + 1]].reshape(self.q[i], self.q[j])

    def log_weight(self, states: np.ndarray) -> np.ndarray:
        """sum h + sum J of each row of states (0-based state = substituent)."""
        out = np.zeros(len(states))
        for i in range(len(self.q)):
            out += self.h(i)[states[:, i]]
        for k, (i, j) in enumerate(self.pairs):
            out += self.J(k)[states[:, i], states[:, j]]
        return out

    def _objective(self, theta, X, w, reg):
        n = len(self.q)
        loss = 0.0
        grad = np.zeros_like(theta)
        R = []
        for i in range(n):
            heff = np.broadcast_to(self.h(i, theta), X[i].shape).copy()
            for k, (a, b) in enumerate(self.pairs):
                if a == i:
                    heff += X[b] @ self.J(k, theta).T
                elif b == i:
                    heff += X[a] @ self.J(k, theta)
            lse = _lse(heff, axis=1)
            loss -= np.sum(w * (np.sum(X[i] * heff, axis=1) - lse))
            r = w[:, None] * (X[i] - np.exp(heff - lse[:, None]))
            R.append(r)
            grad[self.slices[i]:self.slices[i + 1]] -= r.sum(axis=0)
        for k, (i, j) in enumerate(self.pairs):
            g = R[i].T @ X[j] + X[i].T @ R[j]
            grad[self.slices[n + k]:self.slices[n + k + 1]] -= g.reshape(-1)
        loss += 0.5 * reg * float(theta @ theta)
        grad += reg * theta
        return loss, grad

    def fit(self, states: np.ndarray, weights: np.ndarray, reg: float = 1e-5,
            warm: Optional[np.ndarray] = None) -> 'Potts':
        """Maximize the pseudo-likelihood of the rows of states (unique rows
        with their weights, normalized here) with an L2 penalty ``reg``."""
        from scipy.optimize import minimize
        w = np.asarray(weights, float)
        w = w / w.sum()
        X = [np.eye(q)[states[:, i]] for i, q in enumerate(self.q)]
        x0 = self.theta.copy() if warm is None else warm.copy()
        res = minimize(self._objective, x0, args=(X, w, reg), jac=True, method='L-BFGS-B',
                       options={'maxiter': 5000, 'gtol': 1e-9, 'ftol': 1e-13})
        self.theta = res.x
        return self


# Production analysis of a set of replicas

def _unique(states: np.ndarray, nsubs) -> Dict[int, int]:
    keys, counts = np.unique(combo_keys(states, nsubs), return_counts=True)
    return dict(zip(keys.tolist(), counts.tolist()))


def _decode(keys: Sequence[int], nsubs) -> np.ndarray:
    keys = np.asarray(list(keys), np.int64)
    out = np.zeros((len(keys), len(nsubs)), np.int16)
    for i in range(len(nsubs) - 1, -1, -1):
        out[:, i] = keys % (nsubs[i] + 1)
        keys = keys // (nsubs[i] + 1)
    return out


def analyze_replicas(replicas: List[Dict], nsubs: Sequence[int], temperature: float,
                     cutoff: float = 0.99, n_boot: int = 50, potts: bool = True,
                     reg: float = 1e-5, max_combos: int = 200000, seed: int = 2401) -> Dict:
    """Free energies of a production's replicas.

    replicas: dicts with 'L' (frames, substituent blocks) and 'terms'
    (``bias_terms``). Returns 'combos' (rows, 1-based), 'reference' (row
    index), 'wham' / 'potts' (G relative to the reference, NaN where not
    estimated; potts None when skipped), their bootstrap SDs, 'counts'
    (frames per combination, all replicas), 'replica_g' ((R, C) histogram
    G of each replica alone), 'bias_sets' and 'potts_note'."""
    kT = KB * temperature
    R = len(replicas)
    # bias sets
    sets: List[Dict] = []
    set_of = []
    for rep in replicas:
        k = next((k for k, t in enumerate(sets) if same_bias(t, rep['terms'])), None)
        if k is None:
            sets.append(rep['terms'])
            k = len(sets) - 1
        set_of.append(k)
    set_of = np.array(set_of)
    K = len(sets)
    states = [frame_states(rep['L'], nsubs, cutoff) for rep in replicas]
    per_rep = [_unique(s, nsubs) for s in states]
    n_rep = np.array([len(s) for s in states], float)

    # the combinations to report: all, or the sampled ones when there are too many
    n_all = int(np.prod(nsubs))
    if n_all <= max_combos:
        combos = combinations(nsubs)
    else:
        seen = sorted({k for d in per_rep for k in d})
        combos = _decode(seen, nsubs)
        combos = combos[np.all(combos > 0, axis=1)]
    keys = combo_keys(combos, nsubs)
    e_end = np.array([endpoint_energy(combos, t, nsubs) for t in sets])     # (K, C)
    rep_counts = np.array([[d.get(int(k), 0) for k in keys] for d in per_rep], float)  # (R, C)

    # binless WHAM normalizations over every frame (only with several bias sets)
    E_frames = None
    if K > 1:
        allL = np.concatenate([rep['L'] for rep in replicas])
        E_frames = np.array([bias_energy(allL, t) for t in sets])          # (K, N)
    starts = np.concatenate([[0], np.cumsum(n_rep)]).astype(int)

    def wham(sample: np.ndarray) -> np.ndarray:
        counts = np.zeros((K, len(combos)))
        nk = np.zeros(K)
        for r in sample:
            counts[set_of[r]] += rep_counts[r]
            nk[set_of[r]] += n_rep[r]
        used = nk > 0
        if K > 1 and used.sum() > 1:
            cols = np.concatenate([np.arange(starts[r], starts[r + 1]) for r in sample])
            f = wham_offsets(E_frames[used][:, cols], nk[used], kT)
        else:
            f = np.zeros(int(used.sum()))
        return wham_free_energy(counts[used], nk[used], e_end[used], f, kT)

    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, R, R) for _ in range(n_boot)] if R > 1 else []
    g_wham = wham(np.arange(R))
    with np.errstate(divide='ignore'):
        replica_g = np.array([e_end[set_of[r]] - kT * np.log(rep_counts[r] / n_rep[r])
                              for r in range(R)])
    replica_g[~np.isfinite(replica_g)] = np.nan
    g_wham_bs = np.array([wham(s) for s in boots]) if boots else None

    # Potts
    g_potts = g_potts_bs = None
    note = None
    model = None
    if potts and K > 1:
        note = 'the replicas ran with different biases'
    elif potts:
        def fit(sample, warm=None):
            pooled: Dict[int, int] = {}
            for r in sample:
                for k, c in per_rep[r].items():
                    pooled[k] = pooled.get(k, 0) + c
            st = _decode(pooled.keys(), nsubs)
            m = Potts(nsubs).fit(st, np.array(list(pooled.values()), float), reg=reg, warm=warm)
            return m, st, pooled

        def g_of(m, sample_states, pooled):
            g = e_end[0] - kT * m.log_weight(combos)
            # leave out combinations with an unsampled substituent or pair
            seen_sub = [set(sample_states[:, i].tolist()) for i in range(len(nsubs))]
            ok = np.ones(len(combos), bool)
            for i in range(len(nsubs)):
                ok &= np.isin(combos[:, i], list(seen_sub[i]))
            for i, j in Potts(nsubs).pairs:
                both = sample_states[(sample_states[:, i] > 0) & (sample_states[:, j] > 0)]
                pair_keys = set((both[:, i].astype(int) * 1000 + both[:, j]).tolist())
                ok &= np.isin(combos[:, i].astype(int) * 1000 + combos[:, j], list(pair_keys))
            g[~ok] = np.nan
            return g

        model, st, pooled = fit(np.arange(R))
        g_potts = g_of(model, st, pooled)
        if boots:
            g_potts_bs = np.array([g_of(*fit(s, warm=model.theta)) for s in boots])

    # reference: the first combination with every estimate, else the first with one
    finite = np.isfinite(g_wham) & (np.isfinite(g_potts) if g_potts is not None else True)
    ref = int(np.argmax(finite)) if finite.any() else int(np.argmax(np.isfinite(g_wham)))

    def rel(g):
        return None if g is None else g - g[..., ref:ref + 1] if g.ndim > 1 else g - g[ref]

    def sd(bs):
        if bs is None or len(bs) < 2:
            return np.full(len(combos), np.nan)
        b = rel(bs)
        with np.errstate(invalid='ignore'):
            return np.nanstd(np.where(np.isfinite(b), b, np.nan), axis=0)

    return {'combos': combos, 'reference': ref, 'wham': rel(g_wham), 'wham_sd': sd(g_wham_bs),
            'potts': rel(g_potts), 'potts_sd': sd(g_potts_bs) if g_potts is not None else None,
            'counts': rep_counts.sum(axis=0), 'replica_counts': rep_counts,
            'replica_g': rel(replica_g), 'bias_sets': K, 'set_of': set_of.tolist(),
            'potts_note': note, 'potts_model': model, 'kT': kT}


# Sampling of each replica

def sampling(L: np.ndarray, nsubs: Sequence[int], dt_ps: float, cutoff: float = 0.99) -> Dict:
    """Endpoint fraction and entries per ns of each substituent, transitions
    per ns of each site (changes of the substituent at the cutoff, frames
    between endpoints skipped) and the fraction of frames with every site
    at an endpoint."""
    st = frame_states(L, nsubs, cutoff)
    ns = len(L) * dt_ps / 1000.0
    frac, entries, trans = [], [], []
    for i, n in enumerate(nsubs):
        s = st[:, i]
        frac.append([float(np.mean(s == k + 1)) if len(s) else 0.0 for k in range(n)])
        on = s[s > 0]
        changes = np.flatnonzero(on[1:] != on[:-1]) + 1 if len(on) > 1 else np.array([], int)
        trans.append(len(changes) / ns if ns else 0.0)
        first = [int(on[0])] if len(on) else []
        ent = np.bincount(np.concatenate([first, on[changes]]).astype(int), minlength=n + 1)[1:]
        entries.append((ent / ns if ns else ent * 0.0).tolist())
    return {'ns': ns, 'fraction': frac, 'entries_per_ns': entries, 'transitions_per_ns': trans,
            'physical': float(np.mean(np.all(st > 0, axis=1))) if len(st) else 0.0}


def combo_label(row: Sequence[int], names: Optional[List[List[str]]] = None) -> str:
    if names:
        return ' '.join(names[i][int(s) - 1] for i, s in enumerate(row))
    return ' '.join(str(int(s)) for s in row)


def potts_tables(model: Potts) -> Tuple[List[Tuple], List[Tuple]]:
    """(site, state, h) and (site i, state, site j, state, J) rows, kT units."""
    h = [(i + 1, a, float(v)) for i in range(len(model.q)) for a, v in enumerate(model.h(i))]
    J = [(i + 1, a, j + 1, b, float(model.J(k)[a, b]))
         for k, (i, j) in enumerate(model.pairs)
         for a in range(model.q[i]) for b in range(model.q[j])]
    return h, J
