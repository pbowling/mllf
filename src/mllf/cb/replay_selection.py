"""Choose the regression replay set by pair coverage (``replay_selection: coverage``).

The replay anchors the shared model to pretraining while online training
moves it. A random run per combo directory mostly replays runs that did not
sample their pairs well, so with a fixed budget the replay carries little of
what pretraining knows. Here each system instead gets the runs that cover
its pairs best:

1. Every run in the pretraining manifest (ALF groups, manifest combos and
   online runs collected into pretraining alike) has its nodes labelled with
   the system's own (site, substituent) numbering, by matching each node's
   fragment to the system's prep (``_rtf_signature``, as the combo sets do).
2. Each within-site pair a run sampled gets a flatness score
   (``pair_quality``): population balance (min/max, which does not depend on
   run length) times its crossings relative to the run length's success test
   (10 for a full run, 3 for a short one).
3. Each system gets a share of the budget proportional to its number of
   substituents (at least ``min_per_system``), and runs are chosen greedily
   to cover as many pairs as possible with as few graphs as possible. A
   pair's value is min(1, score / ``COVER_QUALITY``): it is fully covered
   once a chosen run samples it at that score, and refining it further is
   worth nothing. A run's gain is the rise in its pairs' values, so a group
   run that samples many pairs well beats a pair run that samples one pair
   slightly better; the sum of scores only breaks ties. (Coverage of this
   kind is monotone submodular, so greedy is within 1 - 1/e of the best
   set.) A system stops below its budget once no run adds coverage, and
   the budget it leaves goes to systems that used all of theirs, in
   proportion to their uncovered pairs (``select_runs``).
4. Each pair is *owned* by the coverage pick with its best score; a coverage
   pick keeps replay weight only on the pairs it owns, so a pair enters the
   replay once, with its best measurement.
5. A system with fewer than ``min_per_system`` graphs (one run sampled all
   its pairs) gets more, chosen to give its pairs a second good measurement;
   each of these keeps weight on every pair it measured well, so no system
   rests on a single run.

Datasets map to systems by name (``system_of_dataset``); only systems in the
manifest (the model's pretraining) are considered. ``plan_replay``
returns the plan and a per-pair report.
"""
import concurrent.futures
import heapq
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from mllf.cb.hindsight import pair_key
from mllf.file_handling.combo_sets import _rtf_signature

FULL_RUN_CROSSINGS = 10     # full-length pair test (hindsight.SUCCESS_MIN_TRANSITIONS)
SHORT_RUN_CROSSINGS = 3     # short-run test (stage explore_min_transitions)
FULL_LENGTH_NS = 1.0
COVER_QUALITY = 0.5         # a pair is fully covered at this score
TIE_WEIGHT = 1e-3           # sum of score rises, to break ties between equal coverage


# ── Datasets, systems, labels ───────────────────────────────────────────────

def dataset_of(path) -> str:
    """The pretraining dataset (directory under ``pretraining/``) of a manifest entry."""
    parts = Path(path).parts
    return parts[parts.index('pretraining') + 1] if 'pretraining' in parts else parts[0]


def system_of_dataset(name: str, known: Sequence[str]) -> str:
    """The system a dataset belongs to: the longest known system the name is
    or starts with (``12benz_solvent_group2``, ``14benz_solv_triplet_combos``,
    ``thrombin_solvent_group_combos``), reading ``_vacuum`` as ``_vac``
    (``123benz_vacuum_group1`` -> ``123benz_vac``); else the name without a
    ``_group<id>`` suffix (``p38_water_groupA`` -> ``p38_water``)."""
    for cand in (name, name.replace('_vacuum', '_vac')):
        hits = [k for k in known if cand == k or cand.startswith(k + '_')]
        if hits:
            return max(hits, key=len)
    return re.sub(r'_group[A-Za-z0-9]+$', '', name)


class FragmentIndex:
    """A system's fragments by signature (site, sub per signature)."""

    def __init__(self, prep_dir=None):
        self.by_sig: Dict[tuple, List[Tuple[int, int]]] = {}
        self.by_atoms: Dict[tuple, List[Tuple[int, int]]] = {}
        self.n_subs = 0
        for f in sorted(Path(prep_dir).glob('site*_sub*_pres.rtf')) if prep_dir else []:
            m = re.match(r'site(\d+)_sub(\d+)_', f.name)
            if m:
                self._add(_rtf_signature(f), (int(m.group(1)), int(m.group(2))))

    def _add(self, sig, label):
        self.by_sig.setdefault(sig, []).append(label)
        self.by_atoms.setdefault(sig[1], []).append(label)
        self.n_subs += 1

    @classmethod
    def from_preps(cls, prep_dirs: Sequence) -> 'FragmentIndex':
        """For a system without its own prep: every distinct fragment in its
        datasets' prep dirs, numbered per site in order of first appearance."""
        idx = cls()
        next_sub: Dict[int, int] = {}
        for prep in prep_dirs:
            for f in sorted(Path(prep).glob('site*_sub*_pres.rtf')):
                m = re.match(r'site(\d+)_sub\d+_', f.name)
                if not m:
                    continue
                site, sig = int(m.group(1)), _rtf_signature(f)
                if any(h[0] == site for h in idx.by_sig.get(sig, [])):
                    continue
                next_sub[site] = next_sub.get(site, 0) + 1
                idx._add(sig, (site, next_sub[site]))
        return idx

    def match(self, rtf, site: int) -> Optional[Tuple[int, int]]:
        sig = _rtf_signature(rtf)
        for hits in (self.by_sig.get(sig), self.by_atoms.get(sig[1])):
            if not hits:
                continue
            same = [h for h in hits if h[0] == site]
            if len(same) == 1:
                return same[0]
            if len(hits) == 1:
                return hits[0]
        return None


def node_labels(run_dir, index: Optional[FragmentIndex],
                _cache: Optional[Dict[str, Optional[Tuple[int, int]]]] = None
                ) -> Optional[List[Tuple[int, int]]]:
    """(site, sub) in the system's numbering per node of a run, in node order
    (sites in order, each site's substituents as ``active_subs_ordered``
    lists them, else by number), from ``graph_info.json`` and the fragments in the combo
    directory's ``prep/`` (else the recorded file path). Without an index,
    the run's own numbering. None if a node cannot be labelled."""
    try:
        info = json.loads((Path(run_dir) / 'graph_info.json').read_text())
    except Exception:
        return None
    nodes, order = info.get('sites') or {}, info.get('active_subs_ordered') or {}
    if not order:            # older runs: blocks follow site, then substituent
        for name, node in sorted(nodes.items(), key=lambda kv: (int(kv[1].get('site', 0)),
                                                                int(kv[1].get('sub', 0)))):
            order.setdefault(f"site{int(node.get('site', 0))}", []).append(name)
    labels = []
    for site_name in sorted(order, key=lambda s: int(re.sub(r'\D', '', s) or 0)):
        for name in order[site_name]:
            node = nodes.get(name) or {}
            site, sub = int(node.get('site', 0)), int(node.get('sub', 0))
            if index is None:
                labels.append((site, sub))
                continue
            fname = (node['master'] + '_pres.rtf') if node.get('master') else (node.get('filename') or '')
            rtf = Path(run_dir).parent / 'prep' / fname
            if not rtf.exists():
                rtf = Path(node.get('filepath') or '')
            if not rtf.is_file():
                return None
            key = str(rtf.resolve())
            if _cache is not None and key in _cache:
                hit = _cache[key]
            else:
                hit = index.match(rtf, site)
                if _cache is not None:
                    _cache[key] = hit
            if hit is None:
                return None
            labels.append(hit)
    return labels or None


# ── Pair quality ─────────────────────────────────────────────────────────────

def _top_count(block: Dict) -> float:
    counts = (block or {}).get('counts') or {}
    if not counts:
        return 0.0
    return float(counts[max(counts, key=float)])


def _finite_ddg(sim: Dict, i: int, j: int) -> bool:
    d = sim.get('ddg_pairs') or {}
    v = d.get(f"{i + 2}_{j + 2}", d.get(f"{j + 2}_{i + 2}"))
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def pair_quality(sim: Dict, i: int, j: int, site: int, site_size: int) -> float:
    """Flatness of nodes i, j (same site) in one run, 0-1: population
    balance min/max (0 if either is unsampled or the pair's ΔG was not
    measured) x crossings / the run length's test, capped at 1. Crossings
    are the pair's own, or, on a site of more than two substituents (where
    lambda moves between them through the others), the site's."""
    pops = sim.get('populations') or {}
    pi, pj = _top_count(pops.get(str(i + 2))), _top_count(pops.get(str(j + 2)))
    if pi <= 0 or pj <= 0 or not _finite_ddg(sim, i, j):
        return 0.0
    balance = min(pi, pj) / max(pi, pj)
    tp = sim.get('transition_pairs') or {}
    crossings = float(tp.get(f"{i + 2}_{j + 2}", tp.get(f"{j + 2}_{i + 2}", 0)) or 0)
    if site_size > 2:
        st = (sim.get('transitions') or {}).get(str(site)) or {}
        crossings = max(crossings, _top_count({'counts': st}))
    try:
        ns = float(sim.get('simulated_ns'))
    except (TypeError, ValueError):
        ns = FULL_LENGTH_NS
    need = SHORT_RUN_CROSSINGS if 0 < ns < 0.5 * FULL_LENGTH_NS else FULL_RUN_CROSSINGS
    return balance * min(1.0, crossings / need)


def run_pair_scores(system: str, run_dir, labels: List[Tuple[int, int]], sim: Dict
                    ) -> Dict[str, Tuple[float, int, int]]:
    """{pair key: (quality, node i, node j)} for a run's sampled within-site pairs."""
    by_site: Dict[int, List[int]] = {}
    for n, (site, _) in enumerate(labels):
        by_site.setdefault(site, []).append(n)
    out = {}
    for site, nodes in by_site.items():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                i, j = nodes[a], nodes[b]
                q = pair_quality(sim, i, j, site, len(nodes))
                if q > 0:
                    out[pair_key(system, labels[i], labels[j])[0]] = (q, i, j)
    return out


# ── Greedy coverage ──────────────────────────────────────────────────────────

def _kth_best(scores: Sequence[float], k: int) -> float:
    s = sorted(scores, reverse=True)
    return s[k - 1] if len(s) >= k else 0.0


def greedy_cover(candidates: Dict[str, Dict[str, float]], budget: int,
                 chosen: Optional[Sequence[str]] = None, level: int = 1,
                 cover_quality: float = COVER_QUALITY) -> List[str]:
    """Chosen candidate ids (``chosen`` first, then new picks up to ``budget``
    in all), picked greedily by the rise in the pairs' coverage values
    min(1, s / ``cover_quality``) where s is each pair's ``level``-th best
    chosen score (1: coverage; 2: a second good measurement), ties broken by
    the rise in s, then id (lazy greedy). Stops when no candidate adds
    coverage at that level."""
    chosen = list(chosen or [])
    scores: Dict[str, List[float]] = {}
    for c in chosen:
        for p, q in candidates[c].items():
            scores.setdefault(p, []).append(q)
    cur = {p: _kth_best(v, level) for p, v in scores.items()}

    def value(q):
        return min(1.0, q / cover_quality)

    def new_level(p, q):
        return _kth_best(scores.get(p, []) + [q], level)

    def cover_gain(c):
        return sum(max(0.0, value(new_level(p, q)) - value(cur.get(p, 0.0)))
                   for p, q in candidates[c].items())

    def gain(c):
        return cover_gain(c) + TIE_WEIGHT * sum(max(0.0, new_level(p, q) - cur.get(p, 0.0))
                                                for p, q in candidates[c].items())
    taken = set(chosen)
    heap = [(-gain(c), c) for c in candidates if c not in taken]
    heapq.heapify(heap)
    while heap and len(chosen) < budget:
        neg, c = heapq.heappop(heap)
        if cover_gain(c) <= 1e-12:
            continue
        g = gain(c)
        if heap and g < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-g, c))     # stale: re-rank
            continue
        chosen.append(c)
        for p, q in candidates[c].items():
            scores.setdefault(p, []).append(q)
            cur[p] = _kth_best(scores[p], level)
    return chosen


def owners(candidates: Dict[str, Dict[str, float]], chosen: Sequence[str]) -> Dict[str, str]:
    """pair -> the chosen candidate with its best score (ties: first chosen)."""
    out: Dict[str, Tuple[float, str]] = {}
    for c in chosen:
        for p, q in candidates[c].items():
            if p not in out or q > out[p][0]:
                out[p] = (q, c)
    return {p: c for p, (_, c) in out.items()}


def system_budgets(n_subs: Dict[str, int], total: int, min_per_system: int = 3) -> Dict[str, int]:
    """Each system's share of ``total``, proportional to its substituents."""
    s = sum(n_subs.values())
    return {k: max(int(min_per_system), int(round(total * v / s)) if s else 0) for k, v in n_subs.items()}


def uncovered(candidates: Dict[str, Dict[str, float]], chosen: Sequence[str],
              cover_quality: float = COVER_QUALITY) -> int:
    """Sampled pairs whose best chosen score is below ``cover_quality``."""
    best: Dict[str, float] = {}
    for c in chosen:
        for p, q in candidates[c].items():
            best[p] = max(best.get(p, 0.0), q)
    pairs = {p for scores in candidates.values() for p in scores}
    return sum(1 for p in pairs if best.get(p, 0.0) < cover_quality)


def select_runs(cand: Dict[str, Dict[str, Dict[str, float]]], budgets: Dict[str, int], total: int,
                min_graphs: int = 3, rounds: int = 5) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Per system ({system: {run: {pair: score}}}): (coverage picks, extra
    picks for the minimum).

    1. Coverage within each system's budget (``greedy_cover``).
    2. Budget left by systems that stopped early (everything covered) goes,
       in up to ``rounds`` rounds, to systems that used all of theirs, in
       proportion to their pairs not yet covered (``uncovered``).
    3. A system with fewer than ``min_graphs`` picks gets more, chosen for a
       second good measurement of its pairs (``greedy_cover(level=2)``), then
       for coverage of any kind if that adds nothing (so a single run does not
       carry a system), as far as its candidates allow."""
    budgets = dict(budgets)
    chosen = {s: greedy_cover(cand[s], budgets[s]) for s in cand}
    for _ in range(rounds):
        left = total - sum(len(v) for v in chosen.values())
        hungry = {s: uncovered(cand[s], chosen[s]) for s in cand
                  if len(chosen[s]) >= budgets[s] and uncovered(cand[s], chosen[s]) > 0}
        if left <= 0 or not hungry:
            break
        n = sum(hungry.values())
        grew = False
        for s, u in hungry.items():
            extra = max(1, int(left * u / n))
            budgets[s] += extra
            before = len(chosen[s])
            chosen[s] = greedy_cover(cand[s], budgets[s], chosen=chosen[s])
            grew |= len(chosen[s]) > before
        if not grew:
            break
    extra: Dict[str, List[str]] = {}
    for s in cand:
        if len(chosen[s]) >= min_graphs:
            continue
        more = greedy_cover(cand[s], min_graphs, chosen=chosen[s], level=2)
        if len(more) < min_graphs:   # no second measurement left to add: any run with sampled pairs
            rest = sorted((c for c in cand[s] if c not in more),
                          key=lambda c: (-sum(min(1.0, q / COVER_QUALITY) for q in cand[s][c].values()), c))
            more += rest[:min_graphs - len(more)]
        extra[s] = more[len(chosen[s]):]
    return chosen, extra


# ── Plan ─────────────────────────────────────────────────────────────────────

def _resolver(manifest):
    bases = [Path.cwd()] + list(Path(manifest).resolve().parents)

    def resolve(d):
        p = Path(d)
        if p.is_absolute():
            return p if p.exists() else None
        for b in bases:
            if (b / p).exists():
                return (b / p).resolve()
        return None
    return resolve


def _scan_dir(args):
    """Candidate runs of one combo directory: [(run_dir, labels, scores, source)]."""
    system, combo_dir, index, include_tagged = args
    out, cache = [], {}
    for run in sorted(Path(combo_dir).iterdir()):
        if not run.is_dir() or not (run / 'simulation_results.json').exists():
            continue
        try:
            meta = json.loads((run / 'metadata.json').read_text())
            if not meta.get('terminated_normally', False):
                continue
            tag = meta.get('collection_tag')
            if tag and not include_tagged:
                continue
            sim = json.loads((run / 'simulation_results.json').read_text())
        except Exception:
            continue
        labels = node_labels(run, index, cache)
        if not labels:
            continue
        scores = run_pair_scores(system, run, labels, sim)
        if scores:
            source = (f"online:{tag}" if tag else
                      'alf' if re.search(r'_group\d+$', Path(combo_dir).name) or
                      re.match(r'run\d', run.name) else 'manifest')
            out.append((str(run), labels, scores, source))
    return out


def plan_replay(manifest, systems_root, total: int, min_per_system: int = 3,
                include_tagged: bool = True, workers: int = 8, verbose: bool = True,
                allowed_dirs: Optional[set] = None) -> Tuple[List[Dict], List[Dict]]:
    """(plan, report). plan: one dict per chosen run {'system', 'run_dir',
    'source', 'role' ('cover' or 'second'), 'owned': [(i, j) node pairs that
    keep weight]}; report: one dict per weighted (run, pair) {'system',
    'pair', 'run_dir', 'source', 'role', 'quality', 'candidates'}.
    ``min_per_system``: the floor of each system's budget and the fewest
    graphs it gets (``select_runs``).

    ``systems_root``: directory of the systems' inputs (``<system>/prep``);
    a system without one has its substituents numbered from the distinct
    fragments of its datasets (``FragmentIndex.from_preps``).
    ``allowed_dirs``: resolved directory paths to consider (e.g. those
    pretraining built graphs for); None = every manifest directory."""
    resolve = _resolver(manifest)
    dirs = json.loads(Path(manifest).read_text())['directories']
    datasets = sorted({dataset_of(d) for d in dirs})
    known = sorted({p.name for p in Path(systems_root).iterdir() if (p / 'prep').is_dir()}
                   | {d for d in datasets if not re.search(r'_group[A-Za-z0-9]+$', d)
                      and not re.search(r'_combos(_v\d+)?$', d)})
    by_system: Dict[str, List[Path]] = {}
    for d in dirs:
        p = resolve(d)
        if p is not None and (allowed_dirs is None or str(p) in allowed_dirs):
            by_system.setdefault(system_of_dataset(dataset_of(d), known), []).append(p)
    # ALF group datasets list their runs directly (dataset/run*), not combo dirs
    for sysn, paths in list(by_system.items()):
        by_system[sysn] = sorted(set(paths))

    indexes: Dict[str, Optional[FragmentIndex]] = {}
    for sysn, paths in by_system.items():
        prep = Path(systems_root) / sysn / 'prep'
        if prep.is_dir():
            indexes[sysn] = FragmentIndex(prep)
        else:
            indexes[sysn] = FragmentIndex.from_preps([p / 'prep' for p in paths if (p / 'prep').is_dir()])
            if verbose:
                print(f"  [replay] {sysn}: no {prep}; substituents numbered from its datasets' "
                      f"fragments ({indexes[sysn].n_subs})")

    tasks = []
    for sysn, paths in by_system.items():
        for p in paths:
            if any(c.is_dir() and c.name.startswith('run') for c in p.iterdir()) and not (p / 'metadata.json').exists():
                tasks.append((sysn, p, indexes[sysn], include_tagged))
    cand: Dict[str, Dict[str, Dict[str, float]]] = {}
    nodes_of: Dict[str, Dict[str, Tuple[int, int]]] = {}
    source_of: Dict[str, str] = {}
    labels_seen: Dict[str, set] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for (sysn, *_), runs in zip(tasks, ex.map(_scan_dir, tasks)):
            for run_dir, labels, scores, source in runs:
                cand.setdefault(sysn, {})[run_dir] = {p: s[0] for p, s in scores.items()}
                nodes_of[run_dir] = {p: (s[1], s[2]) for p, s in scores.items()}
                source_of[run_dir] = source
                labels_seen.setdefault(sysn, set()).update(labels)
    n_subs = {s: (indexes[s].n_subs if indexes.get(s) and indexes[s].n_subs else len(labels_seen.get(s, ())))
              for s in cand}
    budgets = system_budgets(n_subs, total, min_per_system)
    chosen, extra = select_runs(cand, budgets, total, min_graphs=min_per_system)
    plan, report = [], []
    for sysn in sorted(cand):
        own = owners(cand[sysn], chosen[sysn])
        n_cand: Dict[str, int] = {}
        for scores in cand[sysn].values():
            for p in scores:
                n_cand[p] = n_cand.get(p, 0) + 1
        weighted: Dict[str, List[Tuple[str, str]]] = {}     # run -> [(pair, role)]
        for p, c in own.items():
            weighted.setdefault(c, []).append((p, 'cover'))
        for c in extra.get(sysn, []):
            # an extra pick: weight on every pair it measured well
            weighted[c] = [(p, 'second') for p, q in cand[sysn][c].items() if q >= COVER_QUALITY] \
                or [(p, 'second') for p in cand[sysn][c]]
        runs = [c for c in chosen[sysn] + extra.get(sysn, []) if c in weighted]
        for c in runs:
            plan.append({'system': sysn, 'run_dir': c, 'source': source_of[c],
                         'role': weighted[c][0][1], 'owned': [nodes_of[c][p] for p, _ in weighted[c]]})
            for p, role in weighted[c]:
                report.append({'system': sysn, 'pair': p, 'run_dir': c, 'source': source_of[c],
                               'role': role, 'quality': round(cand[sysn][c][p], 4),
                               'candidates': n_cand[p]})
        if verbose:
            n_ext = sum(1 for c in runs if c in extra.get(sysn, []))
            print(f"  [replay] {sysn}: {n_subs[sysn]} subs, budget {budgets[sysn]}, "
                  f"{len(cand[sysn])} candidate runs, {len(runs)} chosen"
                  + (f" ({n_ext} for the minimum)" if n_ext else '')
                  + f", {len(own)}/{len(n_cand)} sampled pairs covered "
                  f"({len(n_cand) - uncovered(cand[sysn], chosen[sysn])} at quality >= {COVER_QUALITY})")
    if verbose:
        print(f"  [replay] total: {len(plan)} graphs over {len(cand)} systems (target {total})")
    return plan, report
