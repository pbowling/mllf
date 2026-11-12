"""Per-system curriculum stages and stage-advance criteria.

With ``curriculum.per_system: true`` (always on in ``mode: deploy``) each
system moves through the curriculum stages on its own: a confident system can
leave the pair stage after a few epochs while a hard one keeps running pairs,
and every epoch's batch mixes systems at different stages. A stage's combo
filters (sites, substituents per site, whitelist) apply to the systems in it;
``systems:`` on a stage now means only those systems enter it (others skip
it). ``epochs`` is the most epochs a system spends in the stage.

A stage advances a system when its ``advance`` criterion holds (after
``min_epochs``), or once it has spent ``epochs`` there (recorded as not
converged)::

    advance:
      type: pairs_resolved     # or combos_balanced, or epochs (default)
      fraction: 0.8            # of the pairs (or combos) considered
      min_epochs: 1
      # pairs_resolved: a pair is resolved after `successes` successful runs in
      # this stage (balanced and crossing, hindsight.pair_succeeded) with its
      # offset posterior sd at most max_offset_sd for the listed coefficients.
      successes: 2
      max_offset_sd: {linear: 1.0, quadratic: 4.0}
      scope: pool              # pool: all pairs in the system's stage combos;
                               # visited: only pairs run in this stage
      # combos_balanced: the latest run of a combo in this stage had, at every
      # site, each substituent at >= min_population_ratio / n_subs of the
      # site's population and >= min_site_transitions transitions.
      min_population_ratio: 0.5
      min_site_transitions: 10
      min_combos: 1
      # scope (both types): pool = every pair / combo of the stage counts
      # (default); visited = only those run in this stage so far.

After the last stage the system is done; its final biases are exported
(``export_biases``). Progress is kept in ``<checkpoint dir>/system_stages.json``.
"""
import json
import math
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from mllf.cb.hindsight import (
    OFFSET_KEY,
    OFFSET_PRIOR_SD,
    combo_node_labels,
    pair_key,
    pair_offset_posterior,
)


STATE_FILE = 'system_stages.json'

# Offset slots per coefficient (skew/end are separate per direction).
_SLOTS = {'linear': ('linear',), 'quadratic': ('quadratic',),
          'skew': ('skew_fwd', 'skew_rev'), 'end': ('end_fwd', 'end_rev')}


def stage_applies(stage: Dict, system: str) -> bool:
    """Return whether a system is included by a stage's optional system list."""
    return not stage.get('systems') or system in stage['systems']


def next_stage(stages: Sequence[Dict], system: str, start: int = 0) -> Optional[int]:
    """Index of the first stage at or after ``start`` the system takes part in."""
    for idx in range(start, len(stages)):
        if stage_applies(stages[idx], system):
            return idx
    return None


def init_state(systems: Iterable[str], stages: Sequence[Dict]) -> Dict[str, Dict]:
    """Create initial progress state at each system's first applicable stage."""
    state = {}
    for s in systems:
        idx = next_stage(stages, s)
        state[s] = {'stage': idx, 'stage_epochs': 0, 'done': idx is None,
                    'converged': None, 'history': [], 'combos': {}}
    return state


def load_state(path) -> Optional[Dict[str, Dict]]:
    """Load saved stage state, returning None when the file is absent or invalid."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_state(path, state: Dict[str, Dict]) -> None:
    """Write per-system stage progress as JSON."""
    Path(path).write_text(json.dumps(state, indent=1))


# Pairs

def combo_sites(combo_path) -> List[Tuple[int, List[int]]]:
    """(original site, original subs) per site in a combo: from mapping.json
    when the directory exists, else from its name (``..._site1_subs_1_5`` or
    ``..._site1_subs_1_2__site2_subs_3_4``; names carry original numbers)."""
    labels = combo_node_labels(combo_path) if Path(combo_path).is_dir() else None
    if labels:
        by_site: Dict[int, List[int]] = {}
        for site, sub in labels:
            by_site.setdefault(site, []).append(sub)
        return sorted(by_site.items())
    out = []
    for part in Path(combo_path).name.split('__'):
        m = re.search(r'site(\d+)_subs_([\d_]+)', part)
        if m:
            out.append((int(m.group(1)), [int(x) for x in m.group(2).strip('_').split('_') if x]))
    return out


def combo_pair_keys(system: str, combo_path) -> List[str]:
    """Pair keys (``hindsight.pair_key``) of every within-site pair in a combo."""
    keys = []
    for site, subs in combo_sites(combo_path):
        for a in range(len(subs)):
            for b in range(a + 1, len(subs)):
                keys.append(pair_key(system, (site, subs[a]), (site, subs[b]))[0])
    return keys


def pair_resolved(entry: Optional[Dict], stage: str, successes: int = 2,
                  max_offset_sd: Optional[Dict[str, float]] = None,
                  prior_sd: Optional[Dict[str, float]] = None) -> bool:
    """``successes`` successful runs in ``stage`` (or enough slow-crossing
    runs, ``partial_resolved`` from hindsight.record_pair_outcomes) and, for
    each coefficient in ``max_offset_sd``, an offset posterior sd at most that
    (a coefficient with no observations has the prior sd); or resolved by its
    scouting run (``apply_scouting``)."""
    entry = entry or {}
    st = (entry.get('stages') or {}).get(stage) or {}
    if st.get('scout_resolved'):
        return True
    if int(st.get('successes', 0)) < successes and not st.get('partial_resolved'):
        return False
    psd = {**OFFSET_PRIOR_SD, **(prior_sd or {})}
    for dim, limit in (max_offset_sd or {}).items():
        for slot in _SLOTS.get(dim, ()):
            var = pair_offset_posterior(0.0, entry.get(slot) or [], psd[dim])[1]
            if math.sqrt(var) > float(limit):
                return False
    return True


def pair_retired(entry: Optional[Dict], stage: str, after_epochs: Optional[int],
                 min_successes: int = 0) -> bool:
    """The pair has run in ``after_epochs`` epochs of ``stage`` (``epochs``,
    see ``hindsight.record_pair_outcomes``) without resolving: it leaves the
    pool and no longer holds back its groups. None: never retired. (Callers
    check ``pair_resolved`` first; a resolved pair is not 'retired'.)
    ``min_successes`` (``retire_terms``): only a pair with at least that many
    good runs in the stage retires; one with fewer stays in the pool and
    keeps holding back its groups."""
    if after_epochs is None:
        return False
    st = ((entry or {}).get('stages') or {}).get(stage) or {}
    if int(st.get('successes', 0)) < int(min_successes):
        return False
    return int(st.get('epochs', 0)) >= int(after_epochs)


def retire_terms(key: str, retire_after: Optional[int], rule) -> Tuple[Optional[int], int]:
    """(after_epochs, min_successes) for pair ``key`` under a stage's
    retirement rule as callers resolve it: a bool / int (good runs every pair
    needs; true = 1) with ``retire_after`` for every pair, or a dict for the
    'reference' setting -- {'keys': the stage's reference pairs,
    'after_epochs': their retirement epochs, 'min_successes': good runs they
    need}; other pairs retire after ``retire_after`` with none."""
    if isinstance(rule, dict):
        if key in rule['keys']:
            return rule.get('after_epochs', retire_after), int(rule.get('min_successes', 1))
        return retire_after, 0
    return retire_after, int(rule)


def pair_settled(entry: Optional[Dict], stage: str, adv: Dict,
                 prior_sd: Optional[Dict[str, float]] = None,
                 retire_after: Optional[int] = None,
                 min_successes: int = 0) -> bool:
    """Resolved (``pair_resolved``) or retired (``pair_retired``)."""
    return (pair_resolved(entry, stage, int(adv.get('successes', 2)), adv.get('max_offset_sd'),
                          prior_sd)
            or pair_retired(entry, stage, retire_after, min_successes))


def combo_resolved(bandit_state: Optional[Dict], system: str, combo, stage: str, adv: Dict,
                   prior_sd: Optional[Dict[str, float]] = None,
                   retire_after: Optional[int] = None,
                   retire_requires_success=False) -> bool:
    """Every within-site pair of a combo is resolved (``pair_resolved``), or,
    with ``retire_after``, resolved or retired (``pair_settled``).
    ``retire_requires_success``: the stage's retirement rule (``retire_terms``)."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    keys = combo_pair_keys(system, combo)
    return bool(keys) and all(
        pair_settled(store.get(k), stage, adv, prior_sd,
                     *retire_terms(k, retire_after, retire_requires_success)) for k in keys)


def combo_pair_flags(bandit_state: Optional[Dict],
                     system: str, combo, stage: str) -> Tuple[bool, bool]:
    """(awaiting_confirm, unvisited) for a combo's pairs: any pair awaits a
    full-length confirming run; no pair has been run in this stage yet."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    keys = combo_pair_keys(system, combo)
    awaiting = any((store.get(k) or {}).get('awaiting_confirm') for k in keys)
    unvisited = all(((store.get(k) or {}).get('stages') or {}).get(stage, {}).get('visits', 0) == 0
                    for k in keys)
    return awaiting, unvisited


def pair_bracketed(entry: Optional[Dict], sticky: bool = False) -> bool:
    """The pair's b has been seen pushing it both ways (closed ``b_bracket``
    from hindsight.grow_directional_b_steps) or the bracket inverted
    (``barrier_trap``): b is near its target, so a failed run is likely too
    short to cross the barrier rather than wrongly biased. ``sticky`` (stage
    key bracketed_runs_sticky): also once it ever was (``long_runs``); a run
    sampling both substituents clears the bracket, which would otherwise send
    a slow-crossing pair back to short runs."""
    entry = entry or {}
    br = entry.get('b_bracket') or {}
    return (bool(entry.get('barrier_trap'))
            or (br.get('lo') is not None and br.get('hi') is not None)
            or (sticky and bool(entry.get('long_runs'))))


def combo_bracketed(bandit_state: Optional[Dict], system: str, combo, sticky: bool = False,
                    settled: Optional[Callable[[str, Optional[Dict]], bool]] = None) -> bool:
    """Any of the combo's pairs is bracketed (``pair_bracketed``).
    ``settled(key, entry)``: pairs it holds for (resolved or retired) keep
    long runs only while their bracket is closed, not by ``sticky``."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    return any(pair_bracketed(store.get(k), sticky and not (settled and settled(k, store.get(k))))
               for k in combo_pair_keys(system, combo))


SCOUT_TIER_UNKNOWN = 1


def combo_scout_tier(tiers: Dict[str, int], system: str, combo) -> int:
    """A combo's scouting tier (stage key scout_priority; 0 = likely hardest):
    the lowest tier of its pairs; a pair the scouting runs did not cover
    counts as SCOUT_TIER_UNKNOWN (the middle)."""
    keys = combo_pair_keys(system, combo)
    return min((int(tiers.get(k, SCOUT_TIER_UNKNOWN)) for k in keys), default=SCOUT_TIER_UNKNOWN)


def apply_scouting(sys_state: Dict, bandit_state: Dict, system: str, stage: str, summary: Dict,
                   group_names: Iterable[str]) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Apply a scouting summary (``mllf scout``, ``scout_summary.json``) to one
    system entering ``stage``, once: each of its groups whose scouting run
    sampled every substituent has passed (``groups_done``, exporting the
    scouting run; also kept in
    ``scouted_groups``), and its pairs whose scouting run passed the full-run
    test are resolved (``scout_resolved``). Returns (groups passed: name ->
    run, pairs resolved, every scouting run of the system)."""
    applied = sys_state.setdefault('scouting_applied', [])
    groups = {k.split('/', 1)[1]: v for k, v in (summary.get('groups') or {}).items()
              if k.split('/', 1)[0] == system}
    if stage in applied:
        return {}, [], []
    applied.append(stage)
    names = set(group_names)
    passed = {g: v['run'] for g, v in groups.items() if g in names and v.get('all_sampled')}
    for g, run in passed.items():
        sys_state.setdefault('groups_done', {}).setdefault(g, run)
        sys_state.setdefault('scouted_groups', {})[g] = run
    store = bandit_state.setdefault(OFFSET_KEY, {})
    resolved = [k for k in summary.get('pass_pairs') or [] if k.split('|')[0] == system]
    for k in resolved:
        stages = store.setdefault(k, {}).setdefault('stages', {})
        stages.setdefault(stage, {'visits': 0, 'successes': 0})['scout_resolved'] = True
    return passed, resolved, [v['run'] for g, v in groups.items() if g in names]


def group_ready(bandit_state: Optional[Dict], system: str, group_combo, ref_keys: set,
                stage: str, adv: Dict, prior_sd: Optional[Dict[str, float]] = None,
                fraction: float = 1.0, retire_after: Optional[int] = None,
                held: Optional[Iterable[str]] = None,
                retire_requires_success=False,
                min_resolved_per_site: int = 0) -> bool:
    """At least ``fraction`` of a group's reference pairs (its within-site pairs
    in ``ref_keys``) are resolved (``pair_resolved``), so the group can run.
    With ``retire_after``, retired pairs (``pair_retired``) are left out of
    the count: the group test, not a pair that cannot be freed, decides.
    ``held``: pair keys the group waits on regardless of ``fraction`` (pairs
    reopened after the group failed, ``reopen_pairs``); each must be
    resolved or retired. ``min_resolved_per_site`` (stage key
    group_min_resolved_per_site): a group spanning more than one site also
    needs at least that many resolved reference pairs at every site, so it
    never opens with a site that has no evidence."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    for k in held or ():
        if not pair_settled(store.get(k), stage, adv, prior_sd,
                            *retire_terms(k, retire_after, retire_requires_success)):
            return False

    def _resolved(k):
        return pair_resolved(store.get(k), stage, int(
            adv.get('successes', 2)), adv.get('max_offset_sd'), prior_sd)
    ref = [k for k in combo_pair_keys(system, group_combo) if k in ref_keys]
    if min_resolved_per_site > 0:
        by_site: Dict[str, List[str]] = {}
        for k in ref:
            by_site.setdefault(k.split('|')[1], []).append(k)
        if len(by_site) > 1 and any(sum(map(_resolved, ks)) < min_resolved_per_site
                                    for ks in by_site.values()):
            return False
    keys = [k for k in ref if _resolved(k) or not pair_retired(
        store.get(k), stage, *retire_terms(k, retire_after, retire_requires_success))]
    if not keys:
        return True
    return sum(map(_resolved, keys)) >= fraction * len(keys)


def group_failed_pairs(system: str, group_combo,
                       saved_data: Union[Dict, Sequence[Dict]]) -> List[str]:
    """Pair keys of a failed group to send back to the pair phase: the
    group's within-site pairs that include a substituent with no population.
    ``saved_data``: one failed run, or the group's failed runs since it was
    last reopened. Runs with no population at all (never at a physical
    endpoint) are left out; over the rest, the substituents empty in every
    run, else those empty in any. Every one of the group's pairs if none is
    identifiable."""
    labels = combo_node_labels(group_combo) if Path(group_combo).is_dir() else None
    runs = [saved_data] if isinstance(saved_data, dict) else list(saved_data)
    keys = combo_pair_keys(system, group_combo)
    if not labels:
        return keys
    runs = [p for p in (list(r.get('populations') or []) for r in runs) if sum(p) > 0]
    if not runs:
        return keys
    empty_in = [{labels[n] for n in range(len(labels)) if n >= len(p) or p[n] <= 0} for p in runs]
    empty = set.intersection(*empty_in) or set.union(*empty_in)
    out = []
    for site, subs in combo_sites(group_combo):
        for a in range(len(subs)):
            for b in range(a + 1, len(subs)):
                if (site, subs[a]) in empty or (site, subs[b]) in empty:
                    out.append(pair_key(system, (site, subs[a]), (site, subs[b]))[0])
    return out or keys


def held_off_sites(saved_data: Dict, cfg: Optional[Dict] = None) -> List[Dict]:
    """Sites of a run whose dominant substituent's endpoint is held off by
    several partners at once (``site_holdoff`` from
    read_output.parse_site_holdoff): it holds the site's largest lambda in at
    least ``dominant_min`` of the frames, reaches lambda >= 0.99 in under
    ``at_max`` of those, and at least ``min_partners`` partners hold
    ``partner_min`` or more of the rest. ``cfg``: bandit.group_holdoff."""
    cfg = cfg or {}
    dom_min = float(cfg.get('dominant_min', 0.7))
    at_max = float(cfg.get('at_max', 0.05))
    part_min = float(cfg.get('partner_min', 0.02))
    n_min = int(cfg.get('min_partners', 2))
    out = []
    for site in saved_data.get('site_holdoff') or []:
        partners = {b: v for b, v in (site.get('partners') or {}).items() if v >= part_min}
        if site.get('dom_frac', 0) >= dom_min and site.get(
                'at_frac', 1) < at_max and len(partners) >= n_min:
            out.append({**site, 'partners': partners})
    return out


def reopen_pairs(bandit_state: Dict, keys: Iterable[str], stage: str) -> int:
    """Send pairs back to the pair phase of ``stage``: their stage successes
    (and partial runs) and epochs restart from 0 (so they must resolve again and get the full
    retirement allowance), their revisit retirement and sticky long runs
    are cleared. Learned offsets are kept. Returns the number reopened."""
    store = bandit_state.setdefault(OFFSET_KEY, {})
    n = 0
    for k in keys:
        entry = store.setdefault(k, {})
        st = entry.setdefault('stages', {}).setdefault(stage, {'visits': 0, 'successes': 0})
        st['successes'] = 0
        st['epochs'] = 0
        st.pop('scout_resolved', None)
        st.pop('partial_runs', None)
        st.pop('partial_resolved', None)
        st.pop('last_epoch', None)
        entry.pop('long_runs', None)            # back to short runs unless b re-brackets
        entry['successes'] = 0
        entry['visits'] = 0
        entry.pop('retired', None)
        entry['reopened'] = int(entry.get('reopened', 0)) + 1
        n += 1
    return n


def pairs_progress(bandit_state: Optional[Dict], pool_keys: Iterable[str], stage: str,
                   adv: Dict, prior_sd: Optional[Dict[str, float]] = None) -> Tuple[int, int]:
    """(resolved, considered) pairs for a ``pairs_resolved`` criterion."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    keys = set(pool_keys)
    if adv.get('scope', 'pool') == 'visited':
        keys = {k for k in keys
                if ((store.get(k) or {}).get('stages') or {}).get(stage, {}).get('visits', 0) > 0}
    n = sum(pair_resolved(store.get(k), stage, int(adv.get('successes', 2)),
                          adv.get('max_offset_sd'), prior_sd) for k in keys)
    return n, len(keys)


# Combos

def site_balanced(saved_data: Dict, nsubs_per_site: Sequence[int],
                  min_population_ratio: float = 0.5, min_site_transitions: int = 10) -> bool:
    """Every site: each substituent was sampled (population > 0) and holds
    >= min_population_ratio / n_subs of the site's population, and the site
    made >= min_site_transitions transitions (CHARMM's per-site total). With
    both thresholds 0 this is "every substituent sampled"."""
    pops = list(saved_data.get('populations') or [])
    trans = saved_data.get('transitions') or []
    if isinstance(trans, dict):
        trans = [trans[k] for k in sorted(trans)]
    start = 0
    for s, n in enumerate(nsubs_per_site):
        site_pops = pops[start:start + n]
        start += n
        if len(site_pops) < n:
            return False
        total = sum(site_pops)
        if total <= 0 or min(site_pops) <= 0 or min(site_pops) / total < min_population_ratio / n:
            return False
        if s >= len(trans) or trans[s] < min_site_transitions:
            return False
    return True


FRAMES_PER_NS = 37500.0     # saved lambda frames per ns (nsavl 10), ours and ALF's


def production_balanced(saved_data: Dict, nsubs_per_site: Sequence[int], cfg: Dict) -> bool:
    """Production test (advance.production): in this one run, every
    substituent spent >= min_end_fraction of the frames at lambda >= 0.99
    (read_output.parse_endpoint_nodes) and entered its endpoint >= min_visits_per_ns
    times per ns (half its transitions with its site partners,
    transition_pairs). Pairs need not cross each other directly; Potts / DCA
    post-processing uses the site's whole state. ``min_end_share`` replaces
    the flat floor with one that scales with site size: each of a site's n
    substituents needs >= min_end_share / n of the frames (its fair share of
    min_end_share), so a large site is not held to a small site's floor.
    False when the run lacks the per-substituent counts."""
    from mllf.cb.hindsight import node_transition_counts
    nd = saved_data.get('endpoint_nodes') or {}
    frames = nd.get('frames') or 0
    if frames <= 0 or not saved_data.get('transition_pairs'):
        return False
    share = cfg.get('min_end_share')
    min_end = float(cfg.get('min_end_fraction', 0.01))
    min_visits = float(cfg.get('min_visits_per_ns', 2.0))
    ns = frames / float(cfg.get('frames_per_ns', FRAMES_PER_NS))
    core = nd.get('core') or {}
    start = 0
    for n in nsubs_per_site:
        nodes = list(range(start, start + n))
        start += n
        counts = node_transition_counts(saved_data, nodes)
        floor = float(share) / n if share is not None else min_end
        for v in nodes:
            if core.get(str(v + 2), 0) / frames < floor:
                return False
            if counts is None or counts[v] / 2.0 / ns < min_visits:
                return False
    return True


def pooled_group_test(history: Sequence[Dict], nsubs_per_site: Sequence[int],
                      min_population_ratio: float,
                      min_site_transitions: int) -> Tuple[bool, Optional[str]]:
    """Group test over a group's recent runs (stage key group_pass_window).

    ``history``: the runs, oldest first, each {'run', 'populations',
    'transitions'}. Populations and per-site transitions are summed over them
    and ``site_balanced`` applied to the totals, so with both thresholds 0 the
    group passes when every substituent was sampled in at least one run.
    Returns (passes, run to export): the run that sampled the most
    substituents, the most recent on ties.
    """
    if not history:
        return False, None
    n = max(len(h.get('populations') or []) for h in history)
    pops = [0] * n
    trans = [0] * len(nsubs_per_site)
    for h in history:
        for i, v in enumerate(h.get('populations') or []):
            pops[i] += v
        t = h.get('transitions') or []
        t = [t[k] for k in sorted(t)] if isinstance(t, dict) else list(t)
        for i, v in enumerate(t[:len(trans)]):
            trans[i] += v
    passes = site_balanced({'populations': pops, 'transitions': trans}, nsubs_per_site,
                           min_population_ratio, min_site_transitions)
    best = max(range(len(history)),
               key=lambda i: (sum(1 for v in history[i].get('populations') or [] if v > 0), i))
    return passes, history[best].get('run')


def combos_progress(sys_state: Dict, adv: Dict,
                    pool_names: Optional[Sequence[str]] = None) -> Tuple[int, int]:
    """(balanced, considered) combos in the system's current stage, from each
    combo's latest run there. scope 'pool' (default, needs ``pool_names``):
    every combo of the stage counts, so all of ALF's groups must be balanced;
    'visited': only combos run so far."""
    combos = sys_state.get('combos') or {}
    if adv.get('scope', 'pool') == 'pool' and pool_names is not None:
        names = set(pool_names)
        return sum(bool(combos.get(n)) for n in names), len(names)
    return sum(bool(v) for v in combos.values()), len(combos)


# Advancing

def criterion_met(adv: Dict, done: int, considered: int) -> bool:
    if considered <= 0:
        return False
    if adv.get('type') == 'combos_balanced' and considered < int(adv.get('min_combos', 1)):
        return False
    return done / considered >= float(adv.get('fraction', 1.0))


def advance(sys_state: Dict, stages: Sequence[Dict], system: str, epoch: int,
            reason: str, progress: Optional[Tuple[int, int]] = None) -> Optional[int]:
    """Move a system past its current stage (in place). ``reason``:
    'criterion', 'max_epochs' (its criterion never held), 'epochs' (a stage
    with no criterion ran its length) or 'no_combos'. Returns the new stage
    index, or None if it finished the last one (``done``; ``converged`` is
    True if that last stage ended on its criterion, False if it ran out of
    epochs first, None if it had no criterion)."""
    idx = sys_state['stage']
    sys_state['history'].append({
        'stage': stages[idx]['name'], 'epochs': sys_state['stage_epochs'],
        'reason': reason, 'progress': list(progress) if progress else None,
        'ended_epoch': epoch})
    nxt = next_stage(stages, system, idx + 1)
    sys_state['stage_epochs'] = 0
    sys_state['combos'] = {}
    if nxt is None:
        sys_state['done'] = True
        sys_state['converged'] = {'criterion': True, 'max_epochs': False}.get(reason)
    sys_state['stage'] = nxt
    return nxt
