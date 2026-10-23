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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


# ── Pairs ────────────────────────────────────────────────────────────────────

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
    """``successes`` successful runs in ``stage`` and, for each coefficient in
    ``max_offset_sd``, an offset posterior sd at most that (a coefficient with
    no observations has the prior sd)."""
    entry = entry or {}
    st = (entry.get('stages') or {}).get(stage) or {}
    if int(st.get('successes', 0)) < successes:
        return False
    psd = {**OFFSET_PRIOR_SD, **(prior_sd or {})}
    for dim, limit in (max_offset_sd or {}).items():
        for slot in _SLOTS.get(dim, ()):
            var = pair_offset_posterior(0.0, entry.get(slot) or [], psd[dim])[1]
            if math.sqrt(var) > float(limit):
                return False
    return True


def pair_retired(entry: Optional[Dict], stage: str, after_epochs: Optional[int]) -> bool:
    """The pair has run in ``after_epochs`` epochs of ``stage`` (``epochs``,
    see ``hindsight.record_pair_outcomes``) without resolving: it leaves the
    pool and no longer holds back its groups. None: never retired. (Callers
    check ``pair_resolved`` first; a resolved pair is not 'retired'.)"""
    if after_epochs is None:
        return False
    st = ((entry or {}).get('stages') or {}).get(stage) or {}
    return int(st.get('epochs', 0)) >= int(after_epochs)


def pair_settled(entry: Optional[Dict], stage: str, adv: Dict,
                 prior_sd: Optional[Dict[str, float]] = None,
                 retire_after: Optional[int] = None) -> bool:
    """Resolved (``pair_resolved``) or retired (``pair_retired``)."""
    return (pair_resolved(entry, stage, int(adv.get('successes', 2)), adv.get('max_offset_sd'), prior_sd)
            or pair_retired(entry, stage, retire_after))


def combo_resolved(bandit_state: Optional[Dict], system: str, combo, stage: str, adv: Dict,
                   prior_sd: Optional[Dict[str, float]] = None,
                   retire_after: Optional[int] = None) -> bool:
    """Every within-site pair of a combo is resolved (``pair_resolved``), or,
    with ``retire_after``, resolved or retired (``pair_settled``)."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    keys = combo_pair_keys(system, combo)
    return bool(keys) and all(
        pair_settled(store.get(k), stage, adv, prior_sd, retire_after) for k in keys)


def combo_pair_flags(bandit_state: Optional[Dict], system: str, combo, stage: str) -> Tuple[bool, bool]:
    """(awaiting_confirm, unvisited) for a combo's pairs: any pair awaits a
    full-length confirming run; no pair has been run in this stage yet."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    keys = combo_pair_keys(system, combo)
    awaiting = any((store.get(k) or {}).get('awaiting_confirm') for k in keys)
    unvisited = all(((store.get(k) or {}).get('stages') or {}).get(stage, {}).get('visits', 0) == 0
                    for k in keys)
    return awaiting, unvisited


def group_ready(bandit_state: Optional[Dict], system: str, group_combo, ref_keys: set,
                stage: str, adv: Dict, prior_sd: Optional[Dict[str, float]] = None,
                fraction: float = 1.0, retire_after: Optional[int] = None,
                held: Optional[Iterable[str]] = None) -> bool:
    """At least ``fraction`` of a group's reference pairs (its within-site pairs
    in ``ref_keys``) are resolved (``pair_resolved``), so the group can run.
    With ``retire_after``, retired pairs (``pair_retired``) are left out of
    the count: the group test, not a pair that cannot be freed, decides.
    ``held``: pair keys the group waits on regardless of ``fraction`` (pairs
    reopened after the group failed, ``reopen_pairs``); each must be
    resolved or retired."""
    store = (bandit_state or {}).get(OFFSET_KEY, {})
    for k in held or ():
        if not pair_settled(store.get(k), stage, adv, prior_sd, retire_after):
            return False
    keys = [k for k in combo_pair_keys(system, group_combo) if k in ref_keys]
    keys = [k for k in keys
            if pair_resolved(store.get(k), stage, int(adv.get('successes', 2)), adv.get('max_offset_sd'), prior_sd)
            or not pair_retired(store.get(k), stage, retire_after)]
    if not keys:
        return True
    n = sum(pair_resolved(store.get(k), stage, int(adv.get('successes', 2)),
                          adv.get('max_offset_sd'), prior_sd) for k in keys)
    return n >= fraction * len(keys)


def group_failed_pairs(system: str, group_combo, saved_data: Dict) -> List[str]:
    """Pair keys of a failed group run to send back to the pair phase: the
    group's within-site pairs that include a substituent with no population
    (every one of the group's pairs if none is identifiable)."""
    labels = combo_node_labels(group_combo) if Path(group_combo).is_dir() else None
    pops = list(saved_data.get('populations') or [])
    keys = combo_pair_keys(system, group_combo)
    if not labels:
        return keys
    empty = {labels[n] for n in range(len(labels)) if n >= len(pops) or pops[n] <= 0}
    out = []
    for site, subs in combo_sites(group_combo):
        for a in range(len(subs)):
            for b in range(a + 1, len(subs)):
                if (site, subs[a]) in empty or (site, subs[b]) in empty:
                    out.append(pair_key(system, (site, subs[a]), (site, subs[b]))[0])
    return out or keys


def reopen_pairs(bandit_state: Dict, keys: Iterable[str], stage: str) -> int:
    """Send pairs back to the pair phase of ``stage``: their stage successes
    and epochs restart from 0 (so they must resolve again and get the full
    retirement allowance) and their revisit retirement is cleared. Learned
    offsets are kept. Returns the number reopened."""
    store = bandit_state.setdefault(OFFSET_KEY, {})
    n = 0
    for k in keys:
        entry = store.setdefault(k, {})
        st = entry.setdefault('stages', {}).setdefault(stage, {'visits': 0, 'successes': 0})
        st['successes'] = 0
        st['epochs'] = 0
        st.pop('last_epoch', None)
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


# ── Combos ───────────────────────────────────────────────────────────────────

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


# ── Advancing ────────────────────────────────────────────────────────────────

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
