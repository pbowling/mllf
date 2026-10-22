"""Concise per-epoch block for ``epoch_summary.log`` (per-system stages).

One block per epoch, written by ``train_workflow`` after the epoch's runs,
stage progress and validation::

    Epoch 12 Summary: 332 runs = 6 groups + 245 reference pairs (30 reversed) + 81 other pairs
      Failed to run: 12benz_solvent/group01_site1_subs_1_2_3_4_5
      Pair runs good: short 43/327 (13.1%), full-length 9/101 (8.9%)
      Groups: 6 run, 2 passed: 12fuzed_vac/group05, luis_bace_solvent/group03
      Groups back to pairs: 14benz_vac/group01 (12 pairs reopened)
      Validation: 7/48 held-out pairs good (14.6%); model accepted
      Model update: loss online 2.943 -> 2.694, replay 4.888 -> 4.457
      Bias steps: median |step| b 2.31, c 4.33, skew 0.94, end 0.39; b grown 53, bisected 88, unlocked 4; c up 57, down 26, unlocked 2; pinned 13, stranded 34, held-off ends 33
      System               Stage          Epoch  Progress     Ref pairs  Retired  Train ns
      12benz_solvent       pairs_groups   13/20  0/3 groups   4/9        0           24.8
      ...
      Stage changes: 1_FAAH_vac pairs_groups -> done (converged)
      Simulated ns: training 503.2, validation 112.0 (not counted)

"Good" is each run length's own pair test (the stage's short test for short
runs, the full test otherwise). Lines with nothing to report are left out.
``parse_epoch_block`` reads a block back for the analysis scripts.
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

# stage-change reasons (system_stages.advance) in words, and back
REASON_WORDS = {'criterion': 'converged', 'max_epochs': 'max epochs, not converged',
                'epochs': 'stage length', 'no_combos': 'no combos'}
_WORD_REASONS = {v: k for k, v in REASON_WORDS.items()}
# progress units by advance type, and back
PROGRESS_UNITS = {'groups_done': 'groups', 'pairs_resolved': 'pairs', 'combos_balanced': 'combos'}
UNIT_METRICS = {'groups': 'groups passed', 'pairs': 'pairs resolved', 'combos': 'combos balanced'}


def _frac(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else f"{n}/{d}"


def _num(x, fmt='{:.2f}') -> str:
    return 'n/a' if x is None else fmt.format(x)


def format_epoch_block(epoch: int, runs: Dict[str, int], failed: Sequence[str] = (),
                       pair_short: Optional[Dict] = None, pair_full: Optional[Dict] = None,
                       groups_run: int = 0, groups_passed: Sequence[str] = (),
                       groups_reopened: Sequence[Tuple[str, int]] = (),
                       validation: Optional[Dict] = None, gate: Optional[str] = None,
                       regression: Optional[Dict] = None, hindsight: Optional[Dict] = None,
                       systems: Sequence[Dict] = (), stage_changes: Sequence[Dict] = (),
                       ns_train: Optional[float] = None, ns_validation: Optional[float] = None) -> str:
    """The epoch's block (see the module docstring).

    runs: {'groups', 'reference', 'reversed', 'other'} combos run this epoch
    (reversed partners are counted inside 'reference'). pair_short /
    pair_full / validation: pair-success dicts ('both', 'total'). systems:
    one dict per active system with 'system', 'stage', 'stage_epoch',
    'stage_max', 'progress' ((done, total, unit) or None), 'ref' ((resolved,
    total) or None), 'retired' (int or None), 'ns'. stage_changes: dicts
    with 'system', 'stage', 'next' (None = done) and 'reason'."""
    total = sum(int(runs.get(k, 0)) for k in ('groups', 'reference', 'other'))
    parts = [f"{runs.get('groups', 0)} groups",
             f"{runs.get('reference', 0)} reference pairs"
             + (f" ({runs['reversed']} reversed)" if runs.get('reversed') else ''),
             f"{runs.get('other', 0)} other pairs"]
    lines = [f"Epoch {epoch} Summary: {total} runs = " + ' + '.join(parts)]
    if failed:
        lines.append("  Failed to run: " + ', '.join(failed))
    good = []
    for lab, ps in (('short', pair_short), ('full-length', pair_full)):
        if ps and ps.get('total'):
            good.append(f"{lab} {_frac(int(ps.get('both', 0)), int(ps['total']))}")
    if good:
        lines.append("  Pair runs good: " + ', '.join(good))
    if groups_run or groups_passed:
        lines.append(f"  Groups: {groups_run} run, {len(groups_passed)} passed"
                     + (": " + ', '.join(groups_passed) if groups_passed else ''))
    if groups_reopened:
        lines.append("  Groups back to pairs: " + ', '.join(
            f"{g} ({n} pairs reopened)" for g, n in groups_reopened))
    if validation and validation.get('total'):
        lines.append(f"  Validation: {_frac(int(validation.get('both', 0)), int(validation['total']))}"
                     f" held-out pairs good" + (f"; {gate}" if gate else ''))
    if regression:
        lines.append(f"  Model update: loss online {_num(regression.get('online_before'), '{:.3f}')} -> "
                     f"{_num(regression.get('online_after'), '{:.3f}')}, replay "
                     f"{_num(regression.get('replay_before'), '{:.3f}')} -> "
                     f"{_num(regression.get('replay_after'), '{:.3f}')}")
    if hindsight:
        med = hindsight.get('median_step') or {}
        names = (('linear', 'b'), ('quadratic', 'c'), ('skew', 'skew'), ('end', 'end'))
        lines.append(
            "  Bias steps: median |step| "
            + ', '.join(f"{lab} {_num(med.get(d))}" for d, lab in names if d in med)
            + f"; b grown {hindsight.get('b_grown', 0)}, bisected {hindsight.get('b_bisected', 0)}, "
              f"unlocked {hindsight.get('b_unlocked', 0)}"
            + f"; c up {hindsight.get('c_up', 0)}, down {hindsight.get('c_down', 0)}, "
              f"unlocked {hindsight.get('c_unlocked', 0)}"
            + f"; pinned {hindsight.get('pinned', 0)}, stranded {hindsight.get('stranded', 0)}, "
              f"held-off ends {hindsight.get('held_off', 0)}")
    if systems:
        w = max(20, max(len(r['system']) for r in systems) + 1)
        sw = max(14, max(len(str(r['stage'])) for r in systems) + 1)
        lines.append(f"  {'System':{w}s} {'Stage':{sw}s} {'Epoch':6s} {'Progress':12s} "
                     f"{'Ref pairs':10s} {'Retired':8s} {'Train ns':>8s}")
        for r in systems:
            prog = r.get('progress')
            prog_s = f"{prog[0]}/{prog[1]} {prog[2]}" if prog else '-'
            ref = r.get('ref')
            ref_s = f"{ref[0]}/{ref[1]}" if ref else '-'
            ret = r.get('retired')
            lines.append(f"  {r['system']:{w}s} {str(r['stage']):{sw}s} "
                         f"{str(r['stage_epoch']) + '/' + str(r['stage_max']):6s} {prog_s:12s} "
                         f"{ref_s:10s} {('-' if ret is None else str(ret)):8s} {r.get('ns') or 0.0:8.1f}")
    if stage_changes:
        lines.append("  Stage changes: " + '; '.join(
            f"{c['system']} {c['stage']} -> {c['next'] or 'done'} "
            f"({REASON_WORDS.get(c['reason'], c['reason'])})" for c in stage_changes))
    if ns_train is not None:
        lines.append(f"  Simulated ns: training {ns_train:.1f}"
                     + (f", validation {ns_validation:.1f} (not counted)" if ns_validation else ''))
    return '\n'.join(lines)


# ── Parsing (analysis scripts) ───────────────────────────────────────────────

_GOOD_RE = re.compile(r"Pair runs good: (.*)")
_GOOD_PART_RE = re.compile(r"(short|full-length) (\d+)/(\d+)")
_VAL_RE = re.compile(r"Validation: (\d+)/(\d+)(?: \([\d.]+%\))? held-out pairs good(?:; (.*))?")
_MODEL_RE = re.compile(r"Model update: loss online (\S+) -> (\S+), replay (\S+) -> (\S+)")
_STEPS_RE = re.compile(r"Bias steps: median \|step\| (.*?);")
_STEP_PART_RE = re.compile(r"\b(b|c|skew|end) ([\d.]+|n/a)")
_ROW_RE = re.compile(r"^  (\S+)\s+(\S+)\s+(\d+)/(\d+)\s+(?:(\d+)/(\d+) (groups|pairs|combos)|-)\s+"
                     r"(?:(\d+)/(\d+)|-)\s+(\d+|-)\s+([\d.]+)$", re.M)
_CHANGE_RE = re.compile(r"(\S+) (\S+) -> (\S+) \(([^)]*)\)")


def _float(v: str) -> float:
    return float('nan') if v in ('n/a', 'nan') else float(v)


def parse_epoch_block(block: str) -> Dict:
    """What ``format_epoch_block`` wrote (the text after the header line):
    {'short': (good, total), 'full': ..., 'validation': (good, total),
    'gate': 'restored' | 'best' | 'accepted' | None, 'regression': (online
    before, after, replay before, after), 'median_step': {head: value},
    'systems': [row dicts], 'changes': {system: (next, reason)}}; keys
    without a line are missing."""
    out: Dict = {}
    m = _GOOD_RE.search(block)
    if m:
        for lab, n, d in _GOOD_PART_RE.findall(m.group(1)):
            out['short' if lab == 'short' else 'full'] = (int(n), int(d))
    m = _VAL_RE.search(block)
    if m:
        out['validation'] = (int(m.group(1)), int(m.group(2)))
        rest = m.group(3) or ''
        out['gate'] = ('restored' if 'restored' in rest else 'best' if 'new best' in rest
                       else 'accepted' if 'accepted' in rest else None)
    m = _MODEL_RE.search(block)
    if m:
        out['regression'] = tuple(_float(v.rstrip(',')) for v in m.groups())
    m = _STEPS_RE.search(block)
    if m:
        head = {'b': 'linear', 'c': 'quadratic', 'skew': 'skew', 'end': 'end'}
        out['median_step'] = {head[k]: _float(v) for k, v in _STEP_PART_RE.findall(m.group(1))}
    rows: List[Dict] = []
    for r in _ROW_RE.finditer(block):
        if r.group(1) == 'System':
            continue
        rows.append({'system': r.group(1), 'stage': r.group(2), 'stage_epoch': int(r.group(3)),
                     'stage_max': int(r.group(4)),
                     'progress': ((int(r.group(5)), int(r.group(6)), UNIT_METRICS[r.group(7)])
                                  if r.group(5) else None),
                     'ref': (int(r.group(8)), int(r.group(9))) if r.group(8) else None,
                     'retired': None if r.group(10) == '-' else int(r.group(10)),
                     'ns': float(r.group(11))})
    if rows:
        out['systems'] = rows
    changes = {}
    for line in block.splitlines():
        if line.strip().startswith('Stage changes:'):
            for sysn, _stage, nxt, why in _CHANGE_RE.findall(line):
                changes[sysn] = (nxt, _WORD_REASONS.get(why, why))
    if changes:
        out['changes'] = changes
    return out
