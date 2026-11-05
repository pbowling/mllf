"""SLURM job scripts from a config's ``slurm:`` block.

Every key of the block becomes an ``#SBATCH`` option, so any sbatch option
can be set without code changes (underscores become dashes)::

    slurm:
      partition: gpu2080
      gres: gpu:1
      time: '02:00:00'
      exclude: [gollum003, gollum007]   # or 'gollum[003,007]'
      exclude_file: bad_nodes.txt       # re-read at every submission
      qos: normal
      mem: 8G
      mail_type: FAIL                   # -> --mail-type=FAIL
      requeue: true                     # -> --requeue
      module: charmm/charmm/c51a1       # or a list; 'module load' lines
      setup:                            # shell lines run before the simulation
        - source ~/miniconda3/etc/profile.d/conda.sh
        - conda activate charmm
      extra:                            # raw directives, written as given
        - '--signal=B:USR1@60'

Values: a list is joined with commas, ``true`` writes a bare flag, ``null`` or
``false`` leaves the option out. ``ntasks: 1``, ``cpus_per_task: 1``,
``export: ALL`` and ``time: '01:00:00'`` are the defaults; set a key to
override one, or to ``null`` to drop it. ``job_name`` and ``output`` are set
by mllf for each job.

``exclude_file`` lists node names (whitespace, comma or line separated; ``#``
starts a comment). It is read whenever a job script is written, so adding a
node to it takes effect for the next submission of a running training job;
its nodes are excluded together with ``exclude``.
"""
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULTS = {'ntasks': 1, 'cpus_per_task': 1, 'export': 'ALL', 'time': '01:00:00'}
RESERVED = ('job_name', 'output')
# keys of the block that are not sbatch options
NOT_OPTIONS = ('module', 'setup', 'extra', 'exclude_file')


def _as_list(value) -> List[str]:
    if value is None or value is False:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v) != '']
    return [str(value)]


def read_exclude_file(path) -> List[str]:
    """Node names listed in ``path`` (missing file: none)."""
    p = Path(path).expanduser()
    if not p.exists():
        return []
    names = []
    for line in p.read_text().splitlines():
        line = line.split('#', 1)[0]
        names.extend(n for n in line.replace(',', ' ').split() if n)
    return names


def excluded_nodes(slurm_cfg: Dict) -> List[str]:
    """``exclude`` plus the nodes in ``exclude_file``, in order, without repeats."""
    nodes = _as_list(slurm_cfg.get('exclude'))
    if slurm_cfg.get('exclude_file'):
        nodes += read_exclude_file(slurm_cfg['exclude_file'])
    return list(dict.fromkeys(nodes))


def sbatch_directives(slurm_cfg: Optional[Dict], job_name: str, output) -> List[str]:
    """The ``#SBATCH`` lines for one job."""
    cfg = dict(slurm_cfg or {})
    bad = [k for k in RESERVED if k in cfg or k.replace('_', '-') in cfg]
    if bad:
        raise ValueError(f"slurm: {', '.join(bad)} is set by mllf for each job; "
                         "remove it from the slurm block")
    options = dict(DEFAULTS)
    options.update({k: v for k, v in cfg.items() if k not in NOT_OPTIONS})
    excluded = excluded_nodes(cfg)
    options['exclude'] = excluded or None

    lines = [f"#SBATCH --job-name={job_name}", f"#SBATCH --output={output}"]
    for key, value in options.items():
        flag = '--' + str(key).replace('_', '-')
        if value is None or value is False:
            continue
        if value is True:
            lines.append(f"#SBATCH {flag}")
        elif isinstance(value, (list, tuple)):
            if value:
                lines.append(f"#SBATCH {flag}={','.join(str(v) for v in value)}")
        else:
            lines.append(f"#SBATCH {flag}={value}")
    for raw in _as_list(cfg.get('extra')):
        raw = raw.strip()
        lines.append(raw if raw.startswith('#SBATCH') else f"#SBATCH {raw}")
    return lines


def setup_lines(slurm_cfg: Optional[Dict]) -> List[str]:
    """``module load`` lines, then the ``setup`` commands."""
    cfg = slurm_cfg or {}
    lines = [f"module load {m}" for m in _as_list(cfg.get('module'))]
    return lines + _as_list(cfg.get('setup'))


def job_script(slurm_cfg: Optional[Dict], job_name: str, output,
               commands: Iterable[str]) -> str:
    """A complete batch script: directives, setup, then ``commands``."""
    parts = ['#!/bin/bash', *sbatch_directives(slurm_cfg, job_name, output), '']
    setup = setup_lines(slurm_cfg)
    if setup:
        parts += setup + ['']
    parts += list(commands)
    return '\n'.join(parts) + '\n'


def quote(path) -> str:
    """A path for a shell command line."""
    return shlex.quote(str(path))


def refresh_exclude(script_path, slurm_cfg: Optional[Dict]) -> None:
    """Rewrite a written job script's ``--exclude`` line from the current
    ``exclude`` / ``exclude_file`` (before resubmitting it)."""
    path = Path(script_path)
    if not path.exists():
        return
    lines = [ln for ln in path.read_text().splitlines()
             if not ln.startswith('#SBATCH --exclude=')]
    nodes = excluded_nodes(slurm_cfg or {})
    if nodes:
        at = next((i + 1 for i, ln in enumerate(lines) if ln.startswith('#SBATCH --output=')), 1)
        lines.insert(at, f"#SBATCH --exclude={','.join(nodes)}")
    path.write_text('\n'.join(lines) + '\n')
