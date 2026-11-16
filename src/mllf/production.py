"""Production runs: independent replicas of one system with fixed biases.

``mllf production run CONFIG`` builds one directory per replica under
``out_dir`` and submits each as its own job; ``mllf production analyze
OUT_DIR`` computes the end-state free energies (``mllf.analysis.production``)
and, with ``analysis: {run: true}`` (the default), is submitted to start
when the replicas finish.

Inputs: a prep directory (``core.pdb``, ``site<i>_sub<j>_pres.rtf`` and
``_frag.pdb``, ``toppar.str``, ...), a variables file with the biases and an
MSLD script (``msld_flat.py``). ``combo:`` names a combination directory of
an online run, which supplies all three except the variables (``prep/``,
``msld_flat.py`` and ``info.py``, plus ``mapping.json`` for the
substituents' numbers in the full system).

Each replica directory holds a copy of the prep directory and of the
script, ``info.py`` and ``variables.py`` (the given variables file with the
production's ``esteps`` and ``nsteps`` appended), so a replica can be rerun
by hand. The script equilibrates for ``esteps``, then runs ``nsteps`` of
production, written to ``res/<name>_flat.lmd``. Every replica starts from
the same minimized structure; their trajectories differ through the random
numbers of the Langevin thermostat, which CHARMM seeds from the clock.
"""
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from mllf.slurm import job_script, quote


log = logging.getLogger(__name__)

DEFAULTS = {'replicas': 5, 'ns': 5.0, 'equil_ns': None, 'name': None}
ANALYSIS_DEFAULTS = {'run': True, 'discard_ns': None, 'cutoff': 0.99, 'bootstrap': 50,
                     'stride': 1, 'potts': True,
                     'slurm': {'cpus_per_task': 4, 'mem': '16G', 'time': '04:00:00'}}
DISCARD_FRACTION = 0.25        # of each replica's production, when discard_ns is not set
CONFIG_NAME = 'production.yaml'
JOBS_NAME = 'jobs.json'
_PRES = re.compile(r'site(\d+)_sub(\d+)_pres\.rtf$')


def replica_name(k: int) -> str:
    return f'replica_{k:02d}'


def count_subs(prep: Path) -> List[int]:
    """Substituents per site, from ``site<i>_sub<j>_pres.rtf``."""
    found: Dict[int, set] = {}
    for p in Path(prep).iterdir():
        m = _PRES.match(p.name)
        if m:
            found.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    if not found:
        raise ValueError(f"{prep}: no site<i>_sub<j>_pres.rtf files")
    sites = sorted(found)
    if sites != list(range(1, len(sites) + 1)):
        raise ValueError(f"{prep}: sites {sites} are not numbered 1..{len(sites)}")
    nsubs = []
    for s in sites:
        subs = sorted(found[s])
        if subs != list(range(1, len(subs) + 1)):
            raise ValueError(f"{prep}: site {s} substituents {subs} are not numbered 1..n")
        nsubs.append(len(subs))
    return nsubs


def _script_value(text: str, name: str) -> Optional[float]:
    m = re.search(rf'^{name}\s*=\s*([0-9.eE+-]+)', text, re.M)
    return float(m.group(1)) if m else None


def _temperature(*paths) -> float:
    for p in paths:
        if p and Path(p).is_file():
            m = re.search(r"\[['\"]temp['\"]\]\s*=\s*([0-9.]+)", Path(p).read_text())
            if m:
                return float(m.group(1))
    return 298.15


def resolve(config: Dict, base: Path = Path('.')) -> Dict:
    """The production settings with paths resolved and inputs checked."""
    cfg = {**DEFAULTS, **{k: v for k, v in config.items() if k not in ('analysis', 'slurm')}}
    cfg['slurm'] = config.get('slurm') or {}
    cfg['analysis'] = {**ANALYSIS_DEFAULTS, **(config.get('analysis') or {})}

    def path(v):
        return None if v is None else (base / Path(v).expanduser()).resolve()
    combo = path(cfg.get('combo'))
    cfg['combo'] = combo
    cfg['prep'] = path(cfg.get('prep')) or (combo / 'prep' if combo else None)
    cfg['script'] = path(cfg.get('script')) or (combo / 'msld_flat.py' if combo else None)
    cfg['variables'] = path(cfg.get('variables'))
    cfg['out_dir'] = path(cfg.get('out_dir'))
    for key in ('prep', 'script', 'variables', 'out_dir'):
        if cfg[key] is None:
            raise ValueError(f"production: {key} is not set" + (
                " (or give combo:)" if key in ('prep', 'script') else ''))
    for key in ('prep', 'script', 'variables'):
        if not cfg[key].exists():
            raise FileNotFoundError(cfg[key])
    cfg['nsubs'] = count_subs(cfg['prep'])
    info = path(cfg.get('info')) or (combo / 'info.py' if combo and (combo / 'info.py').is_file()
                                     else None)
    cfg['info'] = info
    cfg['temperature'] = float(cfg.get('temperature')
                               or _temperature(info, cfg['prep'] / 'alf_info.py'))
    cfg['name'] = re.sub(r'[^A-Za-z0-9_]+', '_', cfg['name'] or cfg['out_dir'].name)
    text = cfg['script'].read_text()
    cfg['timestep_ps'] = _script_value(text, 'timestep') or 0.002
    if cfg['equil_ns'] is None:
        es = _script_value(text, 'esteps')
        cfg['equil_ns'] = (es or 125000) * cfg['timestep_ps'] / 1000.0
    cfg['replicas'] = int(cfg['replicas'])
    cfg['ns'] = float(cfg['ns'])
    _check_variables(cfg)
    return cfg


def _check_variables(cfg: Dict) -> None:
    from mllf.file_handling.read_bias_coeff import parse_new
    nb = sum(cfg['nsubs'])
    data = parse_new(str(cfg['variables']))
    b = data.get('b')
    if b is None:
        raise ValueError(f"{cfg['variables']}: no bias_string with b, c, x, s")
    if len(b) != nb:
        raise ValueError(f"{cfg['variables']}: b has {len(b)} entries; {cfg['prep']} has "
                         f"{nb} substituents ({cfg['nsubs']})")


def steps(cfg: Dict):
    """(esteps, nsteps)."""
    per_ns = 1000.0 / cfg['timestep_ps']
    return int(round(cfg['equil_ns'] * per_ns)), int(round(cfg['ns'] * per_ns))


def _info_text(cfg: Dict) -> str:
    return ("import numpy as np\n"
            "info = {}\n"
            f"info['name'] = {cfg['name']!r}\n"
            f"info['nsubs'] = {list(cfg['nsubs'])}\n"
            "info['nblocks'] = int(np.sum(info['nsubs']))\n"
            "info['ncentral'] = 0\n"
            "info['nreps'] = 1\n"
            "info['nnodes'] = 1\n"
            f"info['temp'] = {cfg['temperature']}\n")


def finished(rep_dir: Path) -> bool:
    from mllf.file_handling.read_output import terminated_normally
    out = rep_dir / 'output.out'
    return out.is_file() and terminated_normally(out.read_text(errors='replace'))


def _queued() -> set:
    try:
        r = subprocess.run(['squeue', '--me', '-h', '-o', '%i'], capture_output=True, text=True,
                           timeout=60)
        return set(r.stdout.split()) if r.returncode == 0 else set()
    except (OSError, subprocess.TimeoutExpired):
        return set()


def _saved_config(cfg: Dict) -> Dict:
    out = {}
    for k, v in cfg.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def build_replica(cfg: Dict, k: int) -> Path:
    """Write replica k's directory (replacing its inputs, keeping nothing of
    an earlier attempt's output)."""
    d = cfg['out_dir'] / replica_name(k)
    for old in ('res', 'dcd', 'output.out'):
        p = d / old
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    d.mkdir(parents=True, exist_ok=True)
    if (d / 'prep').exists():
        shutil.rmtree(d / 'prep')
    shutil.copytree(cfg['prep'], d / 'prep', symlinks=True)
    shutil.copy2(cfg['script'], d / cfg['script'].name)
    (d / 'info.py').write_text(_info_text(cfg))
    es, ns = steps(cfg)
    text = cfg['variables'].read_text().rstrip('\n')
    (d / 'variables.py').write_text(
        f"{text}\n\n# production settings (mllf production)\nesteps = {es}\nnsteps = {ns}\n")
    if cfg['combo'] and (cfg['combo'] / 'mapping.json').is_file():
        shutil.copy2(cfg['combo'] / 'mapping.json', d / 'mapping.json')
    return d


def _sbatch(script: Path, cwd: Path, dependency: Optional[str] = None) -> str:
    cmd = ['sbatch'] + ([f'--dependency={dependency}'] if dependency else []) + [str(script)]
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sbatch {script} failed: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip().split()[-1]


def analysis_command(out_dir: Path) -> List[str]:
    """The analysis job's commands: this interpreter and this copy of mllf.

    A copy run from a source tree (a worktree, or ``PYTHONPATH=src``) is put
    on the job's PYTHONPATH; an installed one is already the interpreter's.
    """
    import site

    import mllf
    src = Path(mllf.__file__).resolve().parents[1]
    run = f'{quote(sys.executable)} -m mllf production analyze {quote(out_dir)}'
    site_dirs = {Path(p).resolve() for p in site.getsitepackages() + [site.getusersitepackages()]}
    if src in site_dirs:
        return [run]
    return [f'export PYTHONPATH={quote(src)}${{PYTHONPATH:+:$PYTHONPATH}}', run]


def submit(config: Dict, base: Path = Path('.'), dry_run: bool = False) -> Dict:
    """Set up and submit the replicas not finished or queued, then the
    analysis. Returns {'replicas': {name: job id or 'finished' / 'queued'},
    'analysis': job id or None}."""
    cfg = resolve(config, base)
    out = cfg['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    jobs_path = out / JOBS_NAME
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    queued = _queued() if not dry_run else set()
    (out / CONFIG_NAME).write_text(yaml.safe_dump(_saved_config(cfg), sort_keys=False))
    es, ns = steps(cfg)
    log.info(f"Production {cfg['name']}: {cfg['replicas']} replicas x {cfg['ns']:g} ns "
             f"(after {cfg['equil_ns']:g} ns of equilibration), nsubs {cfg['nsubs']}")
    log.info(f"  prep {cfg['prep']}\n  script {cfg['script']}\n  variables {cfg['variables']}"
             f"\n  out {out}")
    status, new_ids = {}, []
    for k in range(1, cfg['replicas'] + 1):
        name = replica_name(k)
        d = out / name
        if finished(d):
            status[name] = 'finished'
            continue
        if str(jobs.get(name)) in queued:
            status[name] = 'queued'
            continue
        if dry_run:
            status[name] = 'would submit'
            continue
        d = build_replica(cfg, k)
        script = d / 'run.sh'
        script.write_text(job_script(cfg['slurm'], f"{cfg['name']}_{name}", d / 'slurm.%j.out', [
            'echo $(hostname)', f'cd {quote(d)}',
            f"python3 {quote(cfg['script'].name)} --vars-file variables.py > output.out 2>&1"]))
        script.chmod(0o755)
        jid = _sbatch(script, d)
        jobs[name] = jid
        status[name] = jid
        new_ids.append(jid)
    waiting = new_ids + [str(jobs[n]) for n, s in status.items() if s == 'queued']
    analysis_id = None
    acfg = cfg['analysis']
    if acfg.get('run') and not dry_run and (waiting or all(
            s == 'finished' for s in status.values())):
        script = out / 'analysis.sh'
        script.write_text(job_script(acfg.get('slurm'), f"{cfg['name']}_analysis",
                                     out / 'analysis.%j.out', analysis_command(out)))
        script.chmod(0o755)
        analysis_id = _sbatch(script, out, f"afterany:{':'.join(waiting)}" if waiting else None)
        jobs['analysis'] = analysis_id
    jobs_path.write_text(json.dumps(jobs, indent=1))
    for name, s in status.items():
        log.info(f"  {name}: {s}")
    if analysis_id:
        log.info(f"  analysis: job {analysis_id}" + (" (after the replicas)" if waiting else ''))
    return {'replicas': status, 'analysis': analysis_id}


def load_config(out_dir: Path) -> Dict:
    p = Path(out_dir) / CONFIG_NAME
    if not p.exists():
        raise FileNotFoundError(p)
    return yaml.safe_load(p.read_text())
