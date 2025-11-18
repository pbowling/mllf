from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import subprocess
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from mllf.file_handling.read_output import parse_transitions_and_rates, parse_single_population, terminated_normally


def run_simulation_command(combo_dir: str, cmd: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[int, str, str]:
    """Run a simulation command in `combo_dir` and capture output.

    This helper supports Slurm-style submission where the invoked script
    submits a batch job (e.g. via `sbatch`) and prints the submission line
    like "Submitted batch job 12345". In that case the function will poll
    Slurm (via `squeue -h -j <id>`) until the job is no longer in the queue
    and then return (0, stdout, stderr) for the submit call and allow the
    caller to parse output files produced by the job.

    Args:
        combo_dir: working directory to run the command in.
        cmd: shell command to run. If None, default is to run `./run.sh` if
             present and executable. If no command found, returns (0,'','').
        timeout: optional timeout in seconds for the *entire* waiting period
                 for a submitted Slurm job. If None the function will wait
                 indefinitely.

    Returns:
        Tuple of (returncode, stdout, stderr). On unexpected exceptions the
        function returns (1, '', str(exception)). If a Slurm job is submitted
        the returncode refers to the submission command; job-level failures
        should be discovered by parsing simulator output files after the job
        completes.
    """
    cwd = Path(combo_dir)
    if cmd is None:
        run_sh = cwd / 'run.sh'
        if run_sh.exists() and run_sh.exists() and os.access(str(run_sh), os.X_OK):
            cmd = './run.sh'
        else:
            return 0, '', ''

    try:
        # run the submission/runner command and capture its stdout/stderr.
        # Use a short timeout for the submission itself; if this command
        # submits a Slurm job (sbatch) we'll poll Slurm for completion.
        proc = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=30)
        out = proc.stdout or ''
        err = proc.stderr or ''

        # detect Slurm submit (e.g. "Submitted batch job 12345")
        m = re.search(r"Submitted batch job (\d+)", out)
        if m:
            jobid = m.group(1)
            start = time.time()
            poll_interval = 10
            # poll squeue until job disappears or timeout exceeded
            while True:
                try:
                    sq = subprocess.run(["squeue", "-h", "-j", jobid], capture_output=True, text=True)
                    sqout = sq.stdout or ''
                except Exception:
                    sqout = ''

                # no output -> job is no longer in queue
                if not sqout.strip():
                    break

                if timeout is not None and (time.time() - start) > float(timeout):
                    return 1, out, f"timeout waiting for slurm job {jobid}"

                time.sleep(poll_interval)

        return proc.returncode, out, err
    except Exception as e:
        return 1, '', str(e)


def parse_simulation_results(combo_dir: str) -> Dict[str, Any]:
    """Discover simulation output in `combo_dir` and parse known metrics.

    The parser looks for a small set of candidate filenames (e.g. 'msld.out')
    and, failing that, inspects files for the string 'NORMAL TERMINATION'. If
    output is found and the file indicates normal termination, this function
    attempts to parse both transitions/rates and single-population blocks and
    returns a dict with keys 'terminated', 'transitions', 'rates', 'population'.

    Args:
        combo_dir: directory where the simulator wrote outputs.

    Returns:
        dict summarizing parsed outputs. If no output is found returns {}.
    """
    p = Path(combo_dir)
    candidates = ['msld.out', 'output.txt', 'output.log', 'population.txt', 'results.txt']
    found = None
    for c in candidates:
        f = p / c
        if f.exists():
            found = f
            break
    if found is None:
        for f in p.glob('*'):
            if f.is_file():
                txt = f.read_text(errors='ignore')
                if 'NORMAL TERMINATION' in txt:
                    found = f
                    break
    if found is None:
        return {}

    txt = found.read_text()
    if not terminated_normally(txt):
        return {'terminated': False}

    out: Dict[str, Any] = {'terminated': True}
    try:
        trans, rates = parse_transitions_and_rates(txt)
        out['transitions'] = trans
        out['rates'] = rates
    except Exception:
        out['transitions'] = {}
        out['rates'] = {}

    try:
        pops = parse_single_population(txt)
        out['population'] = pops
    except Exception:
        out['population'] = {}

    return out


def run_simulation_batch(manifest: str, sim_cmd: Optional[str] = None, max_workers: int = 4, timeout: Optional[int] = None) -> Dict[str, Any]:
    """Run simulations for all combos in a manifest concurrently.

    This helper reads a manifest file (one combo dir per line), runs the
    simulation command for each combo concurrently using a thread pool, waits
    for completion (or timeout) and returns a summary mapping each combo to
    its parsed results.

    Args:
        manifest: path to a manifest file listing combo directories (one per line).
        sim_cmd: shell command to run for each combo; if None the command
                 selection logic in `run_simulation_command` applies (./run.sh if present).
        max_workers: maximum concurrent worker threads to use.
        timeout: per-job timeout in seconds (passed to `run_simulation_command`).

    Returns:
        Dict mapping combo_dir -> parsed simulation results (as returned by
        `parse_simulation_results`). The function also prints a small progress
        summary to stdout.
    """
    with open(manifest, 'r', encoding='utf-8') as fh:
        combos = [ln.strip() for ln in fh if ln.strip()]
    results: Dict[str, Any] = {}
    if not combos:
        return results

    futures = {}
    start_ts = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for c in combos:
            futures[ex.submit(run_simulation_command, c, sim_cmd, timeout)] = c

        for fut in as_completed(futures):
            combo_dir = futures[fut]
            try:
                rc, so, se = fut.result()
            except Exception as e:
                results[combo_dir] = {'error': str(e)}
                continue

            if rc != 0:
                results[combo_dir] = {'returncode': rc, 'stdout': so, 'stderr': se}
            else:
                parsed = parse_simulation_results(combo_dir)
                results[combo_dir] = parsed

    dur = time.time() - start_ts
    ok = sum(1 for v in results.values() if v and v.get('terminated') is True)
    print(f"Batch run completed in {dur:.1f}s: {ok}/{len(combos)} terminated normally")
    return results
