"""Lambda trajectories from CHARMM's binary lambda file (``.lmd``).

Dynamics with ``iunldm`` writes a Fortran unformatted file: a header (the
``MSLD`` control record, the time step, a title, the variable biases, the
site of each block, the temperature and the fixed biases), then two records
per saved frame, the lambda of every block (block 1, the environment, first)
and the theta of every substituent block. Reading the file directly is much
faster than parsing the ``traj lamb print`` lines of the output, and it
holds every frame even when the output was cut short.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


AKMA_PS = 0.04888821          # one AKMA time unit in ps


def _record(buf: bytes, pos: int) -> Tuple[bytes, int]:
    n = int(np.frombuffer(buf, '<i4', 1, pos)[0])
    end = pos + 4 + n
    if end + 4 > len(buf) or int(np.frombuffer(buf, '<i4', 1, end)[0]) != n:
        raise ValueError(f"bad record at byte {pos}")
    return buf[pos + 4:end], end + 4


def read_lmd(path) -> Tuple[np.ndarray, Dict]:
    """(lambda frames, header) of one ``.lmd`` file.

    Frames are float32, one row per saved frame and one column per
    substituent block (the environment block dropped). The header holds
    'nblocks' (substituent blocks), 'nsites', 'nsavl', 'dt_ps' (time between
    frames), 'temperature', 'fixed_bias' (b of each substituent block) and
    'frames_in_header'. A file cut short (a job that stopped) gives the
    frames written before the cut."""
    buf = Path(path).read_bytes()
    rec, pos = _record(buf, 0)
    if rec[:4] not in (b'MSLD', b'LAMB'):
        raise ValueError(f"{path}: not a lambda file (header {rec[:4]!r})")
    icntrl = np.frombuffer(rec, '<i4', 20, 4)
    nfile, nsavl, nblocks, nsites = (int(icntrl[0]), int(icntrl[2]), int(icntrl[6]),
                                     int(icntrl[10]))
    rec, pos = _record(buf, pos)
    delta = float(np.frombuffer(rec, '<f4', 1)[0])
    _, pos = _record(buf, pos)                          # title
    _, pos = _record(buf, pos)                          # number of variable biases
    _, pos = _record(buf, pos)                          # the variable biases
    _, pos = _record(buf, pos)                          # site of each block
    rec, pos = _record(buf, pos)
    temperature = float(np.frombuffer(rec, '<f4', 1)[0])
    rec, pos = _record(buf, pos)
    fixed = np.frombuffer(rec, '<f4').astype(float)
    header = {'nblocks': nblocks - 1, 'nsites': nsites - 1, 'nsavl': nsavl,
              'dt_ps': delta * AKMA_PS * nsavl, 'temperature': temperature,
              'fixed_bias': fixed[1:], 'frames_in_header': nfile}
    if pos >= len(buf):
        return np.zeros((0, nblocks - 1), np.float32), header
    lam_len = int(np.frombuffer(buf, '<i4', 1, pos)[0])
    theta_len = int(np.frombuffer(buf, '<i4', 1, pos + lam_len + 8)[0])
    if lam_len != 4 * nblocks:
        raise ValueError(f"{path}: frame of {lam_len // 4} lambdas for {nblocks} blocks")
    size = lam_len + theta_len + 16
    n = (len(buf) - pos) // size
    words = np.frombuffer(buf, '<f4', n * size // 4, pos).reshape(n, size // 4)
    return np.ascontiguousarray(words[:, 2:1 + nblocks]), header
