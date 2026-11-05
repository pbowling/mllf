"""Archive an online run's combo directories as one ``.tar.gz`` per system.

A long run leaves tens of thousands of files per system (one directory per
combo, one subdirectory per run). The ``archive:`` config block packs each
system's ``out_dir`` into ``<archive dir>/<system>.tar.gz``::

    archive:
      when: end        # 'end': after training; 'system': as each system finishes its
                       # last stage (per-system curricula), the rest at the end
      dir: archive     # relative to output.base_dir (or absolute)
      remove: false    # delete the combo directories once their archive is verified

``mllf archive CONFIG`` does the same for a finished or stopped run.
"""
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


log = logging.getLogger(__name__)

WHEN = ('end', 'system')


def archive_config(config: Dict) -> Optional[Dict]:
    """The config's ``archive:`` block with defaults and the archive directory
    resolved, or None when absent or ``enabled: false``."""
    cfg = config.get('archive')
    if cfg is True:
        cfg = {'enabled': True}
    if not isinstance(cfg, dict) or not cfg or cfg.get('enabled', True) is False:
        return None
    when = cfg.get('when', 'end')
    if when not in WHEN:
        raise ValueError(f"archive.when must be one of {WHEN}, got {when!r}")
    base = Path((config.get('output') or {}).get('base_dir', 'checkpoints'))
    dest = Path(cfg.get('dir', 'archive'))
    return {'when': when, 'dir': dest if dest.is_absolute() else base / dest,
            'remove': bool(cfg.get('remove', False))}


def _entries(src: Path) -> int:
    """Files and directories under ``src``, itself included."""
    return 1 + sum(len(dirs) + len(files) for _, dirs, files in os.walk(src))


def archive_directory(src: Path, dest: Path, remove: bool = False) -> Path:
    """Write ``src`` to ``dest`` (``.tar.gz``), with ``src``'s name as the
    top-level directory. The archive is written under a temporary name and
    renamed once complete; with ``remove`` the source is deleted only after
    the archive is read back and holds every file and directory of it."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f'.{dest.name}.{os.getpid()}.tmp')
    with tarfile.open(tmp, 'w:gz') as tar:
        tar.add(src, arcname=src.name)
    with tarfile.open(tmp, 'r:gz') as tar:
        n_archived = len(tar.getmembers())
    n_source = _entries(src)
    if n_archived != n_source:
        tmp.unlink()
        raise RuntimeError(f"archive of {src} holds {n_archived} entries, the directory "
                           f"{n_source}; nothing removed")
    os.replace(tmp, dest)
    if remove:
        shutil.rmtree(src)
    return dest


def archive_systems(systems_cfg: Iterable[Dict], dest_dir: Path,
                    names: Optional[Iterable[str]] = None, remove: bool = False) -> List[Path]:
    """Archive each system's ``out_dir`` to ``dest_dir/<name>.tar.gz``
    (only ``names`` when given). Systems without an ``out_dir`` on disk are
    skipped; a failed archive is reported and leaves the directory in place."""
    wanted = set(names) if names is not None else None
    written = []
    for sys_cfg in systems_cfg:
        name = sys_cfg.get('name')
        if wanted is not None and name not in wanted:
            continue
        src = Path(sys_cfg.get('out_dir') or '')
        if not sys_cfg.get('out_dir') or not src.is_dir():
            log.debug(f"archive: {name} has no combo directory ({src}); skipped")
            continue
        try:
            path = archive_directory(src, Path(dest_dir) / f'{name}.tar.gz', remove=remove)
        except Exception as err:
            log.warning(f"  Warning: could not archive {name}: {err}")
            continue
        removed = "; combo directories removed" if remove else ""
        log.info(f"  Archived {name} to {path}{removed}")
        written.append(path)
    return written
