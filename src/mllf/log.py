"""Logging for mllf.

Every module logs to ``logging.getLogger(__name__)``, under the ``mllf``
logger. Two destinations:

* the console (standard output), plain messages at the level set by
  ``setup_logging``: progress, per-epoch summaries, warnings and errors;
* a run's log file (``log_to_file``; e.g. ``training.log`` in the training
  output directory), which records every message, including the per-combo
  and per-run detail, with a time stamp.

``setup_logging(verbosity)`` (the ``mllf`` command's ``-q``/``-v``) sets the
console level::

    -1  -q    warnings and errors only
     0        progress (the default)
     1  -v    the same plus each command's verbose detail
     2  -vv   every message, prefixed with its level and module

Used as a library (a notebook, a job script), mllf shows the same console
messages by default. To route them elsewhere, configure the ``mllf`` logger,
e.g. ``logging.getLogger('mllf').handlers[:] = [my_handler]``.
"""
import contextlib
import logging
import sys
from pathlib import Path


LOGGER_NAME = 'mllf'
PLAIN = '%(message)s'
DEBUG_FORMAT = '%(levelname)s %(name)s: %(message)s'
FILE_FORMAT = '%(asctime)s %(message)s'
FILE_DATE = '%Y-%m-%d %H:%M:%S'


class StdoutHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stdout`` is at the time of each message, so
    redirected or captured output (``contextlib.redirect_stdout``, pytest)
    receives it."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):  # StreamHandler.__init__ assigns it
        pass


def _logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _console() -> logging.Handler:
    logger = _logger()
    for h in logger.handlers:
        if isinstance(h, StdoutHandler):
            return h
    h = StdoutHandler()
    h.setFormatter(logging.Formatter(PLAIN))
    h.setLevel(logging.INFO)
    logger.addHandler(h)
    return h


def install_default_handler() -> None:
    """Show mllf's INFO messages on stdout unless the application has
    configured the ``mllf`` logger itself (called on ``import mllf``)."""
    logger = _logger()
    if logger.handlers:
        return
    _console()
    logger.setLevel(logging.DEBUG)   # handlers choose what they keep
    logger.propagate = False


def setup_logging(verbosity: int = 0) -> None:
    """Set the console level from a -q/-v count (see the module docstring)."""
    h = _console()
    logger = _logger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if verbosity >= 2:
        h.setLevel(logging.DEBUG)
        h.setFormatter(logging.Formatter(DEBUG_FORMAT))
        return
    h.setLevel(logging.WARNING if verbosity < 0 else logging.INFO)
    h.setFormatter(logging.Formatter(PLAIN))


@contextlib.contextmanager
def log_to_file(path, mode: str = 'a'):
    """Record every mllf message (DEBUG and above) in ``path`` while active.

    Yields the path. Used by each command for its run's log file; the console
    keeps its own level. A path that is already being recorded is left as is.
    """
    path = Path(path)
    logger = _logger()
    if any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve()
           for h in logger.handlers):
        yield path              # already recording there (an enclosing call)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode=mode, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE))
    logger.addHandler(handler)
    try:
        yield path
    finally:
        logger.removeHandler(handler)
        handler.close()


def quiet_unimol_tools() -> None:
    """Keep unimol_tools' own logger (console and its logs/ file) to warnings."""
    logging.getLogger('Uni-Mol Tools').setLevel(logging.WARNING)
