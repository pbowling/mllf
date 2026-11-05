"""Logging for mllf.

Every module logs to ``logging.getLogger(__name__)``, under the ``mllf``
logger. Messages go to standard output as plain text, which is what the
modules used to ``print``: out of the box (including when mllf is used as a
library, e.g. from a notebook or job script) INFO and above are shown.

``setup_logging(verbosity)`` (used by the ``mllf`` command's ``-q``/``-v``)
changes the level::

    -1  -q    warnings and errors only
     0        progress (the default)
     1  -v    the same plus each command's verbose detail
     2  -vv   debug messages, prefixed with the module they come from

To route mllf's messages elsewhere, configure the ``mllf`` logger, e.g.
``logging.getLogger('mllf').handlers[:] = [my_handler]``.
"""
import logging
import sys


LOGGER_NAME = 'mllf'
PLAIN = '%(message)s'
DEBUG_FORMAT = '%(levelname)s %(name)s: %(message)s'


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


def _handler() -> logging.Handler:
    logger = logging.getLogger(LOGGER_NAME)
    for h in logger.handlers:
        if isinstance(h, StdoutHandler):
            return h
    h = StdoutHandler()
    h.setFormatter(logging.Formatter(PLAIN))
    logger.addHandler(h)
    return h


def install_default_handler() -> None:
    """Show mllf's INFO messages on stdout unless the application has
    configured the ``mllf`` logger itself (called on ``import mllf``)."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return
    _handler()
    logger.setLevel(logging.INFO)
    logger.propagate = False


def setup_logging(verbosity: int = 0) -> None:
    """Set mllf's message level from a -q/-v count (see the module docstring)."""
    h = _handler()
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    if verbosity < 0:
        logger.setLevel(logging.WARNING)
    elif verbosity >= 2:
        logger.setLevel(logging.DEBUG)
        h.setFormatter(logging.Formatter(DEBUG_FORMAT))
        return
    else:
        logger.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter(PLAIN))
