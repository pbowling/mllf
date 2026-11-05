"""Deduplicated summary of the warnings raised during a long run."""
import collections
import logging
import re
import threading
import warnings
from pathlib import Path


log = logging.getLogger(__name__)


class WarningSummary:
    """Collect warnings while active and print one deduplicated summary
    (count per message, with paths masked) when done.

    Each entry shows its first full message (``width`` characters), or with
    ``show_template`` the path-masked message the entries are grouped by."""
    _PATH_RE = re.compile(r'/[^\s:*?"<>|]+')

    def __init__(self, show_template: bool = False, width: int = 200):
        self.show_template = show_template
        self.width = width
        self.counts = collections.Counter()
        self.example = {}
        self._lock = threading.Lock()
        self._orig = None

    def _collect(self, message, category, filename, lineno, file=None, line=None):
        text = str(message)
        key = (category.__name__, self._PATH_RE.sub('<path>', text))
        with self._lock:
            self.counts[key] += 1
            self.example.setdefault(key, (text, filename, lineno))

    def __enter__(self):
        self._orig = warnings.showwarning
        warnings.showwarning = self._collect
        return self

    def __exit__(self, *exc):
        warnings.showwarning = self._orig
        if exc[0] is None:
            self.report()
        return False

    def report(self) -> None:
        if not self.counts:
            return
        log.warning(f"Warning Summary ({sum(self.counts.values())} total occurrences, "
                    f"{len(self.counts)} unique types)")
        for (cat, template), count in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            example_text, ex_file, ex_line = self.example[(cat, template)]
            short_file = Path(ex_file).name if ex_file else '?'
            log.warning(f"  [{count:5d}x] {cat} ({short_file}:{ex_line})")
            shown = template if self.show_template else example_text
            log.warning(f"           {shown[:self.width]}")
