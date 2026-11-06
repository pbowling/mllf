"""Shared pieces of the analyses: dependency check, environments, the output
directory (figures, tables, report) and a small pickle cache."""
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence


log = logging.getLogger(__name__)

ENVIRONMENTS = ('vacuum', 'solvent', 'protein')


def require_plotting() -> None:
    """Raise a clear error when the analysis dependencies are missing."""
    missing = []
    for mod in ('matplotlib', 'pandas', 'PIL'):
        try:
            __import__(mod)
        except ImportError:
            missing.append('pillow' if mod == 'PIL' else mod)
    if missing:
        raise ImportError(f"mllf analyze needs {', '.join(missing)}: "
                          "pip install 'mllf[analysis]'")
    # mathtext's font setup reports missing fallback families (e.g. 'cursive')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def environment_of(system: str, solvent_state: Optional[str] = None) -> str:
    """'vacuum', 'solvent' or 'protein', from the system's solvent_state when
    given, else from its name's suffix."""
    state = (solvent_state or '').lower()
    if state in ('gas', 'vacuum', 'vac'):
        return 'vacuum'
    if state == 'protein':
        return 'protein'
    if state in ('solv', 'solvent'):
        return 'solvent'
    n = system.lower()
    if n.endswith(('_vac', '_vacuum', '_gas')):
        return 'vacuum'
    if n.endswith(('_protein', '_prot')):
        return 'protein'
    return 'solvent'


def system_order(systems: Iterable[str], env: Optional[Dict[str, str]] = None) -> List[str]:
    """Systems by environment (vacuum, solvent, protein), then name."""
    env = env or {}
    rank = {e: i for i, e in enumerate(ENVIRONMENTS)}
    return sorted(set(systems), key=lambda s: (rank[env.get(s) or environment_of(s)], s))


def load_cache(path: Path) -> Dict:
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}


def save_cache(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(data, f)
    tmp.replace(path)


FORMATS = ('png', 'pdf')


class Output:
    """The analysis output directory: ``figures/`` (each figure as PNG, the
    default, or as a vector PDF with editable text; ``grayscale`` adds a
    grayscale PNG preview), ``tables/`` (one CSV per figure or table) and
    ``report.txt``. ``only`` limits which figures (or folders of figures)
    are made."""

    def __init__(self, out_dir, only: Optional[Sequence[str]] = None, fmt: str = 'png',
                 grayscale: bool = False):
        if fmt not in FORMATS:
            raise ValueError(f"figure format {fmt!r}; choose from {', '.join(FORMATS)}")
        self.fmt = fmt
        self.grayscale = grayscale
        self.dir = Path(out_dir)
        self.figures = self.dir / 'figures'
        self.tables = self.dir / 'tables'
        for d in (self.figures, self.tables):
            d.mkdir(parents=True, exist_ok=True)
        self.only = set(only) if only else None
        self.written: List[str] = []
        self.skipped: Dict[str, str] = {}

    def wanted(self, name: str) -> bool:
        """``name`` is selected; a name in a folder (``parity_systems/x``) is
        selected by its folder."""
        return self.only is None or name in self.only or name.split('/')[0] in self.only

    def figure(self, name: str, make: Callable[[], Optional[object]]) -> None:
        """Make and save one figure. ``make`` returns the matplotlib figure, or
        None (or raises ValueError) when the data cannot support it."""
        if not self.wanted(name):
            return
        import matplotlib.pyplot as plt

        from mllf.analysis import style as bs
        try:
            fig = make()
        except ValueError as e:
            self.skipped[name] = str(e)
            log.info(f"  {name}: skipped ({e})")
            return
        if fig is None:
            self.skipped[name] = 'no data'
            log.info(f"  {name}: skipped (no data)")
            return
        (self.figures / name).parent.mkdir(parents=True, exist_ok=True)
        bs.save(fig, str(self.figures / name), formats=(self.fmt,), gray_preview=self.grayscale)
        plt.close(fig)
        self.written.append(name)
        log.info(f"  {name}")

    def table(self, name: str, df) -> Path:
        path = self.tables / f'{name}.csv'
        df.to_csv(path, index=False)
        return path

    def report(self, text: str) -> Path:
        path = self.dir / 'report.txt'
        path.write_text(text)
        return path


def table_text(rows: List[Sequence], header: Sequence[str]) -> str:
    """A plain-text table with aligned columns (numbers right-aligned)."""
    cells = [[str(h) for h in header]] + [[_cell(v) for v in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    numeric = [all(_is_num(r[i]) for r in rows) if rows else False for i in range(len(header))]
    lines = []
    for k, r in enumerate(cells):
        parts = [c.rjust(w) if numeric[i] and k else c.ljust(w)
                 for i, (c, w) in enumerate(zip(r, widths))]
        lines.append('  ' + '  '.join(parts).rstrip())
        if k == 0:
            lines.append('  ' + '  '.join('-' * w for w in widths))
    return '\n'.join(lines)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) or v is None


def _cell(v) -> str:
    if v is None:
        return '-'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, float):
        if v != v:            # NaN
            return '-'
        return f'{v:.2f}' if abs(v) < 100 else f'{v:.0f}'
    return str(v)
