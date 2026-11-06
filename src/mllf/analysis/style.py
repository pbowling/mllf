"""
bowling_style.py -- publication figure style for Paige Bowling.

Usage
-----
    import bowling_style as bs
    bs.apply_style()                      # call AFTER any sns.set_theme()/sns.set()
    fig, ax = plt.subplots(figsize=bs.figsize("single"))
    for st, (x, y) in zip(bs.styles(len(data), "line"), data):
        ax.plot(x, y, **st)
    ...
    bs.save(fig, "fig2_transitions")      # writes fig2_transitions.pdf (+ .png)

Everything here is plain matplotlib rcParams plus a few helpers, so it works for
seaborn plots too. Text is saved as editable TrueType (pdf.fonttype 42) so labels
stay selectable in Illustrator/Inkscape.

Accessibility contract: every set of series styles returned by styles() is
distinguishable (1) under deuteranopia, protanopia and tritanopia and (2) when
printed in grayscale, either by lightness alone or by a redundant cue (marker
shape, line style or hatch). check_colors() verifies any custom set.
"""
from __future__ import annotations

import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from cycler import cycler

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
PALETTE = {
    "plum":     "#70307B",  # dark purple    L* 32  (lines, markers, text OK)
    "orange":   "#EB961E",  # orange         L* 69  (lines >= 1.8 pt, markers, bars)
    "forest":   "#104921",  # dark green     L* 27  (lines, markers, text OK)
    "lavender": "#E2CCEC",  # light purple   L* 85  (fills only; needs black edge)
    "apricot":  "#F5CA8E",  # light orange   L* 84  (fills only; needs black edge)
    "sage":     "#B0D5AF",  # light green    L* 82  (fills only; needs black edge)
    "mist":     "#F1F1F1",  # neutral        L* 95  (backgrounds, colormap midpoint)
}
# Extended shades (used by the ordered ramps; fine to use directly).
EXTENDED = {
    "plum_mid":    "#B98AC4",
    "plum_deep":   "#3F1846",
    "forest_mid":  "#5E9A63",
    "forest_deep": "#06280F",
    "orange_dark": "#A8620A",
    "orange_deep": "#5C3504",
}
INK = "black"
MUTED = "#BDBDBD"        # de-emphasised fills (highlight one item, gray the rest)
MUTED_LINE = "#A6A6A6"   # de-emphasised lines
BAND_GRAY = "#E3E3E3"    # parity bands / shaded regions (mist is too faint in print)
GRID_GRAY = "#7F7F7F"    # used with GRID_ALPHA for value-reading gridlines
GRID_ALPHA = 0.45

# Colours safe for thin lines / small markers / coloured text on white.
STRONG = [PALETTE[k] for k in ("plum", "orange", "forest")]
# Dark -> light partner of the same hue, for paired series (method vs. variant).
PAIRS = {"plum": PALETTE["lavender"], "orange": PALETTE["apricot"], "forest": PALETTE["sage"]}

# Redundant (non-colour) cues. Marker order goes from fewer to more vertices.
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", ":", "-."]
HATCHES = ["", "", "", "///", "...", "xxx"]

# Categorical colour orders.
#  lines:  plum, orange, forest. plum/forest are the same gray in print, so line
#          sets ALWAYS also vary marker (or line style); styles() does this.
#  fills:  forest, orange, lavender = dark / mid / light gray in print (dL* >= 15)
#          and well separated for all CVD types; items 4-6 add hatching.
LINE_COLORS = STRONG
FILL_COLORS = [PALETTE[k] for k in ("forest", "orange", "lavender", "plum", "apricot", "sage")]
CATEGORICAL = LINE_COLORS  # default colour cycle

# Ordered (sequential) ramps for ordered series: light partner -> dark anchor -> deeper.
RAMPS = {
    "purple": [PALETTE["lavender"], EXTENDED["plum_mid"], PALETTE["plum"], EXTENDED["plum_deep"]],
    "green":  [PALETTE["sage"], EXTENDED["forest_mid"], PALETTE["forest"], EXTENDED["forest_deep"]],
    "orange": [PALETTE["apricot"], PALETTE["orange"], EXTENDED["orange_dark"], EXTENDED["orange_deep"]],
}
MAX_ORDERED = 5  # beyond this, neighbouring steps blur; use a colormap + colorbar

# --------------------------------------------------------------------------- #
# Colormaps (registered as bowling_div, bowling_purple, bowling_orange,
# bowling_green and their _r reverses)
# --------------------------------------------------------------------------- #
def _register_cmaps():
    P, X = PALETTE, EXTENDED
    maps = {
        # diverging: plum (negative) <-> mist <-> forest (positive). Both ends are
        # dark in grayscale, so annotate cell values (annotate_cells) when the sign
        # must survive B&W printing.
        "bowling_div":    [P["plum"], P["lavender"], P["mist"], P["sage"], P["forest"]],
        # sequential: mist -> light partner -> anchor -> deep (monotonic lightness)
        "bowling_purple": [P["mist"], P["lavender"], X["plum_mid"], P["plum"], X["plum_deep"]],
        "bowling_green":  [P["mist"], P["sage"], X["forest_mid"], P["forest"], X["forest_deep"]],
        "bowling_orange": [P["mist"], P["apricot"], P["orange"], X["orange_dark"], X["orange_deep"]],
    }
    for name, cols in maps.items():
        cm = LinearSegmentedColormap.from_list(name, cols, N=256)
        for c in (cm, cm.reversed()):
            if c.name in mpl.colormaps:
                mpl.colormaps.unregister(c.name)
            mpl.colormaps.register(c)

_register_cmaps()

# --------------------------------------------------------------------------- #
# Colour science: lightness, CVD simulation, checker
# --------------------------------------------------------------------------- #
_CVD = {  # Machado et al. 2009, severity 1.0
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "protanopia":   np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "tritanopia":   np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}

def _lin(c):
    c = np.asarray(to_rgb(c))
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _lab(rgb_lin):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    X, Y, Z = M @ rgb_lin / np.array([0.95047, 1.0, 1.08883])
    f = lambda t: np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.array([116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))])

def lightness(c):
    """CIELAB L* (0 black - 100 white): the gray level a colour prints at."""
    return float(_lab(_lin(c))[0])

def contrast_vs_white(c):
    Y = _lin(c) @ np.array([0.2126, 0.7152, 0.0722])
    return 1.05 / (Y + 0.05)

def to_gray(c):
    """Grayscale equivalent (for previews of B&W printing)."""
    Y = _lin(c) @ np.array([0.2126, 0.7152, 0.0722])
    v = 1.055 * Y ** (1 / 2.4) - 0.055 if Y > 0.0031308 else 12.92 * Y
    return (float(v),) * 3

def check_colors(colors, min_dE=15.0, min_dL=15.0, redundant_cue=False, verbose=True):
    """Check a set of series colours for colour-vision deficiency and B&W print.

    Passes when every pair stays >= min_dE apart (CIELAB) under deuteranopia,
    protanopia and tritanopia, AND either every pair differs by >= min_dL in
    lightness (grayscale-safe on colour alone) or redundant_cue=True (the series
    also differ by marker / line style / hatch). Returns (ok, report)."""
    colors = list(colors)
    pairs = list(itertools.combinations(range(len(colors)), 2))
    if not pairs:
        return True, {}
    cvd = {}
    for name, M in _CVD.items():
        labs = [_lab(np.clip(M @ _lin(c), 0, 1)) for c in colors]
        cvd[name] = min(float(np.linalg.norm(labs[i] - labs[j])) for i, j in pairs)
    Ls = [lightness(c) for c in colors]
    dL = min(abs(Ls[i] - Ls[j]) for i, j in pairs)
    ok_cvd = all(v >= min_dE for v in cvd.values())
    ok_bw = dL >= min_dL or redundant_cue
    report = {"min_dE_cvd": {k: round(v, 1) for k, v in cvd.items()},
              "min_dL_gray": round(dL, 1), "cvd_ok": ok_cvd, "bw_ok": ok_bw}
    if verbose and not (ok_cvd and ok_bw):
        print("check_colors FAILED:", report)
    return ok_cvd and ok_bw, report

# --------------------------------------------------------------------------- #
# Series styles
# --------------------------------------------------------------------------- #
def ordered_colors(n, ramp="purple", kind="line", lo=None, hi=None):
    """n colours from light to dark along one hue ramp, for ORDERED series
    (thresholds, concentrations, cycles). Lightness is monotonic, so order
    survives grayscale and every CVD type. For kind='line' the light end starts
    where the colour reaches >= 2.3:1 contrast on white (orange's level), so no
    line is too pale; kind='fill' may go lighter (fills carry black edges)."""
    if n > MAX_ORDERED:
        print(f"ordered_colors: {n} > {MAX_ORDERED} steps blur together; "
              "consider a colormap + colorbar or splitting panels.")
    cm = LinearSegmentedColormap.from_list(ramp, RAMPS[ramp])
    if lo is None:
        lo = 0.12
        if kind != "fill":
            while contrast_vs_white(to_hex(cm(lo))) < 2.3 and lo < 0.6:
                lo += 0.01
    hi = 0.9 if hi is None else hi
    vals = [0.5 * (lo + hi)] if n == 1 else np.linspace(lo, hi, n)
    return [to_hex(cm(v)) for v in vals]

def styles(n, kind="line", ordered=False, ramp="purple", markers=True):
    """Keyword dicts for n series that stay distinguishable in colour, under
    colour-vision deficiency, and in grayscale print.

    kind='line'   -> for ax.plot / ax.errorbar: color + marker (or linestyle if
                     markers=False, e.g. dense curves).
    kind='marker' -> for ax.scatter: c + marker + black edge.
    kind='fill'   -> for ax.bar / fill_between / hist: color + hatch.
    ordered=True  -> colours from one ramp (light->dark); markers/linestyles
                     still vary so neighbouring steps never rely on shade alone.
    """
    if ordered:
        cols = ordered_colors(n, ramp, kind="fill" if kind == "fill" else "line")
    else:
        pool = FILL_COLORS if kind == "fill" else LINE_COLORS
        if n > len(pool) and kind == "fill":
            raise ValueError(f"{n} categories: more than {len(pool)} fills; group or split panels")
        cols = [pool[i % len(pool)] for i in range(n)]
    out = []
    for i, c in enumerate(cols):
        if kind == "line":
            d = {"color": c}
            if markers:
                d.update(marker=MARKERS[i % len(MARKERS)], mfc=c, mec=INK)
            else:
                d.update(ls=LINESTYLES[i % len(LINESTYLES)])
        elif kind == "marker":
            d = {"c": c, "marker": MARKERS[i % len(MARKERS)], "edgecolors": INK}
        elif kind == "fill":
            d = {"color": c, "edgecolor": INK}
            h = HATCHES[i % len(HATCHES)] if not ordered else ""
            if h:
                d["hatch"] = h
        else:
            raise ValueError("kind must be 'line', 'marker' or 'fill'")
        out.append(d)
    return out

def highlight(n, idx, kind="fill", color=None):
    """Styles for n items where only item(s) idx are coloured; the rest are gray."""
    idx = {idx} if isinstance(idx, int) else set(idx)
    color = color or PALETTE["plum"]
    out = []
    for i in range(n):
        c = color if i in idx else (MUTED if kind == "fill" else MUTED_LINE)
        out.append({"color": c, "edgecolor": INK} if kind == "fill" else {"color": c})
    return out

# --------------------------------------------------------------------------- #
# Sizes (ACS: single column 3.25 in, double column 7.0 in)
# --------------------------------------------------------------------------- #
WIDTHS = {"single": 3.25, "onehalf": 5.0, "double": 7.0}

def figsize(width="single", aspect=0.8, nrows=1):
    """(w, h) in inches. width = 'single' | 'onehalf' | 'double' | float inches.
    aspect = height/width of ONE row; nrows stacks rows."""
    w = WIDTHS.get(width, width)
    return (w, w * aspect * nrows)

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "FreeSans", "DejaVu Sans"]
CONTEXT_BASE = {"paper": 9, "poster": 18, "slides": 20}   # tick-label pt
CONTEXT_SCALE = {"paper": 1.0, "poster": 2.0, "slides": 2.0}  # line/marker widths

def apply_style(context="paper", base=None, ticks="out", mirror=False,
                bold_labels=True, rounded_legend=False, grid=False):
    """Set rcParams for the house style.

    context       'paper' (9 pt ticks at ACS column width), 'poster' or 'slides'
                  (scaled-up text and line weights). base overrides the tick size.
    ticks         'out' (2025-26 papers) or 'in' (Herbert-era papers).
    mirror        also draw ticks on top/right spines (Herbert-era look).
    bold_labels   bold axis labels (2026 papers) vs regular (Herbert-era).
    rounded_legend  rounded legend frame (Herbert-era) vs square.
    grid          value gridlines on every axes (normally use value_grid per axes).
    """
    base = base or CONTEXT_BASE[context]
    s = CONTEXT_SCALE[context]
    lw = 1.2 * s
    rc = {
        # fonts & text -------------------------------------------------------
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": base,
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.default": "it",
        "axes.unicode_minus": True,
        "text.color": INK,
        # axes box ----------------------------------------------------------
        "axes.edgecolor": INK,
        "axes.linewidth": lw,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.facecolor": "white",
        "axes.labelsize": base + 2,
        "axes.labelweight": "bold" if bold_labels else "normal",
        "axes.labelcolor": INK,
        "axes.labelpad": 3,
        "axes.titlesize": base + 2,
        "axes.titleweight": "bold",
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "axes.grid": grid,
        "axes.grid.axis": "y",
        "grid.color": GRID_GRAY,
        "grid.alpha": GRID_ALPHA,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7 * s,
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.05,
        # ticks -------------------------------------------------------------
        "xtick.direction": ticks,
        "ytick.direction": ticks,
        "xtick.top": mirror,
        "ytick.right": mirror,
        "xtick.major.width": lw,
        "ytick.major.width": lw,
        "xtick.major.size": 4 * s,
        "ytick.major.size": 4 * s,
        "xtick.minor.width": 0.8 * s,
        "ytick.minor.width": 0.8 * s,
        "xtick.minor.size": 2 * s,
        "ytick.minor.size": 2 * s,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "xtick.color": INK,
        "ytick.color": INK,
        # legend ------------------------------------------------------------
        "legend.frameon": True,
        "legend.edgecolor": INK,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "legend.fancybox": rounded_legend,
        "legend.fontsize": base + 1,
        "legend.title_fontsize": base + 1,
        "legend.borderpad": 0.5,
        "legend.labelspacing": 0.35,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.6,
        "legend.markerscale": 1.0,
        # marks -------------------------------------------------------------
        "lines.linewidth": 1.8 * s,     # never thinner for orange (2.4:1 contrast)
        "lines.markersize": 5.5 * s,
        "lines.markeredgewidth": 0.8 * s,
        "lines.markeredgecolor": INK,
        "patch.edgecolor": INK,
        "patch.force_edgecolor": True,
        "patch.linewidth": 0.8 * s,
        "hatch.color": INK,
        "hatch.linewidth": 0.6 * s,
        "errorbar.capsize": 2.5 * s,
        "scatter.edgecolors": INK,
        "boxplot.boxprops.color": INK,
        "boxplot.whiskerprops.color": INK,
        "boxplot.capprops.color": INK,
        "boxplot.medianprops.color": INK,
        "image.cmap": "bowling_orange",
        # output ------------------------------------------------------------
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,     # editable TrueType text in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none", # real <text> in SVG
    }
    mpl.rcParams.update(rc)
    _patch_legend_frame(lw)
    return rc

def _patch_legend_frame(lw):
    """The legend frame width has no rcParam; patch Legend so every legend gets it."""
    from matplotlib.legend import Legend
    Legend._bowling_lw = lw
    if getattr(Legend, "_bowling_patched", False):
        return
    orig = Legend.__init__
    def __init__(self, *a, **k):
        orig(self, *a, **k)
        self.get_frame().set_linewidth(Legend._bowling_lw)
    Legend.__init__ = __init__
    Legend._bowling_patched = True

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def panel_label(ax, label, title=None, style="upper", loc="outside"):
    """Panel letter. style='upper' -> 'A' ; style='paren' -> '(a)'.
    loc='outside' puts it above the top-left corner (2026 papers);
    loc='inside' puts '(a) title' inside the top-left of the axes (Herbert-era)."""
    txt = label.upper() if style == "upper" else f"({label.lower()})"
    if title:
        txt = f"{txt} {title}"
    size = mpl.rcParams["font.size"] + 3
    if loc == "outside":
        ax.annotate(txt, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-2.2 * size, 0.3 * size), textcoords="offset points",
                    ha="left", va="bottom", fontsize=size,
                    fontweight="bold" if style == "upper" else "normal",
                    annotation_clip=False)
    else:
        ax.text(0.03, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=size - 1)
    return ax

def label_panels(axes, style="upper", loc="outside", titles=None):
    """Label a flat iterable of axes A, B, C ... (or (a), (b) ...)."""
    import string
    axes = list(getattr(axes, "flat", axes))
    titles = titles or [None] * len(axes)
    for ax, letter, t in zip(axes, string.ascii_lowercase, titles):
        panel_label(ax, letter, title=t, style=style, loc=loc)

def direct_labels(ax, lines=None, labels=None, side="right", pad=6, min_gap=None):
    """Label each line at its end (instead of a legend), in the line's colour
    (black if that colour is too light to read). Labels are nudged apart so they
    never overlap. Call after limits are final (i.e. after plotting)."""
    lines = lines or [l for l in ax.lines if not l.get_label().startswith("_")]
    labels = labels or [l.get_label() for l in lines]
    fs = mpl.rcParams["legend.fontsize"]
    min_gap = min_gap or fs * 1.15  # points
    fig = ax.figure
    fig.canvas.draw()
    to_disp = ax.transData
    pts_per_px = 72 / fig.dpi
    items = []
    for line, lab in zip(lines, labels):
        x, y = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
        i = -1 if side == "right" else 0
        X, Y = to_disp.transform((x[i], y[i]))
        items.append([Y * pts_per_px, X, lab, line.get_color()])
    items.sort(key=lambda t: t[0])
    for k in range(1, len(items)):            # push apart upward
        if items[k][0] - items[k - 1][0] < min_gap:
            items[k][0] = items[k - 1][0] + min_gap
    for yp, X, lab, col in items:
        col = col if contrast_vs_white(col) >= 3 else INK
        xd, _ = to_disp.inverted().transform((X, 0))
        ax.annotate(lab, xy=(xd, 0), xycoords=("data", "figure points"),
                    xytext=(pad if side == "right" else -pad, yp), textcoords=("offset points", "figure points"),
                    ha="left" if side == "right" else "right", va="center",
                    color=col, fontsize=fs, fontweight="bold", annotation_clip=False,
                    in_layout=False)
    # Labels sit outside the axes and are excluded from constrained layout so
    # panel columns stay aligned; save() (bbox tight) still includes them. In a
    # multi-panel figure leave a gap: fig.get_layout_engine().set(wspace=0.1).

def zero_line(ax, axis="y", **kw):
    """Black reference line at 0, drawn behind the data (above gridlines).
    save() adds this automatically to any axes whose y-range crosses 0."""
    kw = {"color": INK, "lw": 0.9, "zorder": 0.8, **kw}
    line = (ax.axhline if axis == "y" else ax.axvline)(0, **kw)
    line._bowling_zero = True
    return line

def value_grid(ax, axis="y", **kw):
    """Low-opacity gray dashed gridlines at the major ticks, for figures where
    the reader needs to read values off the plot. Drawn behind the data."""
    style = dict(color=GRID_GRAY, alpha=GRID_ALPHA, ls="--", lw=0.7)
    style.update(kw)
    if axis in ("y", "both"):
        ax.yaxis.grid(True, **style)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, **style)
    ax.set_axisbelow(True)

hgrid = value_grid  # backwards-compatible name

def ratio_axis(ax, axis="y", lo=0.25, hi=4.0, ref=True):
    """Log2 axis for ratios / fold changes / speedups: symmetric about 1, ticks at
    powers of two labelled as ratios (1/4, 1/2, 1, 2, 4), reference line at 1."""
    from fractions import Fraction
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    setscale = ax.set_yscale if axis == "y" else ax.set_xscale
    setlim = ax.set_ylim if axis == "y" else ax.set_xlim
    a = ax.yaxis if axis == "y" else ax.xaxis
    setscale("log", base=2)
    setlim(lo, hi)
    ticks = [2.0 ** k for k in range(int(np.floor(np.log2(lo))), int(np.ceil(np.log2(hi))) + 1)]
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(
        lambda v, _: str(Fraction(v).limit_denominator(1024)) if v < 1 else f"{v:g}"))
    if ref:
        (ax.axhline if axis == "y" else ax.axvline)(1, color=INK, lw=0.9, zorder=0.8)
    ax._bowling_no_zero = True

def parity(ax, x=None, y=None, lims=None, step=None, band=1.0,
           band_color=BAND_GRAY):
    """Parity / calc-vs-expt plot frame: square axes with IDENTICAL limits and
    ticks on x and y, both ending on a tick, plus a dashed y = x line and a
    shaded ±band (e.g. ±1 kcal/mol; band=None to omit).

    Pass the plotted data (x, y; lists of arrays are fine) so the shared range
    encloses everything, or give lims=(lo, hi) directly. step forces the tick
    spacing; otherwise ~5-7 ticks are chosen. Call after or before plotting."""
    from matplotlib.ticker import MaxNLocator, MultipleLocator
    if lims is None:
        vals = []
        for v in (x, y):
            if v is None:
                continue
            arrs = v if isinstance(v, (list, tuple)) and np.ndim(v[0]) > 0 else [v]
            vals.extend(np.ravel(np.asarray(a, float)) for a in arrs)
        if not vals:
            raise ValueError("parity() needs x/y data or lims=(lo, hi)")
        vals = np.concatenate(vals)
        lo, hi = np.nanmin(vals), np.nanmax(vals)
    else:
        lo, hi = lims
    if step is None:
        ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
        step = np.diff(ticks).min()
    lo = np.floor(lo / step + 1e-9) * step
    hi = np.ceil(hi / step - 1e-9) * step
    xs = np.array([lo, hi])
    if band:
        ax.fill_between(xs, xs - band, xs + band, color=band_color, lw=0, zorder=0)
    ax.plot(xs, xs, ls="--", color=INK, lw=1.2, zorder=1, marker="none")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(step))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax._bowling_no_snap = True   # limits are already on shared ticks
    for line in ax.lines:        # points on the boundary are not cut in half
        line.set_clip_on(False)
    return lo, hi, step

def stat_text(ax, lines, loc="lower right", size=None):
    """Bold, colour-matched statistics (e.g. MUE per series) stacked in a corner.
    lines = [(text, color), ...]; text may use mathtext, e.g. r'MUE$_{\\rm WT}$ = 1.04'.
    Light colours are swapped for black so the text stays readable."""
    size = size or mpl.rcParams["font.size"] + 2
    right = "right" in loc
    top = "upper" in loc
    x = 0.96 if right else 0.04
    y0 = 0.95 if top else 0.05
    step = -1 if top else 1
    ordered = lines if top else list(reversed(lines))
    for i, (t, c) in enumerate(ordered):
        c = c if contrast_vs_white(c) >= 2.3 else INK
        ax.text(x, y0 + step * i * 0.085, t, transform=ax.transAxes, color=c,
                fontweight="bold", fontsize=size,
                ha="right" if right else "left", va="top" if top else "bottom")

def annotate_cells(ax, M, fmt="{:+.1f}", fontsize=None):
    """Write each heatmap value in its cell (black or white text by background),
    so signs/magnitudes survive grayscale printing. Use for <= ~60 cells."""
    M = np.asarray(M)
    im = ax.images[-1]
    fs = fontsize or mpl.rcParams["font.size"] - 1
    for (i, j), v in np.ndenumerate(M):
        bg = im.cmap(im.norm(v))
        col = "white" if lightness(bg) < 50 else INK
        ax.text(j, i, fmt.format(v).replace("-", "\u2212"), ha="center", va="center", color=col, fontsize=fs)

def shared_label(fig, axes, text, axis="y"):
    """One axis label spanning a column/row of stacked panels."""
    axes = list(getattr(axes, "flat", axes))
    for ax in axes:
        (ax.set_ylabel if axis == "y" else ax.set_xlabel)("")
    if axis == "y":
        fig.supylabel(text, fontsize=mpl.rcParams["axes.labelsize"],
                      fontweight=mpl.rcParams["axes.labelweight"])
    else:
        fig.supxlabel(text, fontsize=mpl.rcParams["axes.labelsize"],
                      fontweight=mpl.rcParams["axes.labelweight"])

def colorbar(fig, mappable, ax, label=None, minmax=False, **kw):
    """Black-framed colorbar; minmax=True replaces ticks with 'Min'/'Max'."""
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.outline.set_edgecolor(INK)
    cb.outline.set_linewidth(mpl.rcParams["axes.linewidth"])
    if label:
        cb.set_label(label, fontweight=mpl.rcParams["axes.labelweight"],
                     fontsize=mpl.rcParams["axes.labelsize"])
    if minmax:
        lo, hi = mappable.get_clim()
        cb.set_ticks([lo, hi])
        cb.set_ticklabels(["Min", "Max"])
    return cb

def sns_kwargs(kind):
    """Keyword defaults that make seaborn match the house style.
    Example: sns.violinplot(..., **bs.sns_kwargs('violin'))"""
    k = {
        "violin": dict(linecolor=INK, linewidth=0.8, inner="box", saturation=1),
        "box": dict(linecolor=INK, linewidth=0.8, saturation=1,
                    flierprops=dict(marker="o", markersize=3, markeredgecolor=INK)),
        "bar": dict(edgecolor=INK, linewidth=0.8, saturation=1, err_kws=dict(color=INK, linewidth=1)),
        "hist": dict(edgecolor=INK, linewidth=0.6),
        "scatter": dict(edgecolor=INK, linewidth=0.6),
        "scatter_dense": dict(s=8, edgecolor=INK, linewidth=0.3, alpha=0.7),
        "heatmap": dict(linewidths=0.5, linecolor="white"),
    }
    return dict(k[kind])

def jitter(values, width=0.15, seed=0):
    """Gentle uniform jitter for overlapping discrete values."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    return v + rng.uniform(-width, width, v.shape)

# --------------------------------------------------------------------------- #
# Finalize & save
# --------------------------------------------------------------------------- #
def _is_data_axes(ax):
    return not (hasattr(ax, "_colorbar") or ax.images or not ax.get_visible())

def _choose_tick_limits(ax, name, lo, hi, max_intervals=7):
    """Pick (lo, hi, step) with lo/hi on multiples of a 'nice' step that enclose
    the data, with <= max_intervals intervals and tick labels that fit along the
    axis without crowding. Prefers the tightest fit; non-negative data never
    gets a negative lower limit."""
    from matplotlib.ticker import ScalarFormatter
    span = hi - lo
    mag = 10.0 ** np.floor(np.log10(span / max_intervals))
    cands = sorted({m * mag * f for m in (1, 2, 2.5, 5) for f in (1, 10, 100)})
    fig = ax.figure
    bbox = ax.get_window_extent()
    axis_len_pt = (bbox.width if name == "x" else bbox.height) * 72 / fig.dpi
    fs = mpl.rcParams["xtick.labelsize" if name == "x" else "ytick.labelsize"]
    fs = fs if isinstance(fs, (int, float)) else mpl.rcParams["font.size"]
    best = None
    for st in cands:
        new_lo = np.floor(lo / st + 1e-9) * st
        new_hi = np.ceil(hi / st - 1e-9) * st
        if lo >= 0 and new_lo < 0:
            new_lo = 0.0
        n = int(round((new_hi - new_lo) / st))
        if n < 2 or n > max_intervals:
            continue
        ticks = new_lo + st * np.arange(n + 1)
        dec = max(0, -int(np.floor(np.log10(st) + 1e-9))) + (1 if round(st / 10 ** np.floor(np.log10(st)), 6) == 2.5 else 0)
        labels = [f"{t:.{dec}f}" for t in ticks]
        if name == "x":   # labels side by side: total width must leave gaps
            need = sum(len(l) for l in labels) * 0.55 * fs + n * 0.8 * fs
        else:             # stacked labels: need ~1.6 line heights per tick
            need = (n + 1) * 1.6 * fs
        if need > axis_len_pt:
            continue
        waste = (new_hi - new_lo) - span
        key = (round(waste / span, 2), abs(n - 5))  # tightest, then ~5 intervals
        if best is None or key < best[0]:
            best = (key, (new_lo, new_hi, st))
    return None if best is None else best[1]

def snap_to_ticks(ax, axis="both"):
    """Make autoscaled numeric axes start and end on a major tick that just
    encloses the data. Skips axes whose limits were set by hand, categorical /
    fixed-tick axes, and log axes. Non-negative data never gets a negative
    range. Markers on the boundary are left unclipped."""
    from matplotlib.ticker import FixedLocator, MultipleLocator
    snapped = False
    for name in (("x", "y") if axis == "both" else (axis,)):
        a = ax.xaxis if name == "x" else ax.yaxis
        auto = ax.get_autoscalex_on() if name == "x" else ax.get_autoscaley_on()
        scale = ax.get_xscale() if name == "x" else ax.get_yscale()
        loc = a.get_major_locator()
        categorical = a.get_converter() is not None if hasattr(a, "get_converter") else a.converter is not None
        if not auto or scale != "linear" or isinstance(loc, FixedLocator) or categorical:
            continue
        lo, hi = (ax.dataLim.intervalx if name == "x" else ax.dataLim.intervaly)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            continue
        setter = ax.set_xlim if name == "x" else ax.set_ylim
        lim = _choose_tick_limits(ax, name, lo, hi)
        if lim is None:
            continue
        new_lo, new_hi, st = lim
        setter(new_lo, new_hi)
        # Pin the step: auto locators re-choose it from the final axes size,
        # which can leave the ends off-tick.
        a.set_major_locator(MultipleLocator(st))
        snapped = True
    if snapped:
        for line in ax.lines:
            if line.get_marker() not in (None, "", "None", "none"):
                line.set_clip_on(False)
        for coll in ax.collections:
            if coll.__class__.__name__ == "PathCollection":  # scatter
                coll.set_clip_on(False)

def finalize(fig, zero=True, snap=True):
    """Apply figure-level conventions just before saving:
    1. snap autoscaled numeric axes to start/end on a major tick;
    2. add a black y = 0 line behind the data on every axes whose y-range crosses 0.
    Opt out per axes with ax._bowling_no_snap / ax._bowling_no_zero = True."""
    for ax in fig.axes:
        if not _is_data_axes(ax):
            continue
        if snap and not getattr(ax, "_bowling_no_snap", False):
            snap_to_ticks(ax)
        if zero and not getattr(ax, "_bowling_no_zero", False):
            lo, hi = ax.get_ylim()
            has_zero = any(getattr(l, "_bowling_zero", False) for l in ax.lines)
            from matplotlib.ticker import FixedLocator
            ya = ax.yaxis
            cat_y = isinstance(ya.get_major_locator(), FixedLocator) or (
                (ya.get_converter() if hasattr(ya, "get_converter") else ya.converter) is not None)
            if (ax.get_yscale() == "linear" and not cat_y and min(lo, hi) < 0 < max(lo, hi)
                    and not has_zero):
                zero_line(ax)
                ax.set_ylim(lo, hi)

def save(fig, stem, formats=("pdf", "png"), zero=True, snap=True, gray_preview=False):
    """finalize() then save next to the script: stem.pdf (vector, editable text)
    + stem.png preview. gray_preview=True also writes stem_gray.png, a grayscale
    rendering for checking B&W print legibility."""
    finalize(fig, zero=zero, snap=snap)
    out = []
    for ext in formats:
        fig.savefig(f"{stem}.{ext}")
        out.append(f"{stem}.{ext}")
    if gray_preview:
        from PIL import Image
        fig.savefig(f"{stem}_gray.png", dpi=200)
        Image.open(f"{stem}_gray.png").convert("L").save(f"{stem}_gray.png")
        out.append(f"{stem}_gray.png")
    return out
