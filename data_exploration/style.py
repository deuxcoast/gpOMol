"""
style.py  (data_exploration)
============================
One visual system for every figure in this module, so the poster reads as a set.

Palette and rules follow the validated reference instance of the data-viz method
(light surface ``#fcfcfb``; categorical slots assigned in fixed order, never
cycled; one hue light->dark for magnitude; blue<->red with a neutral grey for
polarity). The three categorical slots used here were validated with
``validate_palette.js`` on the *all-pairs* list (worst CVD dE 9.2, worst
normal-vision dE 24.0, both clear) -- that is the reason multi-series plots in
this module are capped at three series and everything else is faceted or
single-hue.

Print target, so there is no dark mode and no hover layer: the outputs are a
300 dpi PNG (for layout) and a vector PDF (for the printer) of the same figure.

Two conventions worth stating because they are choices, not defaults:

* **Nominal categories get one hue.** The 10 ``data_id`` subsets are an axis, not
  a series; colouring them individually would double-encode bar length as hue.
  Subsets are separated by *position* (facets, ridge rows) and labelled directly.
* **Aqua (slot 3) sits below 3:1 on the light surface**, so any series using it
  carries a visible direct label rather than relying on the legend swatch.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")

# --- chrome & ink (light surface) -----------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- categorical slots (fixed order; only the first three are used) --------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE, ORANGE, AQUA = SERIES

# --- sequential (magnitude): one hue, light -> dark ------------------------
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)
# ordinal use (discrete ordered marks) must stay off the surface: start at 250
SEQ_ORDINAL = LinearSegmentedColormap.from_list("seq_blue_ord", SEQ_STEPS[1:])

# --- diverging (polarity): blue <-> neutral grey <-> red --------------------
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#e88a89", "#b02c2c"]
)

# --- status (reserved; never a series colour) ------------------------------
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}

# --- the 10 OMol25 subsets: fixed order (largest first) + print labels ------
SUBSET_ORDER = [
    "biomolecules", "metal_complexes", "reactivity", "elytes", "trans1x",
    "ani2x", "geom_orca6", "rgd", "orbnet_denali", "spice",
]
SUBSET_LABEL = {
    "biomolecules": "biomolecules",
    "metal_complexes": "metal complexes",
    "reactivity": "reactivity",
    "elytes": "electrolytes",
    "trans1x": "trans1x",
    "ani2x": "ani2x",
    "geom_orca6": "GEOM (orca6)",
    "rgd": "RGD",
    "orbnet_denali": "OrbNet Denali",
    "spice": "SPICE",
    "unknown": "unknown",
}


def use_poster_style():
    """Apply the shared rcParams. Idempotent; call once per figure module."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",          # never dashed
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,             # editable text in the vector output
        "ps.fonttype": 42,
    })


def clean(ax, grid="y", spines=("top", "right")):
    """Recessive chrome: hairline grid behind the marks, no boxing spines."""
    for s in spines:
        ax.spines[s].set_visible(False)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, alpha=0.9)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)
    return ax


def note(fig, text, y=-0.01):
    """One muted provenance line under a figure (N, sampling basis, source)."""
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8.5, color=MUTED)


def save(fig, name, subtitle=None):
    """Write ``figures/<name>.png`` (300 dpi) and ``figures/<name>.pdf``."""
    os.makedirs(FIGDIR, exist_ok=True)
    if subtitle:
        note(fig, subtitle)
    png = os.path.join(FIGDIR, f"{name}.png")
    fig.savefig(png)
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"[fig] wrote figures/{name}.png + .pdf")
    return png


def human_axis(ax, which="y"):
    """Format an axis in k / M units instead of scientific offset notation."""
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: human(v) if v else "0")
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(fmt)
    return ax


def human(n):
    """3,986,754 -> '3.99M'; used in titles and direct labels."""
    n = float(n)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            return f"{n / div:.3g}{suf}"
    return f"{n:.0f}"
