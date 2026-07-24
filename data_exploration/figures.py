"""
figures.py  (data_exploration)
==============================
Poster figures. Every figure reads the census / ``families.npz`` / ``summary.json``
caches -- none of them recompute a headline number, and none of them touch the
LMDB shards.

    python -m data_exploration.figures                # all
    python -m data_exploration.figures --only f1 f3   # a subset

Design rules applied throughout (see ``style.py``): nominal categories are
separated by position and direct labels rather than by hue; magnitude uses one
blue hue light->dark; polarity (log-lift) uses blue<->grey<->red; at most three
categorical series appear together, always direct-labelled; grid and axes are
solid hairlines behind the marks.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from ase.data import chemical_symbols
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from . import style as S
from .census import CACHE
from .stats import DERIVED, SUMMARY, load_all

S.use_poster_style()

_STATE = {}


def state():
    """Load (once) the census, labels, target and precomputed aggregates."""
    if not _STATE:
        c, names, counts, subset, y, fam_examples = load_all()
        with open(SUMMARY) as f:
            summary = json.load(f)
        with np.load(DERIVED) as d:
            derived = {k: d[k] for k in d.files}
        with open(os.path.join(CACHE, "families.json")) as f:
            fam = json.load(f)
        _STATE.update(c=c, names=names, counts=counts, subset=subset, y=y,
                      summary=summary, derived=derived, fam=fam)
    return _STATE


def subsets_by_size(summary):
    return list(summary["subsets"].keys())


def label(s):
    return S.SUBSET_LABEL.get(s, s)


def prov(n=None, extra=""):
    base = "OMol25 train_4M · full census, all 3,986,754 structures"
    if n is not None:
        base = f"OMol25 train_4M · {n:,} structures"
    return base + (f" · {extra}" if extra else "")


# --------------------------------------------------------------------------
# F1 -- periodic table of element coverage
# --------------------------------------------------------------------------

def _pt_layout():
    """(Z -> (row, col)) for the standard 18-column periodic table."""
    pos = {1: (1, 1), 2: (1, 18)}
    for z, col in zip(range(3, 11), [1, 2, 13, 14, 15, 16, 17, 18]):
        pos[z] = (2, col)
    for z, col in zip(range(11, 19), [1, 2, 13, 14, 15, 16, 17, 18]):
        pos[z] = (3, col)
    for i, z in enumerate(range(19, 37)):
        pos[z] = (4, i + 1)
    for i, z in enumerate(range(37, 55)):
        pos[z] = (5, i + 1)
    pos[55], pos[56] = (6, 1), (6, 2)
    for i, z in enumerate(range(57, 72)):       # lanthanides -> row 8
        pos[z] = (8, i + 3)
    for i, z in enumerate(range(72, 87)):
        pos[z] = (6, i + 4)
    pos[87], pos[88] = (7, 1), (7, 2)
    for i, z in enumerate(range(89, 104)):      # actinides -> row 9
        pos[z] = (9, i + 3)
    for i, z in enumerate(range(104, 119)):
        pos[z] = (7, i + 4)
    return pos


def f1_periodic_table():
    st = state()
    n_with = st["derived"]["n_struct_with_z"]
    N = int(st["summary"]["dataset"]["n_structures"])
    pos = _pt_layout()

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    vmax = np.log10(max(n_with.max(), 10))
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    n_present = 0
    for z, (r, col) in pos.items():
        if z >= len(n_with):
            continue
        cnt = int(n_with[z])
        x, yy = col - 0.5, -(r + (0.35 if r >= 8 else 0)) - 0.5
        if cnt == 0:
            ax.add_patch(plt.Rectangle((x, yy), 0.92, 0.92, facecolor=S.SURFACE,
                                       edgecolor=S.GRID, lw=0.8))
            ax.text(x + 0.46, yy + 0.46, chemical_symbols[z], ha="center",
                    va="center", fontsize=8, color="#cfcec8")
            continue
        n_present += 1
        v = norm(np.log10(cnt))
        face = S.SEQ(v)
        # ink flips to white once the tile is dark enough to swallow black text
        ink = "#ffffff" if v > 0.55 else S.INK
        ax.add_patch(plt.Rectangle((x, yy), 0.92, 0.92, facecolor=face,
                                   edgecolor=S.SURFACE, lw=1.4))
        ax.text(x + 0.46, yy + 0.60, chemical_symbols[z], ha="center", va="center",
                fontsize=10.5, color=ink, fontweight="bold")
        pct = 100.0 * cnt / N
        txt = f"{pct:.0f}%" if pct >= 1 else (f"{pct:.1f}%" if pct >= 0.1 else "<0.1%")
        ax.text(x + 0.46, yy + 0.24, txt, ha="center", va="center", fontsize=7,
                color=ink)

    ax.set_xlim(0.2, 19.1)
    ax.set_ylim(-10.6, -0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{n_present} elements appear in OMol25 train_4M — "
                 "main group, 3d/4d/5d metals, lanthanides and actinides",
                 loc="left", pad=16)
    sm = plt.cm.ScalarMappable(cmap=S.SEQ, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.62,
                      aspect=22, ticks=[0, 2, 4, 6, vmax])
    cb.set_label("structures containing the element", fontsize=10)
    cb.ax.set_yticklabels(["1", "100", "10k", "1M", f"{S.human(10 ** vmax)}"])
    cb.outline.set_visible(False)
    ax.text(0.2, -10.35, "tile label: element symbol and the % of all structures "
            "that contain it", fontsize=8.5, color=S.MUTED)
    return S.save(fig, "f1_periodic_table", prov())


# --------------------------------------------------------------------------
# F2 -- dataset anatomy: subset sizes + size distributions
# --------------------------------------------------------------------------

def _ridge(ax, values_by_row, bins, labels, color=S.BLUE, logx=False,
           height=1.1, annotate=None):
    """Stacked normalised histograms, one row per category.

    Row identity is carried by the y tick labels (never by hue), and the per-row
    annotation goes in the right margin -- both kept off the curves so nothing
    overlaps a mark.
    """
    for i, v in enumerate(values_by_row):
        h, _ = np.histogram(v, bins=bins)
        base = -i
        ax.axhline(base, color=S.AXIS, lw=0.6, zorder=0)
        if h.max() == 0:
            continue
        h = h / h.max() * height
        ctr = 0.5 * (bins[1:] + bins[:-1])
        ax.fill_between(ctr, base, base + h, color=color, alpha=0.18, lw=0)
        ax.plot(ctr, base + h, color=color, lw=1.5, solid_capstyle="round")
        if annotate is not None:
            ax.annotate(annotate[i], xy=(1.005, base + 0.06),
                        xycoords=("axes fraction", "data"), fontsize=9.5,
                        color=S.INK2, va="bottom", annotation_clip=False)
    # tick sits a little above its own baseline, inside that row's curve, so a
    # label is never read as belonging to the row below
    ax.set_yticks(-np.arange(len(labels)) + 0.3 * height, labels)
    ax.set_ylim(-len(labels) + 0.25, height + 0.35)
    if logx:
        ax.set_xscale("log")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, alpha=0.9)
    ax.set_axisbelow(True)


def f2_anatomy():
    st = state()
    su = subsets_by_size(st["summary"])
    counts = [st["summary"]["subsets"][s] for s in su]
    N = sum(counts)
    na = st["c"]["n_atoms"]
    sub = st["subset"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    ypos = np.arange(len(su))[::-1]
    ax.barh(ypos, counts, height=0.66, color=S.BLUE)
    for yv, cnt in zip(ypos, counts):
        ax.text(cnt + N * 0.012, yv, f"{S.human(cnt)}   {100 * cnt / N:.1f}%",
                va="center", fontsize=10, color=S.INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label(s) for s in su])
    ax.set_xlabel("structures")
    ax.set_xlim(0, max(counts) * 1.28)
    ax.set_title("What the 4M is made of", loc="left")
    S.clean(ax, grid="x")
    S.human_axis(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    bins = np.arange(0, 264, 4) - 0.5          # integer-aligned: no comb aliasing
    med = [f"median {int(np.median(na[sub == s]))} atoms" for s in su]
    _ridge(ax, [na[sub == s] for s in su], bins, [label(s) for s in su],
           annotate=med)
    ax.set_xlabel("atoms per structure")
    ax.set_title("System size by subset", loc="left")
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    return S.save(fig, "f2_dataset_anatomy",
                  prov(extra="right panel clipped at 260 atoms (p99 = "
                       f"{int(np.percentile(na, 99))}, max {int(na.max())})"))


# --------------------------------------------------------------------------
# F3 -- redundancy and the graph-only ceiling
# --------------------------------------------------------------------------

WL_GP_200K = 0.56          # measured: wl_gp2scale scaling ladder, 3 seeds
OLS_200K = 0.52            # linear baseline on the same embedding


def f3_ceiling():
    """How much of the target each *level of description* could ever explain.

    Each level is a strict refinement of the one above it, so the ceilings are
    ordered, and the molecular-graph level -- which needs the bond graph, i.e.
    Pass B -- is bracketed by the two levels we can compute from metadata alone.
    """
    st = state()
    fam = st["fam"]
    su = subsets_by_size(st["summary"])
    o = fam["overall"]

    levels = [
        ("chemical formula", o["formula"]),
        ("formula + charge + spin", o["formula_x_charge_spin"]),
        ("molecule identity\n(same molecule, any conformer)", o["family"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8),
                             gridspec_kw={"width_ratios": [1.12, 1.0]})

    ax = axes[0]
    ypos = np.array([3.0, 2.0, 0.0])
    vals = [d["r2_max_unbiased"] for _, d in levels]
    ax.barh(ypos, vals, height=0.62, color=S.BLUE)
    for yv, (_, d) in zip(ypos, levels):
        ax.text(d["r2_max_unbiased"] + 0.015, yv,
                f"{d['r2_max_unbiased']:.2f}", va="center", fontsize=11.5,
                color=S.INK)
        ax.text(d["r2_max_unbiased"] + 0.015, yv - 0.28,
                f"{S.human(d['n_groups'])} groups", va="center", fontsize=9,
                color=S.MUTED)
    lo = o["formula_x_charge_spin"]["r2_max_unbiased"]
    hi = o["family"]["r2_max_unbiased"]
    ax.barh([1.0], [hi - lo], left=[lo], height=0.62, color=S.BLUE, alpha=0.16,
            hatch="///", edgecolor=S.BLUE, lw=0)
    ax.text(lo + 0.012, 1.0, "molecular graph — not yet measured",
            va="center", fontsize=10, color=S.INK2)
    ax.axvline(WL_GP_200K, color=S.ORANGE, lw=2.2)
    ax.text(WL_GP_200K + 0.008, 3.62, "WL graph GP\nmeasured, 200k",
            color=S.ORANGE, fontsize=10, va="top")
    ax.set_yticks(ypos, [name for name, _ in levels])
    ax.set_ylim(-0.75, 4.05)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("best achievable held-out R²  on the intensive residual")
    ax.set_title("What each level of description can explain, at best", loc="left")
    S.clean(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    g = "formula_x_charge_spin"
    ceil = [max(fam["per_subset"][s][g]["r2_max_unbiased"], 0.0) for s in su]
    rest = [1.0 - v for v in ceil]
    ypos = np.arange(len(su))[::-1]
    ax.barh(ypos, ceil, height=0.66, color=S.BLUE)
    ax.barh(ypos, rest, left=np.array(ceil) + 0.006, height=0.66, color=S.ORANGE)
    for yv, v, s in zip(ypos, ceil, su):
        rm = fam["per_subset"][s][g]["rmse_within_unbiased"]
        ax.text(1.02, yv, f"{rm:5.1f} eV", va="center", fontsize=9.5,
                color=S.INK2, family="monospace")
        if v > 0.14:
            ax.text(v - 0.015, yv, f"{v:.2f}", va="center", ha="right",
                    fontsize=9.5, color="#ffffff")
        else:
            ax.text(v + 0.015, yv, f"{v:.2f}", va="center", ha="left",
                    fontsize=9.5, color=S.BLUE)
    ax.text(1.02, len(su) - 0.35, "irreducible\nRMSE", fontsize=9,
            color=S.MUTED, va="center")
    ax.text(0.02, len(su) - 0.35, "explained by composition", fontsize=9.5,
            color=S.BLUE, va="center")
    ax.text(0.62, len(su) - 0.35, "needs 3D structure", fontsize=9.5,
            color=S.ORANGE, va="center")
    ax.set_yticks(ypos, [label(s) for s in su])
    ax.set_xlim(0, 1.19)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("share of that subset's residual variance")
    ax.set_title("Where the unreachable variance lives", loc="left")
    S.clean(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return S.save(fig, "f3_description_ceiling",
                  prov(extra="ceilings are one-way random-effects variance "
                       "components (held-out, not in-sample); target = intensive "
                       "residual y = E − m(x), Var(y) = 50.5 eV²"))


def f3b_redundancy():
    """Provenance redundancy vs composition degeneracy -- two different things."""
    st = state()
    fam = st["fam"]
    with np.load(os.path.join(CACHE, "families.npz")) as d:
        ks, row_ccdf, cmp_row = d["ks"], d["row_ccdf"], d["cmp_row_ccdf"]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(ks, cmp_row, color=S.BLUE, lw=2.4)
    ax.plot(ks, row_ccdf, color=S.ORANGE, lw=2.4)
    ax.annotate("same formula, charge and spin", (ks[11], cmp_row[11]),
                color=S.BLUE, fontsize=10.5, xytext=(8, 14),
                textcoords="offset points")
    ax.annotate("same molecule (provenance family)", (ks[7], row_ccdf[7]),
                color=S.ORANGE, fontsize=10.5, xytext=(8, 16),
                textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("group size  k")
    ax.set_ylabel("share of structures in a group of size ≥ k")
    a, b = fam["overall"]["family"], fam["overall"]["formula_x_charge_spin"]
    ax.set_title("train_4M is de-duplicated by provenance but degenerate by "
                 "composition", loc="left")
    ax.text(0.98, 0.94,
            f"{S.human(a['n_groups'])} provenance families "
            f"(mean {a['mean_group_size']:.2f} structures, max {a['max_group_size']})\n"
            f"{S.human(b['n_groups'])} distinct (formula, charge, spin) "
            f"(mean {b['mean_group_size']:.1f}, max {b['max_group_size']:,})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=S.INK2)
    S.clean(ax, grid="both")
    fig.tight_layout()
    return S.save(fig, "f3b_redundancy", prov())


# --------------------------------------------------------------------------
# F5 -- charge x spin coverage
# --------------------------------------------------------------------------

def f5_charge_spin():
    st = state()
    q, s = st["c"]["charge"], st["c"]["spin"]
    qlo, qhi = -4, 4
    smax = 9
    H = np.zeros((smax, qhi - qlo + 1))
    m = (q >= qlo) & (q <= qhi) & (s >= 1) & (s <= smax)
    np.add.at(H, (s[m] - 1, q[m] - qlo), 1)
    N = len(q)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    norm = mcolors.LogNorm(vmin=1, vmax=max(H.max(), 10))
    im = ax.pcolormesh(np.arange(qhi - qlo + 2) - 0.5, np.arange(smax + 1) - 0.5,
                       np.where(H > 0, H, np.nan), cmap=S.SEQ, norm=norm,
                       edgecolors=S.SURFACE, linewidth=1.6)
    ax.set_xticks(range(qhi - qlo + 1))
    ax.set_xticklabels([f"{v:+d}" if v else "0" for v in range(qlo, qhi + 1)])
    ax.set_yticks(range(smax))
    ax.set_yticklabels(range(1, smax + 1))
    ax.set_xlabel("net charge")
    ax.set_ylabel("spin multiplicity")
    for i in range(smax):
        for j in range(qhi - qlo + 1):
            if H[i, j] > 0:
                dark = norm(H[i, j]) > 0.62      # ink flips on the colour, not the count
                ax.text(j, i, S.human(H[i, j]), ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if dark else S.INK)
    ions = 100 * (1 - st["summary"]["charge"]["frac_neutral"])
    open_shell = 100 * (1 - st["summary"]["spin"]["frac_singlet"])
    ax.set_title(f"Charge and spin coverage — {ions:.0f}% of structures are ions, "
                 f"{open_shell:.0f}% open-shell", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("structures")
    cb.outline.set_visible(False)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    cov = 100 * m.mean()
    return S.save(fig, "f5_charge_spin",
                  prov(extra=f"cells shown cover {cov:.1f}% of the dataset "
                       f"(|q| ≤ 4, multiplicity ≤ 9)"))


# --------------------------------------------------------------------------
# F6 -- element co-occurrence
# --------------------------------------------------------------------------

def f6_cooccurrence(top=28):
    st = state()
    cooc = st["derived"]["cooc"].astype(float)
    n_with = st["derived"]["n_struct_with_z"].astype(float)
    N = float(st["summary"]["dataset"]["n_structures"])
    zs = np.argsort(-n_with)[:top]
    zs = np.sort(zs)
    P = cooc[np.ix_(zs, zs)] / N
    p = n_with[zs] / N
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.log2(P / np.outer(p, p))
    np.fill_diagonal(lift, np.nan)
    lift[P == 0] = np.nan

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    v = np.nanpercentile(np.abs(lift), 98)
    cmap = S.DIV.copy()
    cmap.set_bad("#eceae4")            # never co-occur / self -- not "independent"
    im = ax.imshow(lift, cmap=cmap, vmin=-v, vmax=v)
    labs = [chemical_symbols[int(z)] for z in zs]
    ax.set_xticks(range(len(zs)), labs, fontsize=9)
    ax.set_yticks(range(len(zs)), labs, fontsize=9)
    ax.set_title("Which elements travel together (log₂ lift over independence)",
                 loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("log₂ lift   (0 = independent)")
    cb.outline.set_visible(False)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return S.save(fig, "f6_element_cooccurrence",
                  prov(extra=f"{top} most common elements; grey = never co-occur"))


# --------------------------------------------------------------------------
# F7 -- size and cost distributions
# --------------------------------------------------------------------------

def f7_sizes():
    st = state()
    c = st["c"]
    panels = [
        (c["n_atoms"], "atoms", 4),
        (c["n_heavy"], "heavy atoms (Z > 1)", 2),
        (c["n_elec"], "electrons", 16),
        (c["n_basis"], "basis functions (DFT cost proxy)", 64),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.6))
    for ax, (v, xl, step) in zip(axes.ravel(), panels):
        v = v.astype(float)
        hi = np.percentile(v, 99.9)
        bins = np.arange(0, hi + step, step) - 0.5   # integer-aligned
        ax.hist(v, bins=bins, color=S.BLUE, lw=0)
        med = np.median(v)
        ax.axvline(med, color=S.ORANGE, lw=2)
        ax.text(med + hi * 0.012, ax.get_ylim()[1] * 0.95,
                f"median {med:,.0f}", color=S.ORANGE, fontsize=10, va="top")
        ax.set_xlabel(xl)
        ax.set_ylabel("structures")
        ax.set_xlim(0, hi)
        S.clean(ax)
        S.human_axis(ax, "y")
    fig.suptitle("Size and cost distributions", x=0.005, ha="left", fontsize=13)
    fig.tight_layout()
    return S.save(fig, "f7_size_distributions",
                  prov(extra="x-axes clipped at the 99.9th percentile of each field"))


# --------------------------------------------------------------------------
# F8 -- composition diversity (Zipf + unique-formula growth)
# --------------------------------------------------------------------------

def f8_composition():
    st = state()
    formula = st["c"]["formula"]
    _, cnt = np.unique(formula, return_counts=True)
    cnt = np.sort(cnt)[::-1]
    N = len(formula)

    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    sizes = np.unique(np.logspace(2, np.log10(N), 26).astype(int))
    uniq = []
    seen = set()
    prev = 0
    for m in sizes:
        seen.update(formula[perm[prev:m]].tolist())
        uniq.append(len(seen))
        prev = m

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax = axes[0]
    ax.plot(np.arange(1, len(cnt) + 1), cnt, color=S.BLUE, lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("chemical formula, by rank")
    ax.set_ylabel("structures with that formula")
    ax.set_title(f"{S.human(len(cnt))} distinct formulas", loc="left")
    ax.text(0.98, 0.95,
            f"{100 * cnt[cnt > 1].sum() / N:.0f}% of structures share their formula\n"
            f"with at least one other\nlargest formula group {cnt[0]:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=S.INK2)
    S.clean(ax, grid="both")

    ax = axes[1]
    ax.plot(sizes, uniq, color=S.BLUE, lw=2.4, marker="o", ms=4)
    ax.plot(sizes, sizes, color=S.MUTED, lw=1.2)
    ax.text(sizes[-6], sizes[-6] * 1.35, "all distinct", color=S.MUTED, fontsize=9.5,
            rotation=32)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("structures drawn (random order)")
    ax.set_ylabel("distinct formulas seen")
    ax.set_title("Composition diversity keeps growing with sample size", loc="left")
    S.clean(ax, grid="both")
    fig.tight_layout()
    return S.save(fig, "f8_composition_diversity", prov())


# --------------------------------------------------------------------------
# F9 -- the regression target
# --------------------------------------------------------------------------

def f9_target():
    st = state()
    y, e = st["y"], st["c"]["energy"]
    if y is None:
        print("[fig] f9 needs families.npz; skipped")
        return None
    pred = e - y
    su = subsets_by_size(st["summary"])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    ax = axes[0]
    var_e = st["fam"]["target"]["total_energy_var"]
    var_y = st["fam"]["target"]["residual_var"]
    ax.hist(e / 1e3, bins=200, color=S.BLUE, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("total energy  E   [10³ eV]")
    ax.set_ylabel("structures")
    ax.set_title("Raw labels span five orders of magnitude", loc="left")
    ax.text(0.03, 0.95, f"std(E) = {np.sqrt(var_e) / 1e3:,.0f} × 10³ eV\n"
            f"min {e.min():,.0f} eV\nmax {e.max():,.0f} eV",
            transform=ax.transAxes, va="top", fontsize=10, color=S.INK2)
    S.clean(ax)
    S.human_axis(ax, "y")

    ax = axes[1]
    lim = np.percentile(np.abs(y), 99.5)
    bins = np.linspace(-lim, lim, 160)
    ax.hist(y, bins=bins, color=S.BLUE, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("intensive residual  y = E − m(x)   [eV]")
    ax.set_ylabel("structures")
    ax.set_title(f"After the extensive mean: {var_y / var_e:.0e} of the variance "
                 "is left", loc="left")
    ax.text(0.98, 0.95,
            f"std {np.std(y):.2f} eV\nvar {np.var(y):.1f} eV²\n"
            f"|y| > {lim:.0f} eV in 0.5% of rows",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=S.INK2)
    S.clean(ax)
    S.human_axis(ax, "y")
    fig.tight_layout()
    return S.save(fig, "f9_target_construction",
                  prov(extra="m(x) = ridge fit on [element counts, charge, |q|, q², "
                       "spin, n²], refit on all 4M rows; both panels log-count"))


# --------------------------------------------------------------------------
# F10 -- is the target intensive?
# --------------------------------------------------------------------------

def f10_residual_vs_size():
    st = state()
    y, na = st["y"], st["c"]["n_atoms"].astype(float)
    if y is None:
        return None
    lim = np.percentile(np.abs(y), 99)
    m = np.abs(y) < lim
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    hb = ax.hexbin(na[m], y[m], gridsize=(90, 60), bins="log", cmap=S.SEQ,
                   linewidths=0, extent=(0, 260, -lim, lim))
    edges = np.arange(0, 265, 5)
    idx = np.digitize(na, edges) - 1
    med = np.array([np.median(y[idx == i]) if np.any(idx == i) else np.nan
                    for i in range(len(edges) - 1)])
    sd = np.array([np.std(y[idx == i]) if np.any(idx == i) else np.nan
                   for i in range(len(edges) - 1)])
    ctr = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(ctr, med, color=S.ORANGE, lw=2.4)
    ax.plot(ctr, sd, color=S.AQUA, lw=2.4)
    ax.annotate("median", (ctr[24], med[24]), color=S.ORANGE, fontsize=11,
                xytext=(0, -26), textcoords="offset points", ha="center",
                fontweight="bold")
    ax.annotate("standard deviation", (ctr[24], sd[24]), color=S.AQUA, fontsize=11,
                xytext=(0, 12), textcoords="offset points", ha="center",
                fontweight="bold")
    ax.axhline(0, color=S.AXIS, lw=1)
    ax.set_xlabel("atoms per structure")
    ax.set_ylabel("intensive residual  y   [eV]")
    ax.set_title("The residual is centred, but its spread grows 5× with system size",
                 loc="left")
    cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)
    cb.set_label("structures")
    cb.outline.set_visible(False)
    S.clean(ax, grid="both")
    fig.tight_layout()
    return S.save(fig, "f10_residual_vs_size",
                  prov(extra="central 99% of |y|; both curves are per-5-atom-bin "
                       "statistics of the full data, not quantiles of the hexbin"))


# --------------------------------------------------------------------------
# F11 / F12 -- off-equilibrium forces and HOMO-LUMO gaps
# --------------------------------------------------------------------------

def _ridge_by_subset(field, bins, xlabel, title, name, logx, note_extra):
    st = state()
    su = subsets_by_size(st["summary"])
    v = st["c"][field].astype(float)
    sub = st["subset"]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    rows = [v[(sub == s) & np.isfinite(v)] for s in su]
    ann = [f"median {np.median(r):.2f}" if len(r) else "" for r in rows]
    _ridge(ax, rows, bins, [label(s) for s in su], logx=logx, annotate=ann)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    fig.tight_layout(rect=(0, 0, 0.83, 1))
    return S.save(fig, name, prov(extra=note_extra))


def f11_forces():
    return _ridge_by_subset(
        "fmax", np.logspace(-3, 1.7, 90), "max |F| per structure   [eV / Å]",
        "Most structures are far off equilibrium",
        "f11_force_distribution", True,
        "log axis; equilibrium geometries would sit below ~0.05 eV/Å")


def f12_gap():
    return _ridge_by_subset(
        "gap", np.linspace(0, 14, 100), "HOMO–LUMO gap   [eV]",
        "Electronic character by subset", "f12_homo_lumo_gap", False,
        "smaller of the two spin channels")


# --------------------------------------------------------------------------
# F18 -- which free metadata column predicts the target
# --------------------------------------------------------------------------

# columns computable from the input structure alone; everything else is a DFT
# OUTPUT and so is unavailable when predicting a new molecule
INPUT_SIDE = {"n_atoms", "n_heavy", "n_elec", "n_ecp_elec", "charge", "spin", "rg"}


def f18_metadata_signal():
    st = state()
    r2 = st["summary"].get("metadata_r2")
    if not r2:
        return None
    items = [(k, v) for k, v in r2.items()][:8][::-1]
    best_input = max((v for k, v in r2.items() if k in INPUT_SIDE), default=0.0)
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ypos = np.arange(len(items))
    ax.barh(ypos, [v for _, v in items], height=0.66, color=S.BLUE)
    for yv, (_, v) in zip(ypos, items):
        ax.text(v + 0.005, yv, f"{v:.3f}", va="center", fontsize=9.5, color=S.INK2)
    ax.set_yticks(ypos, [k for k, _ in items])
    ax.set_xlabel("univariate R² against the intensive residual y")
    ax.set_title("What the target is made of — and why it is a geometry problem",
                 loc="left")
    jr2 = st["summary"].get("metadata_joint_r2")
    ax.text(0.985, 0.30,
            f"every column of these 8 is a DFT OUTPUT, not an input feature.\n"
            f"the best column knowable from the structure alone\n"
            f"(atom counts, charge, spin, R_g) reaches only R² < 0.001"
            f"  [best {best_input:.5f}]"
            + (f"\nall columns jointly (linear): R² = {jr2:.2f}" if jr2 else ""),
            transform=ax.transAxes, ha="right", fontsize=10, color=S.INK2)
    S.clean(ax, grid="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return S.save(fig, "f18_metadata_signal",
                  prov(extra="the forces are labels, not inputs: their R² measures "
                       "how much of the residual is off-equilibrium strain — the part "
                       "a 3D descriptor can reach and a graph cannot"))


# --------------------------------------------------------------------------
# F19 -- data hygiene
# --------------------------------------------------------------------------

def f19_hygiene():
    st = state()
    su = subsets_by_size(st["summary"])
    h = st["summary"]["hygiene"]["per_subset"]
    panels = [
        ("frac_scf_ge_100", "SCF took ≥ 100 steps"),
        ("frac_s2dev_gt_0p1", "spin contamination  S² dev > 0.1"),
        ("frac_gap_lt_1eV", "HOMO–LUMO gap < 1 eV"),
        ("frac_has_nbo", "NBO charges present"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 5.0), sharey=True)
    ypos = np.arange(len(su))[::-1]
    for ax, (key, title) in zip(axes, panels):
        v = [h[s][key] for s in su]
        ax.barh(ypos, v, height=0.66, color=S.BLUE)
        # each panel is a different measure, so each gets its own scale; the
        # direct % labels are what the reader compares, not bar lengths
        hi = max(max(v), 0.02) * 1.45
        for yv, vi in zip(ypos, v):
            ax.text(max(vi, 0) + hi * 0.03, yv, f"{100 * vi:.1f}%", va="center",
                    fontsize=9, color=S.INK2)
        ax.set_xlim(0, hi)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_title(title, loc="left", fontsize=11)
        S.clean(ax, grid="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ypos, [label(s) for s in su])
    fig.suptitle("Data hygiene by subset", x=0.005, ha="left", fontsize=13)
    fig.tight_layout()
    ov = st["summary"]["hygiene"]
    return S.save(fig, "f19_hygiene",
                  prov(extra="every structure carries an ORCA warning and all the "
                       "frequent ones are boilerplate, so warnings are not a quality "
                       f"signal; only {ov['n_nan_lowdin']} structures in 4M have "
                       "unusable Löwdin charges (the loader's filter costs nothing)"))


FIGURES = {
    "f1": f1_periodic_table, "f2": f2_anatomy, "f3": f3_ceiling,
    "f3b": f3b_redundancy, "f5": f5_charge_spin, "f6": f6_cooccurrence, "f7": f7_sizes,
    "f8": f8_composition, "f9": f9_target, "f10": f10_residual_vs_size,
    "f11": f11_forces, "f12": f12_gap, "f18": f18_metadata_signal,
    "f19": f19_hygiene,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    for k, fn in FIGURES.items():
        if a.only and k not in a.only:
            continue
        fn()


if __name__ == "__main__":
    main()
