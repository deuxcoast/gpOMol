"""Render GP_R2 & OLS_R2 vs N from dim_sweep --out files (cache/sweep_{N}k.npz).

Auto-discovers every cache/sweep_*k.npz, computes the mean +/- std across seeds for
each N, and plots GP vs the OLS (linear-baseline) ceiling on a log-N axis.

    python plot_scaling.py [out.png]           # screen/paper figure (unchanged default)
    python plot_scaling.py --poster [stem]     # POSTER scale -> stem.svg + stem.png

POSTER MODE. Sized and weighted to be read from ~3 m on a 42x36 in poster, where the
figure occupies roughly 30 cm. Everything that differs from the default is a legibility
decision, not a taste one:

  * type at 24-34 pt and lines at 5-6 px, because poster body text below ~24 pt is
    unreadable at standing distance and a 2.5 px line disappears entirely;
  * the +/- std across seeds stays a filled band rather than error bars -- bars read as
    more precise but they clutter a figure whose message is the GAP between two curves;
    pass --no-title if the layout tool is setting the headline instead;
  * transparent background, so the figure sits on whatever panel colour the poster uses
    rather than punching a white rectangle into it;
  * both endpoint R^2 values called out large, since "0.34 -> 0.56" IS the result and a
    reader should get it without consulting the axis;
  * SVG *and* 300 dpi PNG. SVG stays sharp at any print size, but Canva only accepts SVG
    uploads on Pro -- the PNG is the fallback that always works.
"""
import glob
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

argv = [a for a in sys.argv[1:]]
POSTER = "--poster" in argv
NO_TITLE = "--no-title" in argv
argv = [a for a in argv if not a.startswith("--")]
out = argv[0] if argv else ("scaling_poster" if POSTER else "scaling.png")
rec = {}
for f in sorted(glob.glob("cache/sweep_*k.npz")):
    m = re.search(r"sweep_(\d+)k\.npz", f)
    if not m:
        continue
    N = int(m.group(1)) * 1000
    rows = np.load(f)["rows"]           # columns: seed, dim, cutoff, med_nbr, ols, gp
    rec[N] = (rows[:, 4], rows[:, 5])   # (ols across seeds, gp across seeds)

if not rec:
    sys.exit("no cache/sweep_*k.npz files found (run from the dir containing cache/)")

Ns = sorted(rec)
olm = np.array([rec[N][0].mean() for N in Ns]); ols = np.array([rec[N][0].std() for N in Ns])
gpm = np.array([rec[N][1].mean() for N in Ns]); gps = np.array([rec[N][1].std() for N in Ns])

GP_C, OLS_C = "#2a78d6", "#5f5e5a"

if not POSTER:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.fill_between(Ns, gpm - gps, gpm + gps, color=GP_C, alpha=0.15, lw=0)
    ax.fill_between(Ns, olm - ols, olm + ols, color="#898781", alpha=0.15, lw=0)
    ax.plot(Ns, gpm, "o-", color=GP_C, lw=2.5, ms=7,
            label="GP (Wendland + linear prior mean)")
    ax.plot(Ns, olm, "s--", color=OLS_C, lw=2.5, ms=6, label="OLS (linear baseline)")
    ax.set_xscale("log")
    ax.set_xticks(Ns); ax.set_xticklabels([f"{N // 1000}k" for N in Ns])
    ax.set_xlabel("training molecules  N")
    ax.set_ylabel("held-out R²  (intensive residual)")
    ax.set_title("Graph-only GP scales and beats the linear baseline (OMol25)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    for N, g in zip(Ns, gpm):
        ax.annotate(f"{g:.2f}", (N, g), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="#185fa5")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    raise SystemExit

# ------------------------------------------------------------------ poster rendering
plt.rcParams.update({"font.size": 24, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(13, 9))

ax.fill_between(Ns, gpm - gps, gpm + gps, color=GP_C, alpha=0.15, lw=0)
ax.fill_between(Ns, olm - ols, olm + ols, color="#898781", alpha=0.15, lw=0)
ax.plot(Ns, gpm, "o-", color=GP_C, lw=6.5, ms=22, mew=0,
        label="GP (Wendland + linear prior mean)")
ax.plot(Ns, olm, "s--", color=OLS_C, lw=5, ms=17, mew=0,
        label="OLS (linear baseline)")

ax.set_xscale("log")
ax.set_xticks(Ns)
ax.set_xticklabels([f"{N // 1000}k" for N in Ns], fontsize=30)
ax.tick_params(axis="y", labelsize=28, length=8, width=2)
ax.tick_params(axis="x", length=8, width=2)
ax.set_xlabel("training molecules   N", fontsize=32, labelpad=16)
ax.set_ylabel("held-out  R²", fontsize=32, labelpad=16)
if not NO_TITLE:
    ax.set_title("Graph-only GP scales and beats the linear baseline",
                 fontsize=36, pad=26)

# The two endpoints ARE the finding; a poster reader should not have to trace the axis.
for N, g, dy, fs in ((Ns[0], gpm[0], 30, 40), (Ns[-1], gpm[-1], 24, 46)):
    ax.annotate(f"{g:.2f}", (N, g), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=fs, color="#185fa5", fontweight="bold")

# headroom only at the top, so the 0.56 callout clears the axes without clipping
lo = min((olm - ols).min(), (gpm - gps).min())
hi = max((olm + ols).max(), (gpm + gps).max())
ax.set_ylim(lo - 0.015, hi + 0.045)

ax.grid(alpha=0.28, lw=1.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(2)
leg = ax.legend(frameon=False, loc="lower right", fontsize=25,
                handlelength=2.6, borderaxespad=1.2)
# colour each label from its OWN handle -- a hard-coded (GP, OLS) tuple silently swaps
# the moment the plot calls are reordered, which is exactly what happened once already
for t, h in zip(leg.get_texts(), leg.legend_handles):
    t.set_color(h.get_color())

fig.tight_layout()
stem = out[:-4] if out.lower().endswith((".svg", ".png")) else out
for ext, kw in ((".svg", {}), (".png", {"dpi": 300})):
    fig.savefig(stem + ext, transparent=True, bbox_inches="tight", **kw)
    print(f"wrote {stem + ext}")
print("  transparent background; no embedded title (use --title to add one).")
print("  Canva: SVG needs Pro; the PNG works on the free tier.")
