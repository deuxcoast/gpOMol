"""Render GP_R2 & OLS_R2 vs N from dim_sweep --out files (cache/sweep_{N}k.npz).

Auto-discovers every cache/sweep_*k.npz, computes the mean +/- std across seeds for
each N, and plots GP vs the OLS (linear-baseline) ceiling on a log-N axis.

    python plot_scaling.py [out.png]      # run from the dir containing cache/
"""
import glob
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "scaling.png"
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

fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.fill_between(Ns, gpm - gps, gpm + gps, color="#2a78d6", alpha=0.15, lw=0)
ax.fill_between(Ns, olm - ols, olm + ols, color="#898781", alpha=0.15, lw=0)
ax.plot(Ns, gpm, "o-", color="#2a78d6", lw=2.5, ms=7,
        label="GP (Wendland + linear prior mean)")
ax.plot(Ns, olm, "s--", color="#5f5e5a", lw=2.5, ms=6, label="OLS (linear baseline)")
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
