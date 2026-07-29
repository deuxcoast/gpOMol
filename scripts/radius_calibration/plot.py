"""One figure for the radius/calibration result: the two objectives move TOGETHER."""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

SCR = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(SCR, "arms")
Ks = [15, 30, 60, 200, 500]
base = {int(np.load(f)["seed"]): np.load(f)
        for f in sorted(glob.glob(os.path.join(ARMS, "base_s*.npz")))}
seeds = sorted(base)

r2, cv, rho_gp, rho_nn, rho_sc, rho_gpnn, lev = ({k: [] for k in Ks} for _ in range(7))
for K in Ks:
    for s in seeds:
        a = np.load(os.path.join(ARMS, f"arm_s{s}_k{K}.npz"))
        ae = np.abs(a["err"]); sg = np.sqrt(a["v_gp"])
        r2[K].append(r2_score(a["y"], a["mu"]))
        cv[K].append(sg.std() / sg.mean())
        rho_gp[K].append(spearmanr(sg, ae).statistic)
        rho_nn[K].append(spearmanr(base[s]["dnn"], ae).statistic)
        rho_sc[K].append(spearmanr(base[s]["sd_cheap"], ae).statistic)
        rho_gpnn[K].append(spearmanr(sg, base[s]["dnn"]).statistic)
        lev[K].append(np.median(sg) / np.sqrt(float(a["prior_var"])))
M = lambda d: np.array([np.mean(d[k]) for k in Ks])
E = lambda d: np.array([np.std(d[k], ddof=1) / np.sqrt(len(d[k])) for k in Ks])

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
x = np.arange(len(Ks))

a0 = ax[0]
a0.errorbar(x, M(rho_sc), E(rho_sc), marker="s", color="#444", label="size+category (no GP)")
a0.errorbar(x, M(rho_nn), E(rho_nn), marker="^", color="#c1121f", label="dist-to-NN (no GP)")
a0.errorbar(x, M(rho_gp), E(rho_gp), marker="o", color="#1d3557", label="GP $\\sigma^*$")
a0.set(xticks=x, xticklabels=Ks, xlabel="target neighbours $K$ (support radius)",
       ylabel=r"Spearman($\sigma$, |err|)", ylim=(0, 0.36),
       title="Ranking skill: the GP loses at EVERY radius")
a0.legend(fontsize=8, loc="center right"); a0.grid(alpha=.3)
a0.annotate("wider radius helps the GP\nbut never enough", (4, M(rho_gp)[4]),
            xytext=(2.3, 0.055), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=.8))

# The K arms are PAIRED within a seed (same embedding, same mean, same test points), and
# the between-seed spread in R^2 is ~10x the effect of the radius. Plotting raw R^2 with
# across-seed error bars would therefore hide a trend that is consistent in all 3 seeds.
# Show the paired change from the tightest arm instead, which is the quantity the design
# actually estimates.
a1 = ax[1]
dr2 = {k: np.array(r2[k]) - np.array(r2[Ks[0]]) for k in Ks}
a1.errorbar(x, M(dr2), E(dr2), marker="o", color="#1d3557",
            label="held-out $R^2$ (paired $\\Delta$ vs $K$=15)")
for i, s in enumerate(seeds):
    a1.plot(x, [dr2[k][i] for k in Ks], lw=.7, alpha=.45, color="#1d3557", ls=":")
a1.set(xticks=x, xticklabels=Ks, xlabel="target neighbours $K$",
       ylabel="$\\Delta R^2$ from $K$=15 (paired)",
       title="No tradeoff: accuracy and UQ move TOGETHER")
a1.grid(alpha=.3)
a1b = a1.twinx()
dcv = {k: np.array(cv[k]) - np.array(cv[Ks[0]]) for k in Ks}
a1b.errorbar(x, M(dcv), E(dcv), marker="D", color="#e07a5f",
             label=r"CV of $\sigma^*$ (paired $\Delta$)")
a1b.set_ylabel(r"$\Delta$ sd($\sigma^*$)/mean($\sigma^*$)", color="#e07a5f")
a1b.tick_params(axis="y", colors="#e07a5f")
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

a2 = ax[2]
a2.errorbar(x, M(lev), E(lev), marker="o", color="#1d3557",
            label=r"median $\sigma^*/\sqrt{\mathrm{prior}}$")
a2.axhline(1.0, color="#c1121f", ls="--", lw=1)
a2.text(0.05, 0.995, "prior variance (no information used)", fontsize=8,
        color="#c1121f", va="top")
a2.errorbar(x, M(rho_gpnn), E(rho_gpnn), marker="^", color="#2a9d8f",
            label=r"Spearman($\sigma^*$, dist-to-NN)")
a2.set(xticks=x, xticklabels=Ks, xlabel="target neighbours $K$", ylim=(0.5, 1.05),
       title="Why: a tight radius reverts $\\sigma^*$ to the prior")
a2.legend(fontsize=8, loc="lower left"); a2.grid(alpha=.3)

fig.suptitle("Support radius scored on CALIBRATION, not $R^2$ — 20k, 3 split seeds, "
             "800 held-out points/arm, additive kernel, jitter 4.30", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(SCR, "radius_calibration.png")
fig.savefig(out, dpi=150)
print("wrote", out)
