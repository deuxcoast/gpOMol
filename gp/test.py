from variogram_screen import gp_loo_crps, gram_loo_crps, print_table, run_screen
from wl_kernel import make_wl_candidate

cands = [
    make_wl_candidate("wl_h1_raw", h=1, normalize=False),
    make_wl_candidate("wl_h1_norm", h=1, normalize=True),
    make_wl_candidate("wl_h2_raw", h=2, normalize=False),
]
results = run_screen(cands, feats, z_referenced, sizes=atom_counts)
print_table(results)
