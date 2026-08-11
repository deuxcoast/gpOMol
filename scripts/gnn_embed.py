"""Extract per-molecule embeddings from a pretrained eSEN GNN, for the nugget screen.

WHY. Our own diagnostics say the DESCRIPTOR is the bottleneck: gamma/sill is ~0.57 in
the nearest distance bin (57% of the energy variance is already present between the two
closest molecules we can find), and that floor is representation-limited rather than
compression-limited -- quadrupling the PLS rank moved it 0.023 and changing the metric
made it worse. A GNN trained on energies produces a latent space where nearby points
have similar energies by construction, which is exactly the property we lack. This
extracts that latent space so it can go through the same screen.

THIS IS AN UPPER BOUND, NOT A RESULT. `esen-sm-direct-all-omol` was trained on ALL of
OMol25, which includes our test molecules. Its embedding has seen the answers. That is
fine for a go/no-go decision -- if even a leaky, in-domain, purpose-trained
representation cannot beat 0.57, no learned representation will -- but the number can
never be reported. A clean version needs a model trained on a disjoint split.

TWO THINGS THAT ARE EASY TO GET WRONG.

  * ROTATION INVARIANCE. The backbone emits (n_atoms, 9, 128): the 9 is the SO(3) irrep
    stack for lmax=2 (1 scalar + 3 vector + 5 tensor components). Only the l=0 slice is
    invariant; the rest rotate with the molecule. Pooling the whole thing would give a
    descriptor whose distances change when you rotate the input, which is silently wrong
    rather than loudly wrong. We take [:, 0, :].

  * POOLING. Energy is EXTENSIVE, so the energy head effectively sums per-atom
    contributions -- sum-pooling is closest to what the model computes. But our target is
    the INTENSIVE residual after an extensive mean has already been removed, so
    mean-pooling may match it better. Both are emitted; the screen decides.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--model", default="esen-sm-direct-all-omol")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N (timing)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from wl_gp2scale.data import get_data

    ds = get_data(src="train_4M", n=a.n, seed=0)
    atoms = ds.atoms if not a.limit else ds.atoms[:a.limit]
    print(f"[gnn] {len(atoms):,} molecules, model={a.model}, device={a.device}")

    unit = pretrained_mlip.get_predict_unit(a.model, device=a.device)
    calc = FAIRChemCalculator(unit, task_name="omol")
    tgt = dict(unit.model.named_modules())["module.backbone.norm"]

    grab = {}
    tgt.register_forward_hook(lambda m, i, o: grab.__setitem__("h", o))

    mean_p, sum_p, t0 = [], [], time.time()
    for k, at in enumerate(atoms):
        b = at.copy()
        b.info.update(charge=at.info.get("charge", 0), spin=at.info.get("spin", 1))
        b.calc = calc
        b.get_potential_energy()
        # [:, 0, :] is the l=0 (scalar) irrep -- the only rotation-invariant slice
        h = grab["h"][:, 0, :].detach().to(torch.float64).numpy()
        mean_p.append(h.mean(axis=0)); sum_p.append(h.sum(axis=0))
        if (k + 1) % 500 == 0:
            el = time.time() - t0
            print(f"[gnn]   {k + 1:,}/{len(atoms):,}  {el / 60:.1f} min  "
                  f"{1000 * el / (k + 1):.0f} ms/mol  "
                  f"ETA {(len(atoms) - k - 1) * el / (k + 1) / 60:.1f} min", flush=True)

    M, S = np.array(mean_p), np.array(sum_p)
    print(f"[gnn] done in {(time.time() - t0) / 60:.1f} min; embedding dim {M.shape[1]}")
    out = a.out or f"cache/gnn_{a.model.replace('-', '_')}_{a.n}.npz"
    np.savez(out, mean_pool=M, sum_pool=S, model=a.model, n=a.n)
    print(f"[gnn] -> {out}")


if __name__ == "__main__":
    main()
