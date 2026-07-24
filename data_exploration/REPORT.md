# What is in OMol25 `train_4M`

Every number below is measured on **all 3,986,754 structures** (not a sample), by
`data_exploration/census.py`, and is reproducible from `cache/summary.json` /
`cache/families.json`. Figures are in `figures/`.

---

## 1. Shape of the dataset

| | |
|---|---|
| structures | **3,986,754** in 80 `.aselmdb` shards, 21 GB |
| atoms | 218,680,957 |
| distinct elements | **83** |
| `data_id` subsets | 10 — four of ~800k each make up 80% of the data |
| upstream sources | 34 (`source` path root), in 44 production batches |
| system size | median **42 atoms** (22 heavy), p99 207, max 350 |
| basis functions | median 1,032, max 9,243 |
| charge | **43.9% ions**, from −10 to +10 |
| spin | **17.1% open-shell**, multiplicity up to 11; 51.6% run unrestricted |

Subsets (`figures/f2_dataset_anatomy.png`): biomolecules 800.0k, electrolytes
799.8k, metal complexes 797.1k, reactivity 789.8k, ani2x 215.6k, trans1x 213.7k,
GEOM/orca6 199.8k, RGD 75.3k, OrbNet Denali 51.1k, SPICE 44.4k.

Median size varies 5× across subsets — trans1x/ani2x/reactivity sit at 14 atoms,
electrolytes at 77 and biomolecules at 66 with a tail to 350. Anything tuned on one
subset is tuned on one size regime.

## 2. Chemical diversity

`figures/f1_periodic_table.png` — 83 elements, and the coverage is genuinely broad
rather than organic-only:

- Organic core: H 99%, C 97%, O 81%, N 77% of structures.
- Common heteroatoms: S 31%, P 16%, F 14%, Cl 12%, Br 7%, I 5%, B/Si 5%.
- **Every transition metal** appears at 0.2–1% (Ti, V, Ir, Pd, Pt, Ru … ), plus
  the full lanthanide row at ~0.1–0.2%.
- Absent: actinides, Fr, Ra, Po, At, Rn, and the superheavies.

`figures/f6_element_cooccurrence.png` — the co-occurrence structure is exactly what
the chemistry predicts and is worth knowing before designing an element-pair
descriptor: **transition metals strongly exclude one another** (one metal centre per
complex, log₂ lift ≈ −5), the metalloid/heavy main-group set (Ge, As, Se, Sb, Te)
co-occurs above chance, and Li anti-correlates with everything metallic (it lives in
the electrolyte subset, not in complexes).

Composition space (`figures/f8_composition_diversity.png`): **1,127,158 distinct
chemical formulas**. Drawing structures in random order, the count of distinct
formulas grows as N^0.94 up to ~100k and still N^0.72 over the last decade to 4M —
sublinear but nowhere near saturated, so the dataset has not run out of new
compositions at 4M. At the same time 82.8% of structures share their formula with
at least one other, and the largest single formula group has 9,907 members.

## 3. The regression target

`figures/f9_target_construction.png`. Total energies span five orders of magnitude
(std 120 × 10³ eV). The element-referencing extensive mean
`m(x) = ridge[1, element counts, q, |q|, q², spin, n²]`, refit here on all 4M rows,
removes all but **3.5 × 10⁻⁹** of that variance. What is left is the intensive
residual the GP actually regresses:

- **Var(y) = 50.53 eV², std 7.11 eV**, mean 0 by construction, heavy right tail
  (p99 = +24.6 eV, max +249 eV).
- Per-subset variance spans **38×**: trans1x 3.5, RGD 4.0, SPICE 6.7, reactivity
  7.1, ani2x 7.7, OrbNet 10.4, biomolecules 14.3, metal complexes 38.9,
  GEOM 127.6, electrolytes 133.3 eV².

Two properties of `y` that the current model does not account for:

**It is not homoscedastic in system size** (`figures/f10_residual_vs_size.png`).
The residual stays centred at ≈ −2 eV across the whole size range, but its standard
deviation grows from ~3 eV at 10 atoms to ~17 eV at 250 atoms. A single global noise
term and a single signal variance are misspecified with respect to size.

**Nothing knowable at prediction time predicts it linearly**
(`figures/f18_metadata_signal.png`). Of every census column, the best *input-side*
one (atom counts, charge, spin, radius of gyration) reaches R² = 0.00002. The
extensive mean has already absorbed everything trivially available; all remaining
signal has to come from the descriptor. That is good news for the setup — there is
no free lunch left on the table.

What *does* predict it, and why it matters: **max|F| alone gives R² = 0.397** and
the DFT-output columns jointly give 0.468. Forces are labels, not features, so this
is not a usable model — it is a *diagnostic*. It says ~40% of the intensive residual
is off-equilibrium strain energy, which is a property of the 3D geometry that a bond
graph cannot see.

## 4. How far off equilibrium

`figures/f11_force_distribution.png`. This is emphatically **not** a dataset of
relaxed minima: median max|F| = **3.4 eV/Å**, p95 = 10.2 eV/Å. Per subset the
character differs sharply — trans1x (reaction paths) is bimodal with a mode near
0.5 eV/Å and a strained tail past 20 eV/Å, GEOM has a distinct high-force lobe near
25 eV/Å, metal complexes are the closest to equilibrium (median 1.6). 110 structures
sit exactly at 50 eV/Å, an upstream clipping boundary.

`figures/f12_homo_lumo_gap.png`. Gaps are wide and well-behaved (median 7.9 eV);
only 0.2% of structures fall below 1 eV, so near-degenerate electronic structure is
not a significant hazard in this split.

## 5. Redundancy: how many distinct molecules?

`figures/f3b_redundancy.png`. Two kinds of "duplicate" have to be separated, and
they give opposite answers:

**By provenance, `train_4M` is already de-duplicated.** Parsing `source` into
molecule families gives **3.63M families for 3.99M structures** — mean 1.10
structures per family, only **12.2%** of rows share a family with anything, largest
family 174. 26.7% of rows come from a job with a `stepK` suffix, yet the mean is
1.03 frames per job: the upstream 4M subsample kept roughly one frame per
trajectory. **So the near-duplicate rows that made the 20k kernel near-singular are
not provenance duplicates** — they are separate molecules that happen to be close in
descriptor space, which is a descriptor property, not a dataset property.

**By composition it is strongly degenerate.** There are 1,244,665 distinct
(formula, charge, spin) triples, mean 3.2 structures each, largest 9,835, and
**80.3% of rows share their triple with another row**.

## 6. The ceiling on a graph-only descriptor

`figures/f3_description_ceiling.png` — the headline result. A descriptor that cannot
distinguish two structures must predict the same value for both, so its best
possible held-out R² is fixed by the within-group variance of whatever it *can*
distinguish. Using the unbiased variance component (`SS_within/(n−g)`, not the
inflated in-sample split):

| level of description | groups | best held-out R² |
|---|---|---|
| chemical formula | 1.13M | **0.52** |
| formula + charge + spin | 1.24M | **0.53** |
| *molecular graph* (needs Pass B) | — | *between 0.53 and 0.92* |
| molecule identity (any conformer) | 3.63M | **0.92** |

Read against the measured ladder in [wl_gp2scale](../wl_gp2scale): the WL graph GP
reaches **0.56 at 200k**, i.e. it is already *above* the composition-lookup ceiling
of 0.53 — WL is extracting real graph information beyond stoichiometry — but the
same-molecule ceiling is 0.92, so **at least 0.36 of R² is sitting in structure the
graph has not been asked to see.** That headroom is conformational and geometric,
and it is the quantitative case for the geometry channel.

**Where that headroom lives** (right panel; share of each subset's own variance that
composition can reach, and the irreducible RMSE if it cannot):

| subset | composition ceiling | irreducible RMSE |
|---|---|---|
| SPICE | 0.78 | 1.2 eV |
| metal complexes | 0.77 | 3.0 eV |
| reactivity | 0.64 | 1.6 eV |
| biomolecules | 0.61 | 2.4 eV |
| OrbNet Denali | 0.54 | 2.2 eV |
| ani2x | 0.47 | 2.0 eV |
| trans1x | 0.27 | 1.6 eV |
| RGD | 0.26 | 1.7 eV |
| electrolytes | 0.18 | 10.5 eV |
| GEOM (orca6) | 0.04 | 11.1 eV |

The split is not subtle: for metal complexes and SPICE, *what the molecule is* is
almost the whole story; for GEOM, electrolytes, RGD and trans1x, composition is
nearly uninformative and the energy is dominated by *which conformer / which
configuration*. Those four subsets are where a 3D channel earns its keep, and two
of them (electrolytes, GEOM) also carry the largest absolute variance in the whole
dataset.

## 7. Data hygiene

`figures/f19_hygiene.png`. There is no meaningful quality problem in this split:

- **Warnings are useless as a signal** — every structure carries at least one, and
  all the frequent strings are boilerplate (functional/dispersion notices). The one
  informative warning ("system is open-shell, switching RHF→UHF", 672,699 rows)
  simply mirrors the open-shell fraction.
- SCF ≥ 100 steps: 2.7% overall, 8.0% for metal complexes.
- Spin contamination (⟨S²⟩ deviation > 0.1): 5.7% overall — reactivity 13.6%,
  metal complexes 11.9%, zero for the closed-shell organic subsets.
- HOMO–LUMO gap < 1 eV: 0.2%.
- NBO charges are present in only 67.2% of rows, and the gaps are subset-correlated
  (biomolecules 52.5%, electrolytes 41.3%, metal complexes 44.8%, everything else
  ≥ 98%). **Löwdin charges are present essentially everywhere** — only **88
  structures in 4M** have missing/NaN Löwdin charges. Any charge channel should use
  Löwdin, not NBO.
- The loader's NaN-Löwdin filter (`wl_gp2scale/data.py`) therefore costs 88 rows
  and introduces no subset bias. **Open question 6 in `PLAN.md` is answered: it is
  not biasing the subsets.**

---

## 8. What this changes for descriptor design

1. **The geometry channel is justified quantitatively, not just by intuition.**
   Composition tops out at 0.53; molecule identity reaches 0.92. WL at 0.56 has
   captured the composition-level signal and a little more. The rest is 3D.
   Independently, max|F| — pure strain — explains 0.40 on its own.

2. **Target the geometry channel at the subsets where it pays.** GEOM (0.04),
   electrolytes (0.18), RGD (0.26) and trans1x (0.27) have almost no
   composition-level signal and, for the first two, the largest variance in the
   dataset. A geometry channel that helps everywhere equally is not what the data
   asks for.

3. **One global signal variance / length scale is wrong.** Per-subset residual
   variance spans 38× (3.5 → 133 eV²) and per-subset ceilings span 0.04 → 0.78.
   The block-sparse kernel already partitions by `data_id`; it should carry
   **per-category signal variance and noise**, which is a small change to
   `wl_gp2scale/kernel.py` and directly supported by these numbers.

4. **The residual is heteroscedastic in system size** (3 → 17 eV std). Either model
   noise as size-dependent or reconsider the intensivity convention.

5. **Random splits are optimistic, and it is worth quantifying by how much.** 80% of
   structures share their (formula, charge, spin) with another structure, so a
   uniform random train/test split routinely puts two conformers of the same
   molecule on opposite sides. The published ladder numbers (0.34 → 0.56) are
   measured in that setting. A **formula-disjoint split** would measure
   generalisation to unseen molecules, and the gap between the two is a real
   quantity worth reporting on the poster.

6. **Provenance de-duplication is unnecessary.** Only 12% of rows share a family, so
   uniform sampling is fine; the effective N is not meaningfully below the nominal N.
   The conditioning problem seen at 20k is descriptor-space crowding, not duplicated
   data.

7. **Use Löwdin charges, and treat NBO as unavailable** for the three big subsets.

## 9. Next (Pass B, `PLAN.md` steps 4–5)

The one number this pass could not produce is the **molecular-graph ceiling**
itself, bracketed here at [0.53, 0.92]. It needs bond graphs: group by WL hash on a
stratified sample and repeat §6's variance decomposition. That single measurement
tells you how much of the 0.36 headroom WL could in principle capture with more
data, and how much is conformational and needs geometry — which is precisely the
question the additive-kernel work is trying to answer empirically at 200k.
