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

---

# Part II — Pass B: the structural sample

Two samples, both scanned with the production bond perception
(`build_graph(atoms, cutoff_mult=1.2)`), by `geom_sample.py`:

* **stratified** — 199,999 structures, proportional-with-floor across the 10
  subsets. The representative sample: geometry, charges, WL vocabulary.
* **clusters** — 200,017 structures drawn as *whole* (formula, charge, spin)
  groups, 44,054 groups in all. Within-group variance is only estimable from
  groups with members in hand, and a uniform 5% draw shatters them.

## 10. What a structure actually looks like

`figures/f14_fragments_compactness.png`. **32% of `train_4M` is not a molecule —
it is a cluster.** Per subset: biomolecules 82% multi-fragment (median 2),
electrolytes 78% (median 5, max 43), reactivity 29%, SPICE 24%, metal complexes
20%, GEOM 0%. For a third of the dataset the energy therefore contains an
intermolecular term, and a bond-graph descriptor sees only a disjoint union — it
has no representation of how the pieces are arranged relative to each other.

Everything is compact: `R_g = 1.08 n^(1/3)` fits the whole size range, so there is
no extended/chain-like population to model separately.

`figures/f13_bond_geometry.png` and `figures/f15_charges.png` give the raw material
of the geometry and elec channels. Bond-length modes come out at textbook values
(H–C 1.09, C–C 1.39, C–N 1.45, C–O 1.43, H–N 1.01, H–O 0.97, C–S 1.81 Å) — but see
§11 for the secondary peaks past 1.9 Å. Löwdin charges do **not** track
electronegativity: mean charge is +0.09 e on H, −0.19 on C, −0.06 on O, +0.03 on N,
+0.71 on S, +0.55 on Cl. Anyone reading the elec channel chemically should know
that; the median within-structure charge span is 0.85 e and its distribution is
cleanly bimodal (neutral organics near 0.6 e, ionic systems near 1.4 e).

## 11. The perceived "molecular graph" is a geometry fingerprint

This is the main result of Pass B, and it was not on the plan.

The ceiling measurement (§6, `descriptors.py ceiling`) grouped structures by their
WL multiset hash — the exact equivalence class of the shipped descriptor — and
returned a within-group variance of **0.99 eV²**, implying a ceiling of 0.99. That
number is wrong, and the way it is wrong is informative. A **consistency check**
catches it: same molecule ⇒ same bond graph ⇒ same WL labels, so the WL grouping
must be *coarser* than "same molecule" and its within-variance must be *larger*.
Measured on the same rows:

| grouping | within-group variance | dof |
|---|---|---|
| provenance family (same molecule) | **8.74 eV²** | 12,219 |
| WL depth-3 hash | **0.99 eV²** | 20,070 |

The ordering is inverted, which is impossible for a graph invariant. **87% of
multi-member provenance families — same molecule, different conformer — receive
different WL hashes.** Bond perception thresholds interatomic distances, so moving
the atoms changes the perceived bond set, and the "graph" descriptor is partly
fingerprinting geometry.

`figures/f20_graph_is_not_a_graph.png` sweeps the constant responsible. Across
2,613 same-molecule families:

| multiplier | families split | bonds/atom | median rings | acyclic |
|---|---|---|---|---|
| 1.00 | **37.0%** | 0.99 | 1 | 34% |
| 1.10 | 62.4% | 1.03 | 2 | 29% |
| **1.20 (production)** | **87.9%** | 1.19 | 7 | 11% |
| 1.30 | 90.0% | 1.49 | 17 | 2% |

No setting reaches zero, and the production value is close to the worst. The
mechanism is visible directly in `figures/f13_bond_geometry.png`: at 1.2× the
C–C distribution grows a second peak at **2.4 Å** and N–N's *mode* is at **2.29 Å**
— non-bonded contacts admitted as bonds — and **41% of carbon atoms are assigned
more than four bonds** (44% four-coordinate, 33% five, 8% six).

Three consequences:

1. **The [0.53, 0.92] bracket on the graph ceiling stands.** It cannot be closed
   with this data, because no geometry-independent molecular graph exists in the
   shards (no SMILES, no bond orders). The honest statement is the bracket plus the
   knowledge that the WL channel is *not* confined by its upper end, since it sees
   geometry through perception.
2. **It explains the WL channel's instability.** A hash that changes with
   conformation is close to unique per structure, which is exactly what §12
   measures — and an embedding built on near-unique columns is what produces the
   0.076 across-seed spread in the wl-only GP at 20k.
3. **It is cheap to test.** `cutoff_mult` is one constant shared by
   `wl_features.build_graph`, the geometry channels and the strain reference.
   Sweeping it at production N is a small experiment with a large potential payoff,
   and 1.1 is the value to try first: it halves the splitting and cuts the ring
   count from 7 to 2 while barely moving bonds/atom (1.19 → 1.03).

## 12. A frozen WL vocabulary never converges

`figures/f16_wl_vocabulary.png`, on 60,000 structures of the stratified sample.

The vocabulary grows as **N^0.91** — near-linearly, i.e. roughly one new column per
new molecule. 1,355,778 distinct labels appear in 60k structures; 171,528 survive
the production `min_count ≥ 2` prune; extrapolating gives ~7.8M columns at 4M. The
growth is concentrated in the deep labels: depth 1 has 41,447 distinct labels
(14,481 kept), depth 2 has 426,672 (66,142), depth 3 has 887,659 (90,905 kept —
only 10% of depth-3 labels are ever seen twice).

The consequence is the out-of-vocabulary rate, and it improves far too slowly to
outrun:

| vocabulary frozen on | kept columns | labels of a fresh structure that are OOV | structures fully covered |
|---|---|---|---|
| 2,852 | 10,788 | 48.5% | 3.2% |
| 9,647 | 32,309 | 42.1% | 5.6% |
| 24,058 | 74,632 | 37.7% | 8.7% |
| 44,244 | 130,322 | **35.0%** | **10.7%** |

That is ~11 percentage points per decade of training data, so extrapolating to a
vocabulary frozen on all 4M still leaves **~13% of a fresh structure's labels
unrepresented**. Production silently drops OOV labels at transform time, so a third
of every test structure's descriptor is currently being discarded at 200k.

Combined with §11 the diagnosis is coherent: WL labels are near-unique because
perception makes them conformation-dependent, the vocabulary therefore grows with
the data instead of saturating, and the frozen-vocabulary assumption the pipeline
rests on never becomes true. **Capping the depth at 1–2 is the obvious first
experiment** — depth 1 alone has a 41k-label vocabulary of which 35% survive
min_count ≥ 2, versus 10% at depth 3.

## 13. There is no single correlation length scale

`figures/f4_variogram.png`, on 20,000 structures with the production WL and
geometry featurisers reduced to PLS-10.

**No natural cutoff exists.** The pairwise-distance distribution is unimodal in
both embeddings with no gap — there is no "near vs far" structure for compact
support to snap to. The pooled correlation range (first lag reaching 95% of the
sill) is 0.89 in WL against a median pair distance of 0.69, and 4.88 in geometry
against a median of 4.02: correlation persists out to *beyond the typical pair
separation*. The compact-support radius is therefore a compute budget, not a
property of the data — which is what the existing percentile-based cutoff already
assumes, and this confirms it is the right way to think about it.

**The range is not one number.** Normalised by each subset's own median pair
distance, the WL range spans 0.40 (biomolecules) to 1.47 (electrolytes) — a 3.7×
spread — and geometry spans 0.62 to 1.85. A single global length scale is
misspecified across subsets by roughly a factor of four.

**Nor is the noise floor.** The nugget/sill ratio — the share of a subset's target
variance the embedding cannot resolve at any distance — varies from ~0 to nearly 1:

| subset | WL nugget/sill | geometry nugget/sill |
|---|---|---|
| RGD | **0.95** | 0.00 |
| ani2x | 0.26 | 0.19 |
| biomolecules | 0.25 | 0.09 |
| electrolytes | 0.12 | 0.18 |
| metal complexes | 0.09 | 0.19 |
| GEOM (orca6) | 0.00 | 0.00 |

RGD is the extreme case: its target is essentially invisible to the WL embedding
(nugget ≈ sill) and fully visible to geometry. This is an *a priori*, per-category
noise floor — exactly the quantity a per-category noise term in the kernel would
need, available without fitting a GP.

## 14. The block-sparse kernel is discarding real information

`figures/f17_cross_category.png`. The production kernel zeroes every cross-`data_id`
block. Two measurements say that is not free:

| embedding | 10-NN links that cross a boundary | rows with ≥1 cross neighbour | R² of y from same-category neighbour mean | from cross-category |
|---|---|---|---|---|
| WL | **55.3%** | 88.9% | 0.754 | **0.682** |
| geometry | 40.6% | 77.5% | 0.654 | 0.441 |

In the WL embedding more than half of all nearest-neighbour links cross a category
boundary, and those neighbours carry **90% as much information about `y`** as
same-category ones. The neighbour matrix shows where: RGD's neighbours are only 49%
RGD (14% biomolecules, 16% trans1x), reactivity 43%, ani2x 41%, SPICE 33% — the
small subsets are largely embedded *inside* the large ones. Metal complexes (88%
self) and GEOM (76%) are the only genuinely separable categories.

Caveat: these R² values are neighbour-mean predictions computed over the whole 20k
sample, so both are optimistic in absolute terms; the same/cross *comparison* is the
meaningful quantity and both arms carry the same bias. Some of the same-category
advantage is also confounded — `data_id` correlates with composition, and
composition is 53% of the target.

The practical reading: block sparsity is a **compute** decision, and it is currently
being paid for with signal. Worth testing a relaxed version (allow the k nearest
cross-category neighbours, or block only the pairs that are genuinely far) against
the current all-or-nothing structure.

## 15. Next

The two experiments this pass argues for, both in `wl_gp2scale`, both cheap, both
with paired per-seed differences at production N:

1. **Sweep `cutoff_mult`** (1.0 / 1.1 / 1.2). It halves same-molecule graph
   splitting and cuts spurious bonds; it is one constant shared by the WL, geometry
   and strain channels.
2. **Cap the WL depth at 1–2.** Depth 3 contributes 65% of the vocabulary and only
   10% of its labels are ever seen twice.

And two structural questions this pass has now made concrete: whether the
cross-category blocks should be relaxed (§14), and whether the kernel should carry
per-category signal variance and noise (§13, §3).
