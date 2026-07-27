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

## 15. The `cutoff_mult` sweep — run, and the prescription was half wrong

Experiment 1 of §16 has been run: `dim_sweep --bond-mult {1.0, 1.1, 1.2} --n 20000
--channels wl`, **6 seeds** (1, 2, 3, 7, 42, 123), all arms on identical frozen
splits so every comparison is paired. The 1.2 arm reproduces the recorded baseline
exactly (0.4172 / 0.3425 / 0.2334 on seeds 42/7/123), which confirms the arms differ
only in the constant under test.

| `cutoff_mult` | GP R² mean | GP R² **std** | OLS R² | raw WL labels | kept (min_count ≥ 2) | test OOV |
|---|---|---|---|---|---|---|
| 1.00 | 0.2752 | 0.0455 | 0.2960 | 230k | 37.4k | 10.5% |
| **1.10** | **0.3618** | **0.0347** | **0.3580** | 287k | 40.0k | 13.1% |
| 1.20 (default) | 0.3158 | 0.0734 | 0.2998 | 460k | 54.9k | 21.0% |

Paired per-seed differences against the 1.2 default:

| arm | mean Δ | std | t | seeds improved |
|---|---|---|---|---|
| 1.00 − 1.20 | **−0.0406** | 0.0483 | −2.06 | 1 / 6 |
| 1.10 − 1.20 | **+0.0460** | 0.0738 | +1.53 | 5 / 6 |

**The effect is not monotone, so "tighter is more chemical, therefore better" — the
reasoning in §11 — is wrong.** 1.0 is the *worst* arm (loses on 5 of 6 seeds,
t = −2.06): pulling the threshold all the way in drops real long bonds (metal–ligand,
hypervalent main group) along with the spurious contacts. There is an interior
optimum, and at this N it is near 1.1.

**What 1.1 actually buys is stability, not headroom.** The across-seed std falls
0.073 → 0.035, and the gain is concentrated exactly where 1.2 fails: the two worst
1.2 seeds (0.233, 0.238) gain **+0.131 and +0.137**, while the two best barely move
(+0.003, +0.019). 1.1 does not raise the ceiling; it removes the failure mode. That
is precisely the mechanism §11 predicted — unstable perception occasionally yields a
bad vocabulary — even though the direction of the fix was not.

Two supporting observations: OLS on the embedding tracks GP across all three arms
(0.296 / 0.358 / 0.300), so this is the **descriptor** changing rather than the
kernel; and 1.1 cuts the raw vocabulary 38% and test OOV from 21% to 13%, which is
the §12 problem improving at the same time.

**Statistical honesty.** At n = 6 the mean improvement is *not* significant by a
paired t-test (t = 1.53, p ≈ 0.19) — the per-seed differences are large and
skewed because the effect lives in the failing seeds. What is solid is the sign
(5/6), the halved variance, and the vocabulary/OOV reduction. Treat +0.046 as
suggestive and the stability gain as established; more seeds would settle the mean
cheaply (~1.5 min/seed at 20k on CPU).

**Recommendation:** switch the default to **1.1** on the stability and vocabulary
evidence, and re-check at 200k before the full ladder — the optimum may move with N,
since a larger training set estimates rare labels better and may tolerate the wider
vocabulary that 1.2 produces.

Reproduce with `python -m data_exploration.bondmult_report cache/bondmult*_wl_*.npz`.

## 16. Would RDKit perceive a better graph? Yes in principle, no in practice — but there is a free win

`perception_rdkit.py`, on 2,001 structures from 871 same-molecule provenance
families (the §11 sample), scoring each perceiver by the same metric: **what
fraction of same-molecule families does it assign different graphs to?** A true
chemical graph would score 0.

| perceiver | what it is | ok | ms/mol | bonds/atom | families split |
|---|---|---|---|---|---|
| ASE 1.0 | covalent radii × 1.0 | 2001 | 1.7 | 1.033 | 35.1% |
| ASE 1.1 | covalent radii × 1.1 | 2001 | 1.7 | 1.082 | 61.7% |
| **ASE 1.2 (production)** | covalent radii × 1.2 | 2001 | 1.7 | 1.326 | **87.7%** |
| **`DetermineConnectivity()`** | connect-the-dots | 2001 | **0.7** | 1.024 | **28.4%** |
| `DetermineConnectivity(useVdw)` | covalent radii × 1.3 | 2001 | 0.6 | 1.039 | 33.9% |
| `DetermineConnectivity(useHueckel)` | extended Hückel overlap | — | 170 | 1.032 | 33.3%¹ |
| `DetermineBonds()` | xyz2mol + bond orders | **90/300** | **1521** | 0.968 | **2.3%**² |
| `DetermineBonds()`, orders ignored | the connectivity underneath it | 90/300 | 1502 | 0.968 | **0.0%**² |

¹ n=64; the process **segfaults** on the full sample (see below).
² conditioned on the 30% of structures xyz2mol could handle — a favourable subset.

**The hypothesis is right.** A valence-based chemical graph *is* essentially
conformer-invariant: 0.0% split where xyz2mol succeeds, against 87.7% for the
production perceiver. The instability §11 identified is real and a chemistry-aware
perceiver removes it.

**RDKit cannot deliver it on this dataset.** `DetermineBonds` fails on **70%** of
structures, and the failures are exactly where predicted — **98% of metal
complexes** (20% of `train_4M`), 67% of biomolecules, 34% of electrolytes, 0% of
reactivity. xyz2mol assumes organic valences; this dataset is 20% metal complexes and
17% open-shell. It also costs **1.5 s/molecule** (4M ≈ 70 core-days) and, worse,
cannot be time-bounded in-process: the search sits inside a C++ call, so a `SIGALRM`
handler never gets to run. `useHueckel=True` is worse — 170 ms/mol *and* it takes
the interpreter down with a silent segfault, uncatchable by `try/except`.

**But `DetermineConnectivity()` is a free win.** Connect-the-dots is **2.4× faster
than ASE** (0.7 vs 1.7 ms/mol), never fails, and cuts the split rate from 87.7% to
**28.4%** — better than any ASE multiplier including the 1.1 that §15 measured as
worth +0.046 R². It is a drop-in replacement for `build_graph` on both axes.

**One subtlety worth knowing: the bond orders are slightly *destabilising*.**
Ignoring them scores 0.0% and keeping them scores 2.3% on the same molecules — so
xyz2mol occasionally assigns different resonance forms (which C–C is the double bond
in a conjugated system) to different conformers. The prize is the *connectivity*, not
the orders.

### 16b. …and on R² it is a disaster. The split-rate proxy was misleading

`build_graph` now takes a `perceiver` argument (`ase` default, `rdkit_ctd`),
threaded through both featurisers, both pipelines and `dim_sweep --perceiver`, so
connect-the-dots could be run as a fourth arm of the §15 sweep — same 20k subset,
same 6 seeds, same frozen splits.

| arm | bonds/atom | families split | GP R² mean | GP R² std | OLS R² |
|---|---|---|---|---|---|
| ASE 1.0 | 1.03 | 35.1% | 0.2752 | 0.0455 | 0.2960 |
| **ASE 1.1** | 1.08 | 61.7% | **0.3618** | 0.0347 | 0.3580 |
| ASE 1.2 (production) | 1.33 | 87.7% | 0.3158 | 0.0734 | 0.2998 |
| **`rdkit_ctd`** | 0.97 | **28.4%** | **0.0211** | 0.1144 | 0.0540 |

Paired against the production default: `rdkit_ctd` **−0.2946 ± 0.167, t = −4.31,
losing on 6 of 6 seeds**. It is by far the worst arm — one seed goes outright
negative (−0.210).

**The proxy inverted the ranking.** Ordered by split rate, `rdkit_ctd` (28.4%) looked
best and ASE 1.2 (87.7%) worst; ordered by R², that is exactly reversed. The caution
at the end of §15 was warranted, and stronger than expected.

**It is not an outlier artefact.** The first thing that looked like an explanation —
at n=2000 a single trans1x reaction structure landed outside the training embedding
box and dragged OLS to −1.09 — does not survive at 20k: only **0.1% of test rows**
fall outside the training box under `rdkit_ctd`, the same as ASE, and no row exceeds
3× any training column. The embedding is not unstable; it simply carries less
information about the energy.

**The coherent explanation, now supported three independent ways.** The WL channel's
predictive power on this dataset comes substantially from the *geometry that leaks in
through threshold-based perception*. Purify the graph toward chemistry and the
descriptor gets worse:

* §11 — the perceived graph is a geometry fingerprint (87% of conformer families split);
* §15 — ASE 1.0, the most chemically pure threshold, is worse than 1.2;
* §16b — `rdkit_ctd`, valence-aware and the most chemically principled connectivity
  of the four, is worst of all.

RDKit's pruning removes precisely the weak/long contacts that a valence check calls
spurious, and on a dataset whose median max|F| is 3.4 eV/Å those contacts are
carrying real energetic information. Note it is not a simple edge-count effect
either: `rdkit_ctd` (0.97 bonds/atom) is far worse than ASE 1.0 (1.03), and 1.1
(1.08) beats 1.2 (1.33). *Which* edges, not how many.

**Verdict: do not adopt RDKit for the WL channel.** The chemically-correct graph is
the right object for cheminformatics and the wrong object for this regression. If
conformer-stable graph identity is wanted, the way to get it is to keep the loose
perception *and* let the geometry channels carry the 3D information explicitly —
which is what `geom` and `strain` already do — not to purify the graph and hope it
still encodes geometry implicitly.

The plumbing stays in (`perceiver="ase"` is the default, so nothing changes unless
asked) because it is what makes this question answerable, and re-answerable at 200k.

### 16c. The redundancy hypothesis — right, and it was the bond ORDERS all along

§16b judged each perceiver on its **standalone** R². That is the wrong test for an
*additive* kernel, where what matters is a channel's **marginal** contribution given
geometry. The hypothesis: ASE-WL scores well alone because it leaks geometry, but
geometry is already carried better by the `geom` channel, so a purer bond graph could
score worse alone and still contribute *more* on top. `orthogonality.py` and
`orthogonality_organic.py` test it.

**Two corrections to §16 and §16b came out of this.**

*First, a methodological confound.* Under the production reducer the RDKit arm has a
**negative** solo R² — an uninformative-but-neutral feature set would give 0, not
negative. `SparsePLS` is *supervised*: it picks directions maximising covariance with
y on train, so on weak features it fits directions that do not replicate. Repeating
everything with an unsupervised truncated SVD removes that confound, and the
catastrophic §16b verdict softens: over the whole 20k the two perceivers become
indistinguishable (paired increment difference **−0.0002 ± 0.0038**). So "RDKit is a
disaster" was substantially "RDKit **+ supervised PLS** is a disaster".

*Second, a mislabelled object.* §16 credited the 0.0% conformer-split figure to
xyz2mol's valence reasoning. It does not: `DetermineBonds` *is*
`DetermineConnectivity` plus order assignment, so discarding the orders reproduces
connect-the-dots exactly — the arms came out byte-identical. That 0.0% is
connect-the-dots restricted to the molecules xyz2mol can solve, i.e. it measures the
easy organic subset, not bond-order chemistry. To test the orders they have to be
*kept*, and since WL labels nodes rather than edges, they are folded into each atom's
label as the sorted multiset of its incident bond orders (`6#1,1,2` = sp² carbon).

**The definitive test** therefore runs on the organic subsets, where `DetermineBonds`
is 91% successful and **1.7 ms/mol** (the 1.5 s/mol average was entirely metal
complexes grinding in the combinatorial search), keeping only molecules every
perceiver handles so all arms see identical rows — 6.3k molecules, 3 seeds,
unsupervised reducer, ridge, held-out:

| perceiver | solo R² | **increment over geom** | on geom-residual | redundancy | mean CC |
|---|---|---|---|---|---|
| ASE 1.2 | +0.1065 | **+0.0024** | +0.0065 | 26.7% | 0.421 |
| `rdkit_ctd` (connectivity) | +0.1221 | **+0.0046** | +0.0048 | 28.6% | 0.429 |
| **`rdkit_bonds` (+ orders)** | +0.1424 | **+0.0253** | +0.0376 | **24.4%** | **0.397** |

Paired per-seed against ASE: `rdkit_bonds` **+0.0228 ± 0.0067**, positive on all
three seeds (+0.0212, +0.0171, +0.0302). `rdkit_ctd` is +0.0022 ± 0.0046 — a wash.

**The hypothesis holds, and it localises the signal.** Given a geometry channel, the
accurate bond graph contributes **~10× more than the ASE graph** (+0.025 vs +0.002),
and it is simultaneously the *least* redundant with geometry on both measures
(24.4% of its columns linearly predictable from `geom`, mean canonical correlation
0.397 — the lowest of the three). Exactly the predicted signature: weaker alone,
more orthogonal, worth more in combination.

**But the value is in the bond orders, not the connectivity.** `rdkit_ctd` — accurate
connectivity, no orders — contributes no more than ASE (+0.0046 vs +0.0024, inside
noise). Only the arm carrying single/double/triple/aromatic distinctions moves.
That is chemistry a distance threshold cannot express at any multiplier, which is why
it survives conditioning on geometry.

**Caveats, and they are load-bearing.** (i) This is the **organic third** of
`train_4M` — precisely the part xyz2mol can solve; the 20% that is metal complexes
fails 98% of the time and cannot be tested this way at all. (ii) The absolute
increment is small: +0.025 on a 0.61 baseline. (iii) It is measured linearly on
embeddings, not through the GP.

**What this suggests building** is not a perceiver swap but a *hybrid*: keep ASE
perception for the graph everywhere, and add a **bond-order channel** wherever
xyz2mol succeeds, with a per-structure fallback for the rest. The failure rate makes
that a routing problem rather than a replacement, and §14's finding that the kernel
already wastes signal on rigid category blocks suggests the routing is worth the
complexity only if the +0.025 survives at production N and through the GP.

### 16d. Through the actual GP: direction confirmed, but neither WL variant earns its block

§16c was linear-on-embeddings. Run as a real additive gp2Scale channel instead —
organic subsets only, rows every perceiver handles (6,251 of 20,000), `geom` held on
ASE in **both** arms so only the WL channel changes, per-channel signal variances
**trained** by marginal likelihood (frozen `var(y)/n_channels` dilutes a weak channel
by construction and would have measured that instead), 3 seeds:

| WL perceiver | wl alone | geom | additive | **additive − geom** |
|---|---|---|---|---|
| ASE 1.2 | 0.3519 | 0.5939 | 0.5253 | **−0.0686** |
| `rdkit_bonds` | **0.4050** | 0.5939 | 0.5701 | **−0.0238** |

Paired per-seed, `rdkit_bonds` − ASE: **+0.0448 ± 0.0688** on the increment
(2/3 seeds, t = 1.13), **+0.0531 ± 0.0829** standalone (2/3 seeds). `geom` is
byte-identical across arms (0.5834 / 0.6162 / 0.5821, same cutoffs), confirming the
pairing.

**Three things this settles and one it does not.**

*The direction holds under the GP.* The bond-order channel beats the ASE graph both
standalone (+0.053) and in combination (+0.045), and on seed 123 it produces the only
configuration in this whole experiment where adding a WL channel to geometry actually
*helps* (0.6193 vs 0.5821). §16c's linear estimate (+0.023 SVD, +0.045 PLS) and this
GP estimate (+0.045) agree in sign and magnitude.

*On organics the bond graph is the better WL descriptor, full stop.* Standalone
0.4050 vs 0.3519. This is the exact opposite of the whole-dataset §16b result, and
the reason is now clear: `rdkit_ctd` was being run over metal complexes and clusters
it has no business perceiving, whereas here it runs only where xyz2mol succeeds.

*But neither variant earns a kernel block.* Both increments are **negative**:
geometry alone (0.5939) beats geometry + ASE-WL (0.5253) and geometry + bond-order WL
(0.5701). On the organic third of `train_4M`, the best model tested is the geometry
channel by itself. The bond-order channel is *less bad*, not good.

*What n = 3 does not settle* is the +0.045 itself (t = 1.13, one seed negative). The
seed spread is large because the arms differ most exactly where the ASE graph fails.

**The indicated next step is placement, not more seeds.** The project's own
channel-placement criterion says a channel earns a Gram block only to the extent its
effect is *non-additive*, and that the prior mean is the default home — it costs no
block, adds no neighbours, cannot worsen conditioning, and reaches every test point.
A channel whose kernel increment is negative but whose information is real is exactly
the case that criterion was written for. `--mean-channels` already supports it:
putting the bond-order channel in the **mean only** is the experiment that should
follow, and it is cheap.

### 16e. Prior mean instead of kernel: it is not a placement problem

The channel-placement criterion says the prior mean is the default home for a
channel, and the kernel must earn its block. §16d's negative kernel increment made
that the obvious next test. Same organic subset, same 6,251 rows, same 3 seeds,
`geom` on ASE throughout, `--train`:

| placement of the WL channel | ASE | vs geom | `rdkit_bonds` | vs geom |
|---|---|---|---|---|
| **C: not present** (geom only) | 0.5939 | — | 0.5939 | — |
| **B: prior mean only** | 0.5177 | **−0.0762** | 0.5684 | **−0.0255** |
| **A: kernel block + mean** | 0.5253 | −0.0686 | 0.5701 | −0.0238 |

**Moving it to the mean does not rescue it.** Both variants still land below geometry
alone, and by almost exactly the same margin as when they had a kernel block. The WL
channel is not mis-placed on this subset — its information is *actively harmful*
alongside geometry, wherever it is put. That is a clean negative and it closes the
line of inquiry: no arrangement of these two channels beats the geometry channel by
itself on organics.

Two things the split does establish, though:

*The kernel block is worthless for the bond-order channel, and nearly so for ASE.*
A − B, i.e. what the Gram block buys over the same channel in the mean alone:
`rdkit_bonds` **+0.0017 ± 0.0031**, ASE **+0.0077 ± 0.0020**. The bond-order
channel's effect is entirely additive-and-global — exactly the strain-like signature
the placement criterion predicts for a chemistry term. So *if* it is ever used, it
belongs in the mean, and the block is pure cost.

*The `rdkit_bonds` advantage over ASE is stable across placements.* +0.0508 ± 0.0670
in the mean, +0.0448 ± 0.0688 in the kernel, 2/3 seeds both times. The effect is a
property of the descriptor, not of where it is wired in.

**Where this leaves the hypothesis.** The orthogonality reasoning was right and
survived four independent tests — redundancy measures, linear increments, GP kernel,
GP prior mean — all agreeing that the accurate bond graph carries something the ASE
graph does not (~+0.05). What it does *not* survive is the comparison that matters
for shipping: on the organic third of `train_4M`, **the geometry channel alone beats
every WL configuration tested**, by 0.02–0.08. The bond-order channel is the best WL
variant available and still is not worth its place.

The honest reading is that this is a statement about *the organic subset*, where
geometry is unusually dominant (`geom` alone reaches 0.594 there vs ~0.49 dataset-wide).
The subsets where WL might still pay are the ones xyz2mol cannot perceive — metal
complexes above all — which is precisely where this experiment cannot go.

## 17a. WL depth: depth 1 is a 7× scaling win for free

§12 showed the WL vocabulary is overwhelmingly depth-3 (65% of all labels, only 10%
of them ever seen twice) and that this is what drives the 35% OOV rate. Depth is the
second lever on that pathology, after `cutoff_mult` (§15). `dim_sweep --depth`
already existed, so this is a run, not a change: 20k dataset-wide, `--channels wl`,
**6 seeds**, paired splits.

| depth | kept columns | test OOV | GP R² mean | GP R² std | paired vs depth 3 |
|---|---|---|---|---|---|
| **1** | **7,647** | **2.6%** | 0.3085 | **0.0598** | **−0.0073 ± 0.0337** (t = −0.53) |
| 2 | 28,186 | 11.1% | 0.2999 | 0.0931 | −0.0159 ± 0.0261 (t = −1.49) |
| 3 (production) | 54,850 | 21.0% | 0.3158 | 0.0734 | — |

*Correctness check*: the depth-3 arm reproduces the §15 `ase 1.2` arm **exactly**
(mean 0.3158; per-seed 0.269 / 0.238 / 0.395 / 0.343 / 0.417 / 0.233) — same
configuration, so the harness and the pairing are verified.

**Depth 1 costs nothing measurable and saves almost everything.** Its R² is
indistinguishable from depth 3 (−0.007, t = −0.53, 3/6 seeds positive) while using
**7.2× fewer columns** and cutting the out-of-vocabulary rate **8×**, from 21.0% to
2.6%. It also has the *lowest* across-seed spread of the three (0.0598 vs 0.0734).
By the decision rule set for this experiment — parity on R², material savings on
columns and OOV — depth 1 wins outright.

Depth 2 is the worst of the three on both axes, which is worth noting because it
rules out a simple "shallower is better" reading: the ordering is 1 ≈ 3 > 2, not
monotone. As with `cutoff_mult` (§15), the relationship between graph resolution and
predictive power on this dataset is not monotone in the obvious direction.

**Why this matters most at 4M.** A depth-1 label is `(element, sorted multiset of
neighbour elements)` — a space bounded by chemistry, not by dataset size. The
measured growth bears that out: depth-1 kept columns go 7,650 → 14,481 as the fitting
set grows 16k → 60k, an exponent of ≈0.49, against the ≈0.91 of the pooled vocabulary
(§12). Depth 3 is what makes the vocabulary scale nearly linearly with N and
extrapolate to ~7.8M columns at 4M; depth 1 largely removes that problem.

**Recommendation:** switch the default to `depth=1` for the 4M ladder on the cost and
stability evidence, and re-check R² parity at 200k before committing — the same
caveat as §15, since a larger training set estimates rare deep labels better and
could tilt the balance back toward depth 3.

## 17b. Metal complexes: no available library gives a stable graph

§16 could not test metal complexes at all — `DetermineBonds` fails on 98% of them —
and I claimed at the time that nothing existed for the job. That was wrong. Several
tools do exist, so they were screened on the metric used throughout: the fraction of
same-molecule provenance families a perceiver assigns *different* graphs to. 1,200
`metal_complexes` structures from 466 multi-member families:

| perceiver | ok | ms/mol | bonds/atom | **families split** |
|---|---|---|---|---|
| ASE 1.0 | 1200 | 1.8 | 1.090 | 62.0% |
| ASE 1.1 | 1200 | 1.8 | 1.157 | 80.7% |
| **ASE 1.2 (production)** | 1200 | 1.8 | 1.485 | **94.4%** |
| **`rdkit_ctd`** | 1200 | **0.9** | 1.078 | **53.0%** |
| `pymatgen` CrystalNN | 1199 | 688 | 0.935 | 86.5% |
| `pymatgen` VoronoiNN (tol 0.5) | 1200 | 928 | 1.436 | 98.9% |
| `pymatgen` EconNN | 1200 | 38 | 0.772 | 79.6% |

**None of the materials-science algorithms help.** CrystalNN — the top scorer in the
MaterialsCoord benchmark, and the one with the strongest prior claim here since metal
coordination is its home ground — splits 86.5% of same-molecule families at 688
ms/mol. VoronoiNN is worse than doing nothing (98.9%). The best perceiver for metals
is the same connect-the-dots that won the pooled comparison, at 53.0%, and it is also
by far the cheapest (0.9 ms/mol, 765× faster than CrystalNN).

*Internal control*: a perceiver can trivially lower its split rate by finding fewer
bonds, so bonds/atom is reported alongside. `rdkit_ctd` wins while finding **more**
bonds than CrystalNN (1.078 vs 0.935) and than EconNN (0.772), so its advantage is
real rather than an artefact of under-bonding.

**Two things this settles.**

*Production perception is at its worst exactly where it is hardest.* ASE at 1.2
splits 94.4% of metal families, against 87.7% pooled — and for metals the ordering
reverses, with the *tightest* multiplier best (1.0 → 62.0%). Metal–ligand distances
are long, so a 1.2× covalent multiplier over-connects the coordination sphere badly.
This is in tension with §15, where 1.1 was the best *R²* arm dataset-wide; the two
metrics disagree, and R² is the one that decides production.

*A conformer-stable metal graph is not obtainable off the shelf.* The best available
option still assigns different graphs to half of all same-molecule pairs. That is
consistent with the literature: the `xyz2mol_tm` paper reports only **>70% agreement**
between three independent ways of assigning TMC connectivity on tmQMg. Metal
connectivity is an open problem, not a library call.

**Correcting the record on what exists.** For anyone revisiting this:
[`xyz2mol_tm`](https://github.com/jensengroup/xyz2mol_tm) (xyz2mol extended to TMCs
via extended Hückel, *J. Cheminform.* 2025),
[`molSimplify`](https://github.com/hjkgrp/molSimplify) (TMC graphs plus ML models for
coordinating atoms), `pymatgen.analysis.local_env` (screened above), and `xtb --wbo`
(GFN2 Wiberg bond orders, Z ≤ 86). Two practical notes: our shards store
`nbo_charges` but **not** Wiberg/NBO bond orders, so the electronic route the
`xyz2mol_tm` paper uses as ground truth would have to be recomputed rather than read;
and in pymatgen 2026.5.4 `EconNN` advertises `molecules_allowed = True` but its
`get_nn_info` still calls `_get_image`, which needs `frac_coords` a molecule
neighbour does not have — all three algorithms have to be run through a padded
periodic box.

**Line closed.** No GP work follows from this: nothing beats the perceiver we already
have, and the one that is best is already available and already screened (§16b) as
harmful to R² dataset-wide.

## 17. Next

1. ~~Sweep `cutoff_mult`~~ — done, §15.
2. **Cap the WL depth at 1–2.** Depth 3 contributes 65% of the vocabulary and only
   10% of its labels are ever seen twice. Untested.

And two structural questions this pass has now made concrete: whether the
cross-category blocks should be relaxed (§14), and whether the kernel should carry
per-category signal variance and noise (§13, §3).
