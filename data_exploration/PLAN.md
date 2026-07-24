# `data_exploration` — plan

> **Status (2026-07-24): steps 1–3 are done** — the full 4M census, the redundancy
> and ceiling analysis, and 14 figures. Results in `REPORT.md`, facts in
> `DATASET_CARD.md`, numbers in `cache/summary.json`. Steps 4–5 (the 3D structural
> pass and the descriptor diagnostics) are not started.
>
> One planned figure changed shape once the data came in. **F3 was designed as
> "redundancy CCDF + within/between variance"**, on the hypothesis that `train_4M`
> was full of conformer duplicates. It is not: 3.63M provenance families for 3.99M
> structures. So F3 became the **description ceiling ladder** (what each level of
> description could explain, at best) and the redundancy CCDF moved to F3b. The
> ceilings are also reported as unbiased variance components rather than the
> in-sample split described in §3.4 — with 1.2M groups over 4M rows the in-sample
> number is inflated by a third and would have overstated the case.

**Goal.** A complete, quantitative characterisation of `train_4M/` (the OMol25 4M
training subset) with two audiences:

1. **Poster** — a small set of publication-quality figures that show *what this
   dataset is* and *how chemically diverse it is*.
2. **Descriptor design** — measurements that decide open modelling questions in
   `wl_gp2scale/` and `descriptor_eval/`: what a graph-only descriptor can and
   cannot explain, whether the category block-sparse kernel is justified, how much
   of the 4M is redundant, and how big a descriptor vocabulary 4M demands.

Everything is measured, cached, and reproducible; nothing is asserted from the
OMol25 paper without checking it against these shards.

---

## 0. What the data actually is (already verified this session)

| fact | value | how |
|---|---|---|
| structures | **3,986,754** | `len(AseDBDataset({'src':'train_4M'}))` |
| shards | 80 `.aselmdb`, ~49,835 rows each | `ls`, single-shard open |
| on disk | 21 GB | `du -sh` |
| sequential read | **~3.3k structures/s/process** | 3000 rows in 0.9 s |
| random read | ~1.3k/s | 4000 random `get_atoms` in 3.0 s |
| elements seen in a 4k sample | **79** | element counter |
| `data_id` subsets | 10: `biomolecules`, `metal_complexes`, `reactivity`, `elytes`, `ani2x`, `trans1x`, `geom_orca6`, `rgd`, `orbnet_denali`, `spice` | 4k sample |
| sizes | 2 – 340 atoms, median 42, p99 200 | 4k sample |
| charge / spin | 57% neutral, charges −3…+4; 83% singlet, up to spin 8 | 4k sample |

**Per-structure metadata available** (`atoms.info`, 23 keys — richer than the
current pipeline uses):

`source`, `reference_source`, `data_id`, `charge`, `spin`, `num_atoms`,
`num_electrons`, `num_ecp_electrons`, `n_scf_steps`, `n_basis`, `unrestricted`,
`nl_energy`, `integrated_densities`, `homo_energy`, `homo_lumo_gap`, `s_squared`,
`s_squared_dev`, `warnings`, `mulliken_charges`, `lowdin_charges`, `nbo_charges`
(present in only ~66% of rows), `composition`, `sid`
— plus `positions`, `numbers`, `energy`, `forces`.

Two of these are load-bearing and currently unused:

- **`source`** is a provenance *path* (e.g.
  `orbnet_denali/orbnet_CHEMBL1935278_conformers_<hash>_675082_0_1/orca.tar.zst`,
  `tm_react/MOR31_Sr2_Charge0_UHF0_swaplig_..._step18_0_1/orca.tar.zst`). It
  encodes the **conformer / trajectory family** each structure came from. This is
  the key to measuring redundancy — see §3.
- **`homo_lumo_gap`, `s_squared_dev`, `n_scf_steps`, `warnings`** are QC and
  electronic-structure descriptors we have never looked at.

**Feasibility.** A full-census pass over all 4M rows costs ~20 min single-process,
**~3–4 min across 8 processes** (10 cores available). So the census is *exact*, not
sampled. Only the 3D-geometry pass (§4) is sampled.

---

## 1. Module layout

```
data_exploration/
├── PLAN.md                 ← this file
├── README.md               ← how to run + what each artefact is (written with the code)
├── style.py                ← poster matplotlib rcParams, colourblind-safe subset palette
├── census.py               ← PASS A: full 4M metadata scan (parallel by shard) → cache/census.npz
├── families.py             ← `source` → (subset, family, member) parsing + redundancy stats
├── geom_sample.py          ← PASS B: stratified ~200k structural sample → cache/geom_sample.npz
├── descriptors.py          ← WL-vocab growth, distance/variogram diagnostics (reuses existing code)
├── stats.py                ← all aggregations; writes cache/summary.json
├── figures.py              ← one function per figure, all reading the caches
├── run_all.py              ← driver: census → families → geom → descriptors → stats → figures → REPORT.md
├── cache/                  ← gitignored artefacts
└── figures/                ← .png (300 dpi, poster) + .pdf (vector) per figure
```

**Dependencies: none new.** `numpy`, `matplotlib`, `ase`, `fairchem`, `scipy` are
all present in the `omol` env; `pyarrow`/`pandas` are *not*, so the census is stored
as an `.npz` of column arrays (4M × ~20 numeric columns ≈ 320 MB, fits in the 16 GB
box) with string columns stored as integer codes + a name table. No pandas needed.

**Reuse, don't reimplement:** `wl_gp2scale/data.py::ExtensiveMean` (the target),
`wl_gp2scale/wl_features.py::SparseWLFeaturizer`, `wl_gp2scale/geometry_features.py`,
`wl_gp2scale/reduce.py::SparsePLS`, `descriptor_eval/variogram.py`. This module
*measures* the dataset; the featurisers stay where they are.

---

## 2. Pass A — the full census (exact, all 3,986,754 rows)

`census.py`: `multiprocessing.Pool` over the 80 shards; each worker opens one
`.aselmdb`, iterates sequentially, and returns column arrays. Per structure we
record:

- **identity**: `sid`, shard, `data_id` code, `source` → (top-level source code,
  family hash `uint64`, member index)
- **size / cost**: `num_atoms`, `n_heavy`, `num_electrons`, `n_basis`,
  `n_scf_steps`, `unrestricted`
- **state**: `charge`, `spin`
- **energy**: `energy`, `nl_energy`, `homo_energy` (α/β), `homo_lumo_gap` (min of
  the two), `s_squared`, `s_squared_dev`
- **forces**: `max|F|`, `RMS|F|` (the off-equilibrium coordinate)
- **charges**: has-NBO flag, Löwdin charge min/max/RMS, molecular dipole magnitude
  from Löwdin charges
- **composition**: element-count row → a `(4M × 79) uint8` matrix (316 MB) +
  a `uint64` hash of the sorted formula for exact composition-duplicate counting

Cost: one full read of the 21 GB, ~4 min on 8 workers. Written to
`cache/census.npz` (+ `cache/element_counts.npy`, `cache/name_tables.json`).
Everything downstream in §5–§7 reads only this file — the LMDBs are touched once.

**Checkpointing:** per-shard `.npz` parts so an interrupted scan resumes.

---

## 3. Redundancy: how many *distinct* molecules are in "4M"?

This is the single most decision-relevant analysis, because
[[wl-gp2scale-conditioning]] traced the CG stall to **near-duplicate molecules
making the kernel near-singular**, and the graph-only R² ceiling (0.56 at 200k) is
hypothesised to be **within-conformer-family energy variance that a graph cannot
see**. Both are directly measurable here.

`families.py` parses `source` into `(dataset, family_key, member)` and computes:

1. **Family-size distribution** — how many structures per molecule/trajectory,
   per subset. Reported as a CCDF plus "fraction of the dataset living in families
   of size ≥ k".
2. **Effective N** — number of distinct families, and the inverse Simpson index
   `1/Σp_f²` (the redundancy-corrected effective sample size). Headline number:
   *"4M structures = X distinct molecular families."*
3. **Exact-composition duplicates** — distinct formula hashes; Zipf plot of formula
   frequency.
4. **Variance decomposition of the regression target.** Fit `ExtensiveMean` on the
   census (it needs only element counts, charge, spin, energy — all in the census),
   get the intensive residual `y`, then split
   `Var(y) = Var_between-family(ȳ_f) + E_f[Var_within-family(y)]`.
   The within-family share is **the fraction of the target that no graph-only
   descriptor can ever explain** — an *a priori* ceiling for WL, computed without
   fitting a single GP. Also computed per subset, and against a stricter grouping
   (identical WL graph hash, §5) so the ceiling is descriptor-exact rather than
   provenance-exact.
5. **Redundancy → sampling advice**: with the measured family sizes, what does a
   uniform random 200k draw actually contain (how many families, expected duplicate
   pairs), and what would a family-deduplicated 200k contain? This feeds directly
   into whether the 4M ladder should sample uniformly or by family.

---

## 4. Pass B — the structural sample (stratified, ~200k)

Geometry-level quantities need positions and an O(n²)-ish neighbour build
(~22 ms/molecule measured previously), so this pass runs on a **stratified sample
of ~200k** (proportional-with-floor across the 10 subsets, so the small subsets are
resolvable; seeds frozen, indices cached). ~1.2 core-hours → minutes on 8 cores.

Per sampled structure: covalent bond graph (ASE neighbour list), **number of
disconnected fragments** (is this one molecule or a cluster/complex?), bond-length
and bond-angle distributions by element pair, coordination numbers by element,
radius of gyration and `R_g / n_atoms^{1/3}` (compactness), longest intramolecular
distance, min intermolecular contact distance, ring count (cyclomatic number
`E − V + fragments`), rotatable-bond count, per-element Löwdin charge samples, and
the **WL graph hash** at 1–3 iterations.

---

## 5. Descriptor-design diagnostics

`descriptors.py`, on the §4 sample:

1. **WL vocabulary growth** — unique WL labels vs. N (log-log), per subset and
   pooled, with the `min_count ≥ 2` filter applied as in production. Extrapolate to
   4M: *how large does the descriptor vocabulary have to be, and what is the OOV
   rate of a fresh draw against a vocabulary frozen at 200k?* This directly sizes
   the 4M run and tests whether the frozen-vocabulary assumption survives.
2. **Graph-collision analysis** — how many *distinct WL graphs* are in the sample
   vs. distinct geometries; energy spread within one WL graph (the descriptor-exact
   version of §3.4).
3. **Distance-concentration check** — pairwise-distance histograms in each
   embedding (WL-PLS, geometry-PLS) on a 20k subsample. Bimodal ⇒ compact support
   is well-posed; unimodal-concentrated ⇒ the cutoff has no natural scale. Run per
   subset *and* pooled.
4. **Variogram per channel and per subset** — `γ(h)` for WL and geometry
   embeddings, giving nugget, sill, and range **per `data_id`**. Two decisions come
   out of this: (a) whether one global length scale is defensible or the kernel
   needs per-category scales, (b) an *a priori* per-subset noise floor.
5. **Cross-category structure** — for each pair of subsets, the fraction of
   nearest neighbours that cross the category boundary, and whether cross-category
   pairs carry target correlation. This is a **direct test of the block-sparse
   kernel's core assumption** (it currently zeroes all cross-`data_id` blocks): if
   cross-category neighbours are common *and* correlated, the block structure is
   discarding signal, not just saving flops.
6. **Which metadata predicts the residual** — univariate and joint R² of the
   census columns (`homo_lumo_gap`, `max|F|`, `n_scf_steps`, `s_squared_dev`,
   fragment count, `R_g`, charge spread) against `y`. Cheap, powerful: any column
   with real signal is a free prior-mean feature or a free kernel channel. (The
   current extensive mean uses only element counts, charge, spin, n².)

---

## 6. Data hygiene

A short, honest QC panel — dataset quality is a legitimate poster line and a real
risk to the regression:

- structures with `warnings` set; frequency by warning type
- SCF near-non-convergence (`n_scf_steps` tail)
- spin contamination (`s_squared_dev` tail) — mostly metal complexes, expected
- energy outliers after the extensive mean (|z| > 6), by subset
- missing/NaN fields: `nbo_charges` absent in ~34% of rows (measured), any NaN
  Löwdin charges (the current loader silently *filters on this* when drawing
  subsets — worth quantifying how much of the dataset that rejects, and whether the
  rejection is subset-correlated, i.e. **whether our 200k subsets are biased**)

---

## 7. Figures

Shared `style.py`: one colourblind-safe colour per `data_id` used in *every*
figure, ≥11 pt fonts, no chartjunk, 300 dpi PNG + vector PDF, square-ish aspect for
poster columns. Every figure carries N and the sampling basis in the caption.

### Headline (the four that go on the poster)

| # | figure | what it shows |
|---|---|---|
| **F1** | **Periodic-table heatmap** — each element tile shaded by log₁₀(count) with the % of structures containing it | the chemical-diversity money shot: 79 elements, main group + 3d/4d/5d transition metals + lanthanides |
| **F2** | **Dataset anatomy** — horizontal bar of the 10 subsets (exact counts) with an inset ridgeline of `num_atoms` per subset | what "4M" is made of, and that size varies 2–340 atoms across chemistry types |
| **F3** | **The description ceiling** (built; replaces the planned redundancy figure) — left: best achievable held-out R² per level of description (formula → formula+charge+spin → *molecular graph, bracketed* → molecule identity) with the measured WL GP marked; right: per-subset composition ceiling and irreducible RMSE | composition tops out at 0.53, molecule identity at 0.92, WL measures 0.56 — the quantitative justification for the geometry channel, and which subsets it should target |
| **F3b** | Redundancy CCDF: provenance families vs (formula, charge, spin) groups | `train_4M` is provenance-de-duplicated (12% of rows share a family) but composition-degenerate (80% share a formula/charge/spin) |
| **F4** | **Descriptor-space structure** — pairwise-distance histogram (bimodality) + variogram `γ(h)` per subset with sill/range marked | that compact support is well-posed and where the correlation range sits, per chemistry |

### Supporting (report, and spares for the poster)

| # | figure |
|---|---|
| F5 | Charge × spin joint heatmap, per subset (small multiples) — open-shell/charged coverage |
| F6 | Element co-occurrence matrix (lift, heavy elements) — metal–ligand chemistry |
| F7 | Size distributions: `num_atoms`, `num_electrons`, `n_basis` (cost proxy), heavy-atom count |
| F8 | Composition Zipf: formula frequency rank plot + unique-formula growth vs. N |
| F9 | Target construction: E_total vs. extensive-mean fit (5 orders of magnitude, R² ≈ 1) → residual histogram per subset, log-y (tails) |
| F10 | Residual vs. `num_atoms` — is the target actually intensive? heteroscedasticity check |
| F11 | Off-equilibrium: max\|F\| distribution per subset (minima vs. MD frames vs. reaction paths) |
| F12 | HOMO–LUMO gap distribution per subset; gap vs. residual |
| F13 | Geometry: bond-length distributions by element pair; coordination-number histograms |
| F14 | Fragment count and compactness (`R_g` vs. `n_atoms`) — clusters vs. single molecules |
| F15 | Löwdin charge distribution per element; molecular dipole magnitude |
| F16 | WL vocabulary growth vs. N (log-log) + OOV rate of a fresh draw against a 200k-frozen vocab |
| F17 | Cross-category nearest-neighbour matrix — the block-sparsity assumption test |
| F18 | Metadata → residual R² bar chart (which free columns carry signal) |
| F19 | QC panel: warnings, SCF steps, spin contamination, energy outliers, NaN-charge rejection rate by subset |

---

## 8. Non-figure outputs

- `cache/census.npz` (4M × ~20 columns), `cache/element_counts.npy`,
  `cache/families.npz`, `cache/geom_sample.npz`, `cache/wl_vocab_curve.npz`
- **`cache/summary.json`** — every headline number as a machine-readable record
  (counts, quantiles, effective N, variance shares, ranges, R²s), so poster text and
  the report can never drift from the computation
- **`REPORT.md`** — a written walkthrough: every figure inline, every number
  sourced, and a closing **"implications for descriptor design"** section stating
  what each measurement means for the WL / geometry / additive-kernel decisions
- **`DATASET_CARD.md`** — one page, the facts only (provenance, size, elements,
  charge/spin coverage, licence-relevant provenance, known QC issues)

---

## 9. Order of work

1. `style.py`, `census.py` → run the full scan (~4 min), sanity-check against the
   4k sample numbers above.
2. `families.py` + the variance decomposition → **F3**, the highest-value result.
3. `stats.py` + F1, F2, F5–F12, F19 (all census-only, no second data pass).
4. `geom_sample.py` → F13–F15.
5. `descriptors.py` → F4, F16, F17, F18.
6. `run_all.py`, `REPORT.md`, `DATASET_CARD.md`, `README.md`.

Steps 1–3 are self-contained and deliver the poster's diversity story; 4–6 deliver
the descriptor-design story.

## 10. Open questions this exploration is designed to answer

1. How many **distinct molecules** does "4M structures" actually contain, and what
   is the effective sample size after redundancy?
2. What fraction of the intensive-energy target is **within-family (conformational)**
   — i.e. the hard ceiling on any graph-only descriptor? (Predicts whether the
   measured 0.56 at 200k is near the ceiling or far from it.)
3. Is the **category block-sparse kernel** discarding real cross-subset signal?
4. Does the **WL vocabulary** frozen at 200k survive 4M, and at what OOV rate?
5. Does any **free metadata column** (gap, forces, fragments, `R_g`, charge spread)
   predict the residual well enough to belong in the prior mean or as a kernel
   channel?
6. Is the **NaN-Löwdin filter** in the current loader biasing our subsets?
7. Do the descriptor-distance distribution and variogram range **differ per subset**
   enough to require per-category length scales?
