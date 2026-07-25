# data_exploration

A complete characterisation of `train_4M/` (the OMol25 4M training split) for two
audiences: the **poster** (what this dataset is, how chemically diverse it is) and
**descriptor design** (what a graph descriptor can and cannot reach, and which
modelling choices in `wl_gp2scale/` the data actually supports).

`PLAN.md` is the design document; `REPORT.md` is the written result with every
figure and number; `DATASET_CARD.md` is the one-page fact sheet.

## Status

| stage | what it does | cost | done |
|---|---|---|---|
| 1. census | exact metadata scan of all 3,986,754 structures | ~5 min, 8 workers | ✅ |
| 2. families | redundancy + variance-component ceilings | ~3 min | ✅ |
| 3. stats + figures | aggregates → `summary.json`, figures | ~4 min | ✅ |
| 4. geom_sample | two 200k structural (3D) samples | ~6 min, 8 workers | ✅ |
| 5. descriptors | ceiling, WL vocabulary, perception, variogram, cross-category | ~30 min | ✅ |

## Run it

```bash
python -m data_exploration.run_all --workers 8
```

Stages are individually runnable (`python -m data_exploration.census`, `.families`,
`.stats`, `.figures [--only f1 f3]`) and the census resumes from existing shard
parts, so only the first run pays for the scan.

Dependencies are those already in the `omol` env — numpy, scipy, matplotlib, ase,
fairchem. Nothing new (no pandas/pyarrow): the census is stored as an `.npz` of
column arrays.

## Files

| file | role |
|------|------|
| `sources.py` | parse `atoms.info["source"]` → (top, group, job, family, step) |
| `census.py` | PASS A: parallel per-shard scan → `cache/census_parts/*.npz`; `load_census()` |
| `families.py` | redundancy, effective N, and the ANOVA variance-component ceilings |
| `geom_sample.py` | PASS B: the two 200k structural samples (bond graph, geometry, WL hashes) |
| `descriptors.py` | step 5: `ceiling`, `vocab`, `perception`, `variogram`, `cross` |
| `strain.py` | the strain-from-positions gate (promoted into `wl_gp2scale`) |
| `soap_screen.py` | SOAP / dscribe screen against the hand-rolled geometry channel |
| `stats.py` | every aggregate → `cache/summary.json`; `load_all()` shared loader |
| `style.py` | the shared visual system (palette, rcParams, `save()`) |
| `figures.py` | one function per figure → `figures/*.png` + `*.pdf` |
| `run_all.py` | driver (census → families → stats → figures) |

## Outputs

`cache/` (gitignored, regenerable):

| artefact | contents |
|---|---|
| `census_parts/shard_*.npz` | 80 parts, ~30 columns × 3.99M rows + element-count COO |
| `families.npz` | the target `y`, group keys, size CCDFs |
| `families.json` | ceilings and redundancy stats, overall and per subset |
| `summary.json` | **every headline number**, machine-readable |
| `derived.npz` | element co-occurrence matrix, per-element counts |
| `warnings.json` | the 5 distinct ORCA warning strings and their counts |

`figures/` (tracked): 14 figures, each as 300 dpi PNG **and** vector PDF.

## Pass B and step 5

```bash
python -m data_exploration.geom_sample --sample both --n 200000 --workers 8
python -m data_exploration.descriptors ceiling      # the graph ceiling attempt
python -m data_exploration.descriptors vocab        # WL vocabulary growth + OOV
python -m data_exploration.descriptors perception   # cutoff_mult sweep
python -m data_exploration.descriptors variogram --n 20000
python -m data_exploration.descriptors cross --n 20000
```

Results land in `cache/descriptors.json`; figures F13–F17 and F20 read it.
`variogram` and `cross` share a cached embedding (`cache/emb_*.npz`), so the
featurisation is paid once.

## Three things to know before reading the numbers

**The family parse is a heuristic.** `sources.py` infers "same molecule" from job
naming conventions that are not documented. Every ceiling is therefore also
reported for the parse-free grouping `(formula, charge, spin)`, and the report
quotes the bracket rather than a single number.

**In-sample group means are not a ceiling.** With 1.2M groups over 4M rows, group
means fit their own rows almost perfectly. Everything quoted as a ceiling is the
one-way random-effects variance component (`SS_within / (n − g)`), i.e. what a
descriptor constant within groups could reach on *held-out* rows; the inflated
in-sample value is kept in `families.json` alongside it for comparison.

**The WL hash is not a graph invariant on this data, so the graph ceiling is still
a bracket.** `descriptors.py ceiling` prints a number for it and then a consistency
check that invalidates that number: bond perception is distance-thresholded, so 87%
of same-molecule conformer families receive *different* WL hashes, and the
within-WL variance comes out below the within-molecule variance — impossible for a
true invariant. The ceiling stays [0.53, 0.92]; see `REPORT.md` §11.
