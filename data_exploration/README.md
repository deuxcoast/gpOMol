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
| 3. stats + figures | aggregates → `summary.json`, 14 figures | ~4 min | ✅ |
| 4. geom_sample | stratified ~200k structural (3D) pass | ~10 min | pending |
| 5. descriptors | WL vocabulary growth, variograms, cross-category test | — | pending |

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
| `stats.py` | every aggregate → `cache/summary.json`; `load_all()` shared loader |
| `style.py` | the shared visual system (palette, rcParams, `save()`) |
| `figures.py` | one function per figure → `figures/*.png` + `*.pdf` |
| `run_all.py` | driver |

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

## Two things to know before reading the numbers

**The family parse is a heuristic.** `sources.py` infers "same molecule" from job
naming conventions that are not documented. Every ceiling is therefore also
reported for the parse-free grouping `(formula, charge, spin)`, and the report
quotes the bracket rather than a single number.

**In-sample group means are not a ceiling.** With 1.2M groups over 4M rows, group
means fit their own rows almost perfectly. Everything quoted as a ceiling is the
one-way random-effects variance component (`SS_within / (n − g)`), i.e. what a
descriptor constant within groups could reach on *held-out* rows; the inflated
in-sample value is kept in `families.json` alongside it for comparison.
