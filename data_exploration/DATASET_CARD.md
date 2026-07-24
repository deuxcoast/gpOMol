# OMol25 `train_4M` — dataset card

One page of facts, all measured on the full split by `data_exploration/census.py`
(2026-07-24). Interpretation lives in `REPORT.md`.

## Identity

| | |
|---|---|
| name | OMol25 train_4M (`train_4M/`, 80 `.aselmdb` shards, 21 GB) |
| structures | 3,986,754 |
| atoms | 218,680,957 |
| level of theory | ORCA, a B97M-V/ωB97M-V-family functional with self-consistent DFT-NL dispersion — this is what the per-structure `warnings` strings and the `nl_energy` field state; the basis set name is not stored in the shards (only `n_basis`) |
| labels | total energy (eV), forces (eV/Å) |
| extra per-structure data | Mulliken / Löwdin / NBO charges and spins, HOMO, HOMO–LUMO gap, ⟨S²⟩, SCF steps, basis size, electron count |

## Composition of the split

| `data_id` | structures | share | median atoms |
|---|---|---|---|
| biomolecules | 799,988 | 20.1% | 66 |
| electrolytes (`elytes`) | 799,799 | 20.1% | 77 |
| metal_complexes | 797,122 | 20.0% | 72 |
| reactivity | 789,845 | 19.8% | 14 |
| ani2x | 215,642 | 5.4% | 14 |
| trans1x | 213,731 | 5.4% | 14 |
| geom_orca6 | 199,805 | 5.0% | 51 |
| rgd | 75,316 | 1.9% | 18 |
| orbnet_denali | 51,102 | 1.3% | 46 |
| spice | 44,404 | 1.1% | 37 |

34 upstream sources in 44 production batches; the largest roots are `omol`
(1.43M), `ani1xbb` (737k), `ani2x` (216k), `trans1x` (214k), `geom_orca6` (200k),
`tm_react` (158k).

## Coverage

- **83 elements.** H 99%, C 97%, O 81%, N 77%, S 31%, P 16%, F 14%, Cl 12%,
  Br 7%, I 5%. All transition metals present (0.2–1% each); full lanthanide row
  (~0.1–0.2%). No actinides, Fr, Ra, Po, At, Rn.
- **Size:** 2–350 atoms, median 42 (22 heavy); 18–9,243 basis functions,
  median 1,032.
- **Charge:** −10 … +10; 56.1% neutral, 43.9% ionic.
- **Spin:** multiplicity 1–11; 82.9% singlet, 17.1% open-shell; 51.6% unrestricted.
- **Composition:** 1,127,158 distinct formulas; 82.8% of structures share a formula
  with another; largest formula group 9,907.
- **Geometry:** median max|F| 3.4 eV/Å (p95 10.2) — predominantly off-equilibrium
  structures, not relaxed minima. Median radius of gyration 4.0 Å.
- **Electronic:** median HOMO–LUMO gap 7.9 eV; 0.2% below 1 eV.

## Redundancy

- 3,633,107 provenance families (mean 1.10 structures, max 174); only 12.2% of
  structures share a family. 3,888,293 distinct ORCA jobs (mean 1.03 frames).
- 1,244,665 distinct (formula, charge, spin) triples (mean 3.2, max 9,835); 80.3%
  of structures share a triple with another.

## Regression target used in this repo

`y = E_total − m(x)` with `m` a ridge-fit extensive mean over
`[1, element counts, q, |q|, q², spin, n²]`, refit on all 4M rows.

- `m` removes all but 3.5 × 10⁻⁹ of the total-energy variance.
- Var(y) = 50.53 eV², std 7.11 eV; per-subset variance 3.5 (trans1x) → 133.3 eV²
  (electrolytes).
- Best achievable held-out R² given composition + charge + spin: **0.53**;
  given full molecule identity: **0.92**.

## Known caveats

- **`warnings` is not a quality field** — all 3,986,754 structures carry
  boilerplate warnings; only "open-shell → UHF" (672,699) is informative.
- **NBO charges are missing in 32.8% of rows**, concentrated in biomolecules
  (47.5% missing), electrolytes (58.7%), metal complexes (55.2%). Löwdin charges
  are present in all but **88** structures.
- 2.7% of calculations took ≥ 100 SCF steps (metal complexes 8.0%); 5.7% show
  ⟨S²⟩ deviation > 0.1 (reactivity 13.6%, metal complexes 11.9%).
- 110 structures sit exactly at max|F| = 50 eV/Å, an upstream clipping boundary.
- Molecule "family" grouping is inferred from `source` naming conventions and is
  a heuristic; parse-free `(formula, charge, spin)` numbers are reported alongside.
