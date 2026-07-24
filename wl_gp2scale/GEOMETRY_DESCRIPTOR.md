# The Geometry + Electrostatics Descriptor

Design document for the 3D geometry/charge molecular descriptor implemented in
[`geometry_features.py`](geometry_features.py) (the scaled, gp2Scale production version)
and originally screened in [`descriptor_eval/geometry.py`](../descriptor_eval/geometry.py).

---

## 1. Purpose and context

The primary descriptor in this project is a **Weisfeiler–Lehman (WL) graph** descriptor
([`wl_features.py`](wl_features.py)): a hashed subtree-pattern histogram over the bond
graph. WL is a *topological* object — it encodes which atoms are bonded to which, and is
invariant to how the molecule is folded in space. Two conformers of the same molecule
(same bond graph, different 3D shape) therefore have **identical** WL vectors but
**different** energies.

The regression target `y` is the *intensive energy residual* — the total DFT energy minus
an extensive, element-referencing mean that already removes composition, net charge, spin,
and size. What remains is dominated by exactly the physics WL cannot see:

- **conformational energy** — rotamers, ring pucker, cis/trans, strain;
- **non-bonded / through-space interactions** — hydrogen bonds, steric clashes, dispersion;
- **internal electrostatics** — the *arrangement* of partial charge, beyond net charge.

This descriptor is engineered to capture that residual. Empirically (see the project
memory / screening notes), the geometry channel alone reaches held-out R² ≈ 0.45–0.49 at
8k–16k training molecules where WL is ≈ 0.21, and the two are complementary as N grows.

The descriptor is deliberately **cheap and hand-engineered** rather than learned or
high-dimensional. This is a design decision: it must scale to millions of molecules (one
pass, embarrassingly parallel, ~O(n²) per molecule with small n), and it feeds a
supervised PLS reduction + compact-support Gaussian-process kernel, so a low-dimensional,
smooth, physically interpretable vector is preferable to a large learned embedding.

---

## 2. Design principles

Four principles, standard in the atomistic machine-learning literature [Behler 2007;
Bartók 2013; Musil 2021], govern every channel.

### 2.1 Invariance to the symmetries of the energy

Energy is invariant under **translation**, **rotation**, and **permutation** of identical
atoms, so the descriptor must be too. We achieve this by working only with *internal*
geometric quantities — interatomic distances, bond angles, dihedral angles — which are
automatically translation- and rotation-invariant, and by **summing over all atoms /
pairs / triples / quadruples** (via histograms), which makes the result permutation
invariant. No atom is ever referenced by index in the output.

### 2.2 A body-order expansion

Invariant many-body descriptors are naturally organized by *body order* — the number of
atoms a term involves. This is the organizing idea behind the Behler–Parrinello symmetry
functions [Behler & Parrinello 2007; Behler 2011], the Many-Body Tensor Representation
[Huo & Rupp 2022], and, systematically, the Atomic Cluster Expansion [Drautz 2019]. Our
descriptor mirrors this hierarchy with one channel per body order:

| Body order | Geometric quantity | Channel |
|---|---|---|
| 2-body | interatomic distance | `rdf` |
| 3-body | bond angle | `angle` |
| 4-body | dihedral (torsion) angle | `torsion` |
| — | partial-charge electrostatics | `elec` |

### 2.3 Chemical (element) resolution

A purely geometric descriptor is chemistry-blind: a carbon ring and a nitrogen ring at the
same distances look identical. Following the species-resolved treatment in partial radial
distribution functions [Schütt et al. 2014], SOAP's per-species density channels
[Bartók et al. 2013], and MBTR's element weighting [Huo & Rupp 2022], the `rdf` and `elec`
channels are resolved by chemical element. The element set is chosen once from the training
data (the `top_k` most common elements) and **frozen**, so the output width is fixed across
all molecules.

### 2.4 Smoothness (Gaussian broadening) and intensivity

- **Broadening.** Distances and angles are placed onto fixed grids as sums of **Gaussians**
  rather than hard histogram bins ([`_gaussian_hist`](geometry_features.py)). This makes
  the representation smooth: two molecules with slightly different geometries map to nearby
  vectors, which a compact-support kernel requires. Gaussian broadening of atomic densities
  / distributions is the same device used by SOAP [Bartók 2013] and MBTR [Huo & Rupp 2022].
- **Intensivity.** Each channel is normalized to be *intensive* (per-atom), matching the
  intensive residual target and making the descriptor size-transferable — a standard
  requirement for energy-learning descriptors [Behler 2011; Musil 2021].

---

## 3. The four channels

### 3.1 `rdf` — element-pair partial radial distribution functions (2-body)

For each unordered pair of frozen elements (H–H, C–O, N–H, …), a Gaussian-broadened
histogram of the interatomic distances between atoms of those two elements, on a fixed
radial grid, divided by atom count.

- **Literature basis.** The radial distribution function `g(r)` is the classical two-body
  structural correlation function of liquid-state theory [Hansen & McDonald, *Theory of
  Simple Liquids*]. Its element-pair-resolved (*partial*) form `g_αβ(r)` as a machine-learning
  descriptor is due to Schütt et al. [2014]. As a Gaussian-broadened two-body distribution
  it is the k=2 term of the Many-Body Tensor Representation [Huo & Rupp 2022] and the radial
  part of the Behler–Parrinello G2 symmetry functions [Behler 2011].
- **What it captures.** Through-space proximity, including *non-bonded* contacts (H-bonds,
  steric contacts) that the bond graph omits entirely. This is the strongest single
  contributor to the channel's predictive power.
- **Design choices.** `r_max = 6 Å` (covers relevant non-bonded interactions; beyond this
  the intensive-energy signal is negligible), `n_rdf = 24` bins, `sigma_rdf = 0.2 Å`. The
  descriptor is the plain distance histogram of the original hybrid descriptor made
  *chemically resolved* — the single highest value-per-cost geometry upgrade.

### 3.2 `angle` — bond-angle histogram (3-body)

A Gaussian-broadened histogram over `[0°, 180°]` of every bond angle j–i–k where j and k
are covalent neighbours of a central atom i, per atom count. Connectivity comes from ASE
covalent-radius perception ([`build_graph`](wl_features.py), `cutoff_mult = 1.2`), the same
graph WL uses — so WL sees *that* a triple exists, and this channel sees its *geometry*.

- **Literature basis.** Angular distribution functions are the standard three-body term:
  the Behler–Parrinello G4/G5 angular symmetry functions [Behler & Parrinello 2007;
  Behler 2011] and the k=3 angle distributions of MBTR [Huo & Rupp 2022].
- **What it captures.** Hybridization and angular strain — the bent-vs-linear geometry the
  topology cannot express.
- **Design choices.** `n_angle = 18` bins (10° resolution), `sigma_angle = 5°`.
  Element-blind (the angle *value* carries the 3D information); an element-resolved variant
  was deliberately avoided to keep the dimensionality low.

### 3.3 `torsion` — dihedral-angle histogram (4-body)

A Gaussian-broadened histogram over `[0°, 180°]` of the **absolute** dihedral angle for
every bonded quadruple i–j–k–l about each covalent bond j–k, per atom count.

- **Literature basis.** The dihedral (torsion) angle is the four-body internal coordinate;
  four-body terms are the next order in the body-order expansion formalized by the Atomic
  Cluster Expansion [Drautz 2019]. Explicit torsion *distributions* are less common in
  general-purpose ML descriptors than 2- and 3-body terms, but the dihedral is the classical
  **rotamer coordinate** of conformational analysis — rotation about a bond *is* a change in
  torsion — which is exactly the conformational energy WL is blind to. Including it as a
  dedicated channel is a targeted design choice for this residual.
- **Design choice — `|dihedral|`.** We fold the sign by taking the absolute value, so that
  mirror-image (enantiomeric) conformers, which have equal energy, map to the same feature.
- **Parameters.** `n_torsion = 18` bins, `sigma_torsion = 10°`.

### 3.4 `elec` — electrostatics from Löwdin partial charges

Built from the per-atom Löwdin partial charges [Löwdin 1950]. The extensive mean removes
*net* charge, but not the *distribution* of partial charge, which carries substantial
internal electrostatic energy (e.g. hydrogen bonds and dipole interactions are
electrostatic). The channel packages this at several levels of detail:

- **Internal Coulomb sum** — `Σ_{i<j} q_i q_j / r_ij`, computed over the whole molecule and
  within a short-range cutoff (`r_short = 3 Å`), each per atom. This is a *classical
  point-charge proxy* for the internal electrostatic energy — a genuine, often dominant
  additive component of the total energy.
  - *Literature basis.* The pairwise `1/r_ij` construction is the Coulomb-Matrix idea
    [Rupp et al. 2012] and its Bag-of-Bonds variant [Hansen et al. 2015], but with **partial
    (Löwdin) charges** `q_i q_j` in place of nuclear charges `Z_i Z_j` — i.e. the descriptor
    approximates the *classical electrostatic interaction energy* used by point-charge force
    fields, rather than the nuclear-repulsion-like quantity of the original Coulomb Matrix.
- **Multipole magnitudes** — the magnitudes of the dipole and (traceless) quadrupole about
  the geometric centroid, per atom. Rotation-invariant summaries of the anisotropy of the
  charge distribution; standard molecular electrostatic descriptors.
- **Per-element charge moments** — the mean and standard deviation of the Löwdin charge on
  each frozen element (how electron-rich the O's run vs. the H's, etc.). A compact,
  chemically direct polarization fingerprint (species-resolved, in the spirit of §2.3).
- **Global charge spread** — the variance and range of the partial charges.

- **Design choice / caveat.** Löwdin charges are an *output* of the same DFT calculation
  that produces the energy. Using them as inputs is legitimate here because the OMol25
  records always provide them (the data loader admits only records with valid charges), but
  it means the descriptor assumes charges are available at prediction time — the same class
  of assumption as using forces. Empirically the `elec` channel is weak on its own but adds
  a genuine, orthogonal increment in combination with the structural channels.

---

## 4. Assembly and downstream use

Per molecule, the four channels are concatenated into one raw vector (≈ 558 dimensions for
`top_k = 6` elements: 21 element pairs × 24 RDF bins + 18 angle + 18 torsion + 18 electro-
static features). The `channel_slices()` method exposes each channel's column span so a
future kernel can split them (e.g. an additive geometry + charge kernel).

That vector is **dense** and low-dimensional, so — unlike the sparse WL matrix — it is
reduced by the same supervised **streaming PLS** ([`reduce.py`](reduce.py), natural /
N-invariant scaling) to a 10-D embedding, and fed to a compact-support **Wendland kernel**.
The Gaussian broadening (§2.4) is what makes "nearby distance/angle/torsion/charge
distributions" translate into "nearby embedding" and hence "similar predicted energy."

---

## 5. Relationship to established descriptors

This descriptor is intentionally a **lightweight, engineered** member of the geometric-
descriptor family rather than a heavyweight learned one:

- **vs. SOAP** [Bartók et al. 2013] and **ACSF** [Behler 2011]: those are per-atom,
  higher-dimensional, and capture angular information through a rotationally-averaged power
  spectrum / symmetry-function bank. Ours is a small global histogram set — cheaper and
  interpretable, at the cost of a lower representational ceiling (the geometry channel
  saturates around R² ≈ 0.5). SOAP/MBTR/ACSF are the natural richer replacements if the
  ceiling needs raising.
- **vs. MBTR** [Huo & Rupp 2022]: our `rdf` + `angle` channels are, in effect, a compact,
  element-pair-resolved subset of MBTR's k=2 and k=3 terms, plus an explicit 4-body torsion
  channel and a partial-charge electrostatics channel that standard MBTR does not include.
- **vs. Coulomb Matrix / Bag of Bonds** [Rupp et al. 2012; Hansen et al. 2015]: the `elec`
  channel adopts the pairwise `1/r` electrostatic construction but with partial charges,
  and only as one of four channels rather than the whole descriptor.

Reference implementations of SOAP, MBTR, and ACSF are collected in the DScribe library
[Himanen et al. 2020]; this module deliberately reimplements a minimal subset in NumPy (no
external descriptor dependency) so it can be shipped self-contained and parallelized at
scale.

---

## 6. Parameter summary

| Parameter | Value | Meaning |
|---|---|---|
| `top_k` | 6 | most-common elements kept (H, C, N, O, P, S on OMol25) |
| `r_max`, `n_rdf`, `sigma_rdf` | 6 Å, 24, 0.2 Å | partial-RDF radial grid + broadening |
| `n_angle`, `sigma_angle` | 18, 5° | bond-angle grid + broadening |
| `n_torsion`, `sigma_torsion` | 18, 10° | dihedral grid + broadening |
| `r_short` | 3 Å | short-range Coulomb cutoff |
| `cutoff_mult` | 1.2 | covalent-radius multiplier for bond perception |
| `charge_key` | `lowdin_charges` | per-atom partial-charge source |

Channels are individually switchable (`channels=("rdf","angle","torsion","elec")`) for
ablation.

---

## 7. References

1. J. Behler and M. Parrinello, "Generalized Neural-Network Representation of
   High-Dimensional Potential-Energy Surfaces," *Phys. Rev. Lett.* **98**, 146401 (2007).
2. J. Behler, "Atom-centered symmetry functions for constructing high-dimensional neural
   network potentials," *J. Chem. Phys.* **134**, 074106 (2011).
3. A. P. Bartók, R. Kondor, and G. Csányi, "On representing chemical environments,"
   *Phys. Rev. B* **87**, 184115 (2013). (SOAP)
4. H. Huo and M. Rupp, "Unified representation of molecules and crystals for machine
   learning," *Mach. Learn.: Sci. Technol.* **3**, 045017 (2022); arXiv:1704.06439. (MBTR)
5. R. Drautz, "Atomic cluster expansion for accurate and transferable interatomic
   potentials," *Phys. Rev. B* **99**, 014104 (2019). (ACE)
6. K. T. Schütt, H. Glawe, F. Brockherde, A. Sanna, K. R. Müller, and E. K. U. Gross,
   "How to represent crystal structures for machine learning: Towards fast prediction of
   electronic properties," *Phys. Rev. B* **89**, 205118 (2014). (partial RDF descriptor)
7. M. Rupp, A. Tkatchenko, K.-R. Müller, and O. A. von Lilienfeld, "Fast and Accurate
   Modeling of Molecular Atomization Energies with Machine Learning," *Phys. Rev. Lett.*
   **108**, 058301 (2012). (Coulomb Matrix)
8. K. Hansen, F. Biegler, R. Ramakrishnan, W. Pronobis, O. A. von Lilienfeld, K.-R. Müller,
   and A. Tkatchenko, "Machine Learning Predictions of Molecular Properties: Accurate
   Many-Body Potentials and Nonlocality in Chemical Space," *J. Phys. Chem. Lett.* **6**,
   2326 (2015). (Bag of Bonds)
9. P.-O. Löwdin, "On the Non-Orthogonality Problem Connected with the Use of Atomic Wave
   Functions in the Theory of Molecules and Crystals," *J. Chem. Phys.* **18**, 365 (1950).
   (Löwdin population analysis)
10. F. Musil, A. Grisafi, A. P. Bartók, C. Ortner, G. Csányi, and M. Ceriotti,
    "Physics-Inspired Structural Representations for Molecules and Materials," *Chem. Rev.*
    **121**, 9759–9815 (2021). (review of invariances and representations)
11. L. Himanen, M. O. J. Jäger, E. V. Morooka, F. Federici Canova, Y. S. Ranawat,
    D. Z. Gao, P. Rinke, and A. S. Foster, "DScribe: Library of descriptors for machine
    learning in materials science," *Comput. Phys. Commun.* **247**, 106949 (2020).
12. J.-P. Hansen and I. R. McDonald, *Theory of Simple Liquids*, 4th ed., Academic Press
    (2013). (radial distribution function)
