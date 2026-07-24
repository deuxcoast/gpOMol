"""
sources.py  (data_exploration)
==============================
Parse the OMol25 ``atoms.info["source"]`` provenance path into the pieces that
make redundancy measurable.

A source looks like ``<top>/<dirs...>/<job>[/stepK]/orca.tar.zst``, e.g.

    orbnet_denali/orbnet_CHEMBL1935278_conformers_<sha>_675082_0_1/orca.tar.zst
    omol/electrolytes/md_based/outputs_241029/3015_CF3O3S-1_3_group_96_shell_296_0_5/step1/orca.tar.zst
    trans1x/t1x_rxn0444_1154_551_0_1/orca.tar.zst

We extract four levels, coarse to fine:

``top``     first path component -- the upstream dataset (``omol``, ``ani1xbb``, ...).
``group``   every directory above the job -- the production batch
            (``omol/electrolytes/md_based/outputs_241029``).
``job``     the job directory itself, minus a trailing ``stepK`` -- one ORCA
            calculation series, i.e. ONE structure sampled one or more times.
``family``  the job with its per-conformer / per-frame / charge-spin suffixes
            stripped -- our best guess at "the same molecule".

Only ``top``/``group``/``job``/``step`` are read directly off the path and are
therefore exact. ``family`` is a *heuristic*: the naming conventions differ per
upstream dataset and are not documented. It is validated downstream in
``families.py`` by composition purity (members of a real family must share a
chemical formula), and every ceiling number is additionally reported against the
parse-free grouping ``(formula, charge, spin)`` so no conclusion rests on the
heuristic alone.
"""

from __future__ import annotations

import hashlib
import re

# job suffixes that identify a member WITHIN a family rather than the family
_STEP = re.compile(r"^step(\d+)$")
_TRAIL_INT = re.compile(r"_\d+$")

# per-upstream-dataset rules: first group of the match is the family key
_FAMILY_RULES = (
    # orbnet_denali: <mol>_conformers_<sha>_<confid>_<charge>_<spin>
    ("orbnet_denali", re.compile(r"^(orbnet_[A-Za-z0-9]+)_conformers_")),
    # trans1x: t1x_rxn0444_1154_551 -- frames along one reaction
    ("trans1x", re.compile(r"^(t1x_rxn\d+)_")),
    # ml_mo: mo_md_mace_mp_43279_0_1_1_406 -- MD frames of one complex
    ("ml_mo", re.compile(r"^(mo_md_\w+?_\d+)_")),
)


def _blake(s: str) -> int:
    """Stable 64-bit hash. (Python's ``hash`` is per-process randomised and so
    cannot be used to join rows produced by different worker processes.)"""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def parse_source(src: str, charge: int, spin: int):
    """``source`` -> ``(top, group, job, family, step)``; ``step`` is -1 if absent."""
    if not src:
        return "", "", "", "", -1
    parts = [p for p in src.split("/") if p]
    if parts and "." in parts[-1]:          # drop the trailing file (orca.tar.zst)
        parts = parts[:-1]
    step = -1
    if parts:
        m = _STEP.match(parts[-1])
        if m:
            step = int(m.group(1))
            parts = parts[:-1]
    if not parts:
        return "", "", "", "", step
    top = parts[0]
    job = parts[-1]
    group = "/".join(parts[:-1])
    return top, group, job, _family_key(top, job, charge, spin), step


def _family_key(top: str, job: str, charge: int, spin: int) -> str:
    """Strip the within-family suffixes from a job name.

    Generic pass: a trailing ``_<charge>_<spin>`` (ubiquitous across OMol25 job
    names) and then a trailing integer (a frame / conformer index). Datasets whose
    convention we can read off the name get an explicit rule instead.
    """
    for name, rule in _FAMILY_RULES:
        if top == name:
            m = rule.match(job)
            if m:
                return f"{top}/{m.group(1)}"
            break
    key = job
    tail = f"_{int(charge)}_{int(spin)}"
    if key.endswith(tail) and len(key) > len(tail):
        key = key[: -len(tail)]
    else:
        # some names carry a frame index between the id and the charge/spin pair
        stripped = _TRAIL_INT.sub("", key)
        if stripped.endswith(tail) and len(stripped) > len(tail):
            key = stripped[: -len(tail)]
    return f"{top}/{key}"


def hash_source(src: str, charge: int, spin: int):
    """``parse_source`` with the four string levels hashed to ``uint64``.

    Returns ``(top, group, job_hash, family_hash, step)`` -- ``top`` and ``group``
    stay strings (low cardinality, kept as name tables), while ``job``/``family``
    have millions of distinct values and are only ever used for grouping.
    """
    top, group, job, family, step = parse_source(src, charge, spin)
    return top, group, _blake(job), _blake(family), step
