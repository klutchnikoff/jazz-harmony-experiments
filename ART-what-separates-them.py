"""Article data for Section 6.3, "What separates them".

The gap of Section 6.2 decomposes over degrees, exactly.  Writing D_A(d) for the
share of a set's duration spent on the degree d,

    sum_d D_A(d) Phi_p(d)

is the duration-weighted mean reading of A, so the gap between two sets is

    sum_d ( D_C(d) - D_J(d) ) Phi_p(d),

and each degree contributes ( D_C(d) - D_J(d) ) [Phi_p(d)]_j to the gap on the
mode j.  The contributions sum to the gap by construction, which the script
checks rather than trusts.

This is the chord level, as the size control of Section 6.2 is: D_A is a share of
duration over pooled chords, not a mean over works.  The gap it decomposes is
therefore 0.102 among major-key works and not the 0.093 of the work-level
comparison.

Run:  LSA_LOCAL=1 .venv/bin/python ART-what-separates-them.py
"""
import collections

import numpy as np
import pandas as pd

from article_data import export
from article_setup import cache_directory
from chord_scale import SYSTEM, MODES
from corpus import key_exact, load_corpus

ORDER = 0.15
ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII",
         "VII"]


def degree_reading(root, kind, cache={}):
    key = (root, kind)
    if key not in cache:
        content = {(root + i) % 12
                   for i in (0,) + tuple(j + 1 for j in range(11) if kind[j])}
        content |= {0}
        intervals = [i - 1 for i in range(1, 12) if i in content]
        m = np.mean(SYSTEM[:, intervals] ** ORDER, axis=1) ** (1 / ORDER)
        cache[key] = m / m.sum()
    return cache[key]


def label(root, kind):
    """The degree named as harmony names it: numeral, quality, seventh."""
    third = "m" if kind[2] else ("M" if kind[3] else "")
    fifth = "o" if (kind[5] and not kind[6]) else ("+" if kind[7] and not kind[6]
                                                   else "")
    seventh = "7" if kind[9] else ("maj7" if kind[10] else "")
    return f"{ROMAN[root % 12]}{third}{fifth}{seventh}"


def annotated_modes():
    out = {}
    for name in ("key_audit.csv", "common_practice_key_audit.csv"):
        table = pd.read_csv(cache_directory() / name)
        for song_id, annotated in zip(table["id"], table["annotated"]):
            minor = isinstance(annotated, str) and "min" in annotated.lower()
            out[str(song_id)] = "minor" if minor else "major"
    return out


def main():
    songs, _titles, ids, _styles, n_jazz = load_corpus()
    keep = key_exact(ids)
    mode = annotated_modes()

    share = {"J": collections.defaultdict(float), "C": collections.defaultdict(float)}
    total = {"J": 0.0, "C": 0.0}
    reading = {}
    for n, (song, k, song_id) in enumerate(zip(songs, keep, ids)):
        if not k or mode.get(str(song_id), "major") != "major":
            continue
        where = "J" if n < n_jazz else "C"
        for (root, kind), duration in song:
            # keyed by the degree itself: two kinds may wear one name, C and C6
            # both reading as "IM", and they do not have one reading
            share[where][(root % 12, kind)] += duration
            total[where] += duration
            reading.setdefault((root % 12, kind), degree_reading(root, kind))
    for where in "JC":
        for name in share[where]:
            share[where][name] /= total[where]

    names = set(share["J"]) | set(share["C"])
    delta = {d: share["C"][d] - share["J"][d] for d in names}
    gap = sum(delta[d] * reading[d] for d in names)

    # the identity the decomposition rests on, checked and not assumed
    direct = (sum(share["C"][d] * reading[d] for d in names)
              - sum(share["J"][d] * reading[d] for d in names))
    assert np.abs(gap - direct).max() < 1e-12, "the decomposition does not sum"
    assert abs(np.abs(gap).sum() - 0.1019) < 5e-4, (
        "the chord-level gap among major-key works has moved from the 0.102 "
        "Section 6.2 states")

    values = {}
    for target in ("Ionian", "Dorian"):
        j = MODES.index(target)
        contribution = {d: delta[d] * reading[d][j] for d in names}
        ranked = sorted(contribution, key=lambda d: -abs(contribution[d]))[:4]
        print(f"\ncontributions to the {target} gap of {gap[j]:+.4f}, "
              f"major-key works")
        print(f"   {'degree':10s} {'jazz':>7s} {'cp':>7s} {'delta':>8s} "
              f"{'reads':>7s} {'contributes':>12s}")
        for d in ranked:
            print(f"   {label(*d):10s} {share['J'][d]:7.3f} {share['C'][d]:7.3f} "
                  f"{delta[d]:+8.3f} {reading[d][j]:7.3f} "
                  f"{contribution[d]:+12.4f}")

    # named by their exact kind and not by the label, which ignores the sixth
    # and would let V and V6 answer to one name
    def kind_of(*intervals):
        return tuple(1 if i in intervals else 0 for i in range(1, 12))

    wanted = {"VM": (7, kind_of(4, 7)),
              "IIm7": (2, kind_of(3, 7, 10)),
              "IMmaj7": (0, kind_of(4, 7, 11))}
    for name, d in wanted.items():
        assert d in names, f"the degree {name} is no longer in the corpus"
        assert label(*d) == name, f"{name} is not what label() calls it"
        values[f"jazz_{name}"] = f"{100 * share['J'][d]:.1f}"
        values[f"cp_{name}"] = f"{100 * share['C'][d]:.1f}"

    ionian, dorian = MODES.index("Ionian"), MODES.index("Dorian")
    v, ii = wanted["VM"], wanted["IIm7"]
    values["gap_ionian"] = f"{gap[ionian]:.3f}"
    values["VM_reads_ionian"] = f"{reading[v][ionian]:.2f}"
    values["VM_contributes"] = f"{delta[v] * reading[v][ionian]:.3f}"
    values["gap_dorian"] = f"{gap[dorian]:.3f}"
    values["IIm7_reads_dorian"] = f"{reading[ii][dorian]:.2f}"
    values["IIm7_contributes"] = f"{delta[ii] * reading[ii][dorian]:.3f}"
    export("what-separates-them", values)


if __name__ == "__main__":
    main()
