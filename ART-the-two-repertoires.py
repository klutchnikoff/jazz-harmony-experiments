"""Article data for Section 6.2, "The two repertoires".

Every figure the subsection states, and the claims it makes that carry no number.

  the filter        How many works each partition keeps, and what the filter is
                    worth against the Weimar Jazz Database, the one expert
                    annotation overlapping the jazz repertoire.

  the gaps          l1 distances between mean representations, at the level of
                    works: pooled, then within each annotated mode.  The mean
                    representation of a set A is the mean over its works of
                    Phi_p(S), not the duration-weighted mean over their chords;
                    the two differ, and the subsection uses the first.

  the permutation   Labels shuffled within a mode group, 20,000 times, the
                    statistic being the l1 distance between the two means.  The
                    per-mode test uses the maximum absolute difference across
                    the nine, which controls the family-wise rate without a
                    correction applied afterwards.

  the size control  Here the level changes, and the subsection says so: D_A(n)
                    is a share of duration and psi_A(n) a mean over degrees, so
                    the standardised comparison is between duration-weighted
                    means over chords and not between means over works.  Its raw
                    figures are therefore not the ones above.

Run:  LSA_LOCAL=1 .venv/bin/python ART-the-two-repertoires.py
"""
import collections
import re

import jams
import numpy as np
import pandas as pd

from article_data import export
from article_setup import cache_directory
from chord_scale import SYSTEM, MODES
from corpus import DATA, _annotation, key_exact, load_corpus
from leadsheetanalyser.constants import NOTE_TO_PC

ORDER = 0.15
PERMUTATIONS = 20_000
# Which modes lie on which side, group by group.  Not the same list twice: among
# major-key works the Aeolian gap runs the other way and the Mixolydian one does
# among minor-key works, both too small to survive the test, and an earlier draft
# of Section 6.2 stated one list for both.
SIDES = {
    "major": {"cp": ["Lydian", "Ionian"], "jazz": ["Mixolydian", "Dorian"]},
    "minor": {"cp": ["Lydian", "Ionian"],
              "jazz": ["Dorian", "Aeolian", "Phrygian"]},
}


def degree_reading(root, kind, cache={}):
    key = (root, kind)
    if key not in cache:
        content = {(root + i) % 12
                   for i in (0,) + tuple(j + 1 for j in range(11) if kind[j])}
        content |= {0}
        intervals = [i - 1 for i in range(1, 12) if i in content]
        m = np.mean(SYSTEM[:, intervals] ** ORDER, axis=1) ** (1 / ORDER)
        cache[key] = (len(intervals), m / m.sum())
    return cache[key]


def annotated_modes():
    out = {}
    for name in ("key_audit.csv", "common_practice_key_audit.csv"):
        table = pd.read_csv(cache_directory() / name)
        for song_id, annotated in zip(table["id"], table["annotated"]):
            minor = isinstance(annotated, str) and "min" in annotated.lower()
            out[str(song_id)] = "minor" if minor else "major"
    return out


def weimar_check():
    """Annotated-major charts the Weimar Jazz Database hears as minor.

    Before the filter and after it.  Weimar annotates performances rather than
    charts, so a tonic may differ for musical reasons; the mode is what carries
    over, and the mode is what is compared.
    """
    def norm(title):
        t = re.sub(r"^(the|a|an)\s+", "", str(title).lower().strip())
        return re.sub(r"[^a-z0-9]+", "", t)

    def mode_of(value):
        if ":" not in str(value):
            return None
        note, mode = str(value).split(":")[0], str(value).split(":")[1]
        if NOTE_TO_PC.get(note) is None:
            return None
        return "minor" if mode.strip().lower().startswith("min") else "major"

    meta = pd.read_csv(DATA / "meta.csv")
    expert = {}
    for _, row in meta[meta["id"].astype(str).str.startswith("weimar")].iterrows():
        try:
            j = jams.load(str(DATA / "jams_files" / f"{row['id']}.jams"),
                          validate=False)
        except Exception:
            continue
        keys = _annotation(j, "key_mode")
        if keys is None or not len(keys.data):
            continue
        m = mode_of(keys.data[0].value)
        if m:
            expert.setdefault(norm(row["title"]), m)

    audit = pd.read_csv(cache_directory() / "key_audit.csv")
    shared = kept = disagree_all = disagree_kept = 0
    for song_id, title, annotated, category in zip(
            audit["id"], audit["title"], audit["annotated"], audit["category"]):
        if not (isinstance(annotated, str) and "maj" in annotated.lower()):
            continue
        t = norm(title)
        if t not in expert:
            continue
        shared += 1
        disagree_all += expert[t] == "minor"
        if category == "exact":
            kept += 1
            disagree_kept += expert[t] == "minor"
    return shared, disagree_all, kept, disagree_kept


def main():
    songs, _titles, ids, _styles, n_jazz = load_corpus()
    keep = key_exact(ids)
    mode = annotated_modes()

    works = collections.defaultdict(list)
    chords = collections.defaultdict(lambda: [0.0, np.zeros(9)])
    for n, (song, k, song_id) in enumerate(zip(songs, keep, ids)):
        if not k:
            continue
        where = "J" if n < n_jazz else "C"
        m = mode.get(str(song_id), "major")
        total, weighted = 0.0, np.zeros(9)
        for (root, kind), duration in song:
            size, reading = degree_reading(root, kind)
            weighted += duration * reading
            total += duration
            cell = chords[(where, m, size)]
            cell[0] += duration
            cell[1] += duration * reading
        works[(where, m)].append(weighted / total)
    W = {k: np.array(v) for k, v in works.items()}

    pooled = {c: np.vstack([W[(c, "major")], W[(c, "minor")]]) for c in "JC"}
    gaps = {"pooled": np.abs(pooled["C"].mean(0) - pooled["J"].mean(0)).sum()}
    print(f"\n{'group':8s} {'jazz':>6s} {'cp':>5s} {'l1 gap':>8s} {'p':>9s}")
    print(f"{'pooled':8s} {len(pooled['J']):6d} {len(pooled['C']):5d} "
          f"{gaps['pooled']:8.4f}")

    rng = np.random.default_rng(0)
    beyond = {}
    for m in ("major", "minor"):
        A, B = W[("J", m)], W[("C", m)]
        observed = B.mean(0) - A.mean(0)
        gaps[m] = np.abs(observed).sum()
        both = np.vstack([A, B])
        null_l1 = np.empty(PERMUTATIONS)
        null_max = np.empty(PERMUTATIONS)
        for t in range(PERMUTATIONS):
            order = rng.permutation(len(both))
            d = both[order[len(A):]].mean(0) - both[order[:len(A)]].mean(0)
            null_l1[t] = np.abs(d).sum()
            null_max[t] = np.abs(d).max()
        p = (null_l1 >= gaps[m]).mean()
        beyond[m] = int(sum((null_max >= abs(observed[j])).mean() < 0.05
                            for j in range(9)))
        print(f"{m:8s} {len(A):6d} {len(B):5d} {gaps[m]:8.4f} {p:9.4g}"
              f"   {beyond[m]} of nine modes beyond the family-wise 5%")
        assert p < 1 / PERMUTATIONS, f"the {m} gap is now reachable by chance"
        for name in SIDES[m]["cp"]:
            assert observed[MODES.index(name)] > 0, (
                f"the common practice no longer carries the more {name} mass "
                f"among {m}-key works")
        for name in SIDES[m]["jazz"]:
            assert observed[MODES.index(name)] < 0, (
                f"the jazz no longer carries the more {name} mass among "
                f"{m}-key works")

    # the size control, at the level of chords and not of works
    print()
    standardised = {}
    for m in ("major", "minor"):
        sizes = [n for n in range(1, 12)
                 if chords[("J", m, n)][0] > 0 and chords[("C", m, n)][0] > 0]
        mean = {c: sum(chords[(c, m, n)][1] for n in range(1, 12))
                / sum(chords[(c, m, n)][0] for n in range(1, 12)) for c in "JC"}
        share = np.array([chords[("J", m, n)][0] for n in sizes])
        share = share / share.sum()
        under_jazz = sum(w * chords[("C", m, n)][1] / chords[("C", m, n)][0]
                         for w, n in zip(share, sizes))
        raw = np.abs(mean["C"] - mean["J"]).sum()
        std = np.abs(under_jazz - mean["J"]).sum()
        standardised[m] = (raw, std)
        print(f"   {m:8s} chord-level gap {raw:.6f}, "
              f"under the jazz size mix {std:.6f}")
        assert std > raw, (
            f"standardising no longer widens the {m} gap, so size no longer "
            "works against the difference as Section 6.2 says")

    shared, disagree_all, kept, disagree_kept = weimar_check()
    print(f"\nWeimar: {disagree_all} of {shared} annotated-major charts heard as "
          f"minor, {disagree_kept} of {kept} after the filter")

    minor_share = {c: 100 * len(W[(c, "minor")]) / len(pooled[c]) for c in "JC"}
    export("the-two-repertoires", {
        # the checker reads the manuscript with LaTeX digit grouping stripped
        "jazz_kept": f"{len(pooled['J'])} lead sheets",
        "cp_kept": f"{len(pooled['C'])} works",
        "weimar_before": f"{disagree_all} of {shared}",
        "weimar_after": f"{disagree_kept} of {kept}",
        "gap_pooled": f"{gaps['pooled']:.3f}",
        "gap_major": f"{gaps['major']:.3f}",
        "gap_minor": f"{gaps['minor']:.3f}",
        "minor_share_cp": f"{minor_share['C']:.0f}",
        "minor_share_jazz": f"{minor_share['J']:.0f}",
        "size_raw_major": f"{standardised['major'][0]:.3f}",
        "size_std_major": f"{standardised['major'][1]:.3f}",
        "size_raw_minor": f"{standardised['minor'][0]:.3f}",
        "size_std_minor": f"{standardised['minor'][1]:.3f}",
        "beyond_major": f"{beyond['major']} of the nine",
        "beyond_minor": f"{beyond['minor']} of the nine",
    })


if __name__ == "__main__":
    main()
