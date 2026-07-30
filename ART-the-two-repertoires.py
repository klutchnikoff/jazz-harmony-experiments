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
                    Monte Carlo p-value includes the observed labelling and is
                    therefore never zero.  Bonferroni correction covers the two
                    key-type comparisons.  Within each, the coordinate tests
                    use the maximum absolute difference across the nine; the
                    two corrections together control the overall family-wise
                    rate under the joint null.

  the size control  Here the level changes, and the subsection says so: chord
                    size is the actual pitch-class cardinality 1 + |k|, not the
                    size |k^q| of the tonic-relative input to Phi.  rho is a
                    share of duration and nu a duration-weighted conditional
                    mean over chords.  Both repertoires are standardized to the
                    common-practice size distribution on their shared support.
                    The only common-practice size outside that support is a
                    negligible share of the minor-key duration.  Inference
                    still permutes whole works, not chords, and recomputes the
                    complete standardization after every reallocation.

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
KEY_TYPE_TESTS = 2
# Which modes lie on which side, group by group.  Not the same list twice: among
# major-key works the Aeolian gap runs the other way and the Mixolydian one does
# among minor-key works, both too small to survive the test, and an earlier draft
# of Section 6.2 stated one list for both.
SIDES = {
    "major": {"cp": ["Lydian", "Ionian"], "jazz": ["Mixolydian", "Dorian"]},
    "minor": {"cp": ["Lydian", "Ionian"],
              "jazz": ["Dorian", "Aeolian", "Phrygian"]},
}

# One generator per key type, not one consumed in sequence: with a single stream
# the minor-key null depends on how many draws the major-key one took, so an
# edit anywhere above it would silently move a figure this subsection states.
# The l1 and max-T nulls deliberately share their reallocations, both being read
# off the same shuffles.
SEED = 20260729
KEY_TYPES = {"major": 0, "minor": 1}


def stream(kappa):
    return np.random.Generator(
        np.random.PCG64([SEED, KEY_TYPES[kappa]])
    )


def size_stream(kappa):
    """An independent stream for the chord-size permutation statistic.

    Stream identifiers 1--3 are used by ART-the-brightness-contrast.py; 4 is
    reserved here so that no two stochastic computations share a bit stream.
    """
    return np.random.Generator(
        np.random.PCG64([SEED, 4, KEY_TYPES[kappa]])
    )


def english_list(names):
    """Join the coordinate names exactly as the manuscript states them."""
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def degree_reading(root, kind, cache={}):
    key = (root, kind)
    if key not in cache:
        content = {(root + i) % 12
                   for i in (0,) + tuple(j + 1 for j in range(11) if kind[j])}
        content |= {0}
        intervals = [i - 1 for i in range(1, 12) if i in content]
        if not intervals:
            raise ValueError("Phi_p(0) is undefined")
        m = np.mean(SYSTEM[:, intervals] ** ORDER, axis=1) ** (1 / ORDER)
        cache[key] = m / m.sum()
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


def size_statistics(jazz, common):
    """Raw and size-standardized gaps from work-level chord summaries.

    Each array has one row per work, one cell per chord size, and ten columns
    per cell: duration followed by its nine duration-weighted modal sums.
    Keeping works separate here is what lets the permutation test preserve the
    dependence among chords from the same work.
    """
    cells = {"J": jazz.sum(axis=0), "C": common.sum(axis=0)}
    duration = {c: cells[c][:, 0] for c in "JC"}
    support = {
        c: np.flatnonzero(duration[c] > 0).tolist()
        for c in "JC"
    }
    shared = sorted(set(support["C"]) & set(support["J"]))
    if not shared:
        raise ValueError("the two groups have no shared chord size")

    mean = {
        c: cells[c][:, 1:].sum(axis=0) / duration[c].sum()
        for c in "JC"
    }
    mean_size = {
        c: np.arange(len(duration[c])) @ duration[c] / duration[c].sum()
        for c in "JC"
    }
    excluded = (
        duration["C"][list(set(support["C"]) - set(shared))].sum()
        / duration["C"].sum()
    )
    target = duration["C"][shared]
    target = target / target.sum()
    adjusted = {
        c: sum(
            weight * cells[c][size, 1:] / duration[c][size]
            for weight, size in zip(target, shared)
        )
        for c in "JC"
    }
    return {
        "support": support,
        "shared": shared,
        "mean_size": mean_size,
        "excluded": excluded,
        "raw": np.abs(mean["C"] - mean["J"]).sum(),
        "standardized": np.abs(adjusted["C"] - adjusted["J"]).sum(),
    }


def main():
    songs, _titles, ids, _styles, n_jazz = load_corpus()
    keep = key_exact(ids)
    mode = annotated_modes()

    works = collections.defaultdict(list)
    size_works = collections.defaultdict(list)
    for n, (song, k, song_id) in enumerate(zip(songs, keep, ids)):
        if not k:
            continue
        where = "J" if n < n_jazz else "C"
        m = mode.get(str(song_id), "major")
        total, weighted = 0.0, np.zeros(9)
        by_size = np.zeros((13, 10))
        for (root, kind), duration in song:
            reading = degree_reading(root, kind)
            size = 1 + sum(kind)
            weighted += duration * reading
            total += duration
            by_size[size, 0] += duration
            by_size[size, 1:] += duration * reading
        works[(where, m)].append(weighted / total)
        size_works[(where, m)].append(by_size)
    W = {k: np.array(v) for k, v in works.items()}
    H = {k: np.array(v) for k, v in size_works.items()}

    pooled = {c: np.vstack([W[(c, "major")], W[(c, "minor")]]) for c in "JC"}
    gaps = {"pooled": np.abs(pooled["C"].mean(0) - pooled["J"].mean(0)).sum()}
    print(f"\n{'group':8s} {'jazz':>6s} {'cp':>5s} {'l1 gap':>8s} {'p':>9s}")
    print(f"{'pooled':8s} {len(pooled['J']):6d} {len(pooled['C']):5d} "
          f"{gaps['pooled']:8.4f}")

    beyond = {}
    for m in ("major", "minor"):
        shuffle = stream(m)
        A, B = W[("J", m)], W[("C", m)]
        observed = B.mean(0) - A.mean(0)
        gaps[m] = np.abs(observed).sum()
        both = np.vstack([A, B])
        null_l1 = np.empty(PERMUTATIONS)
        null_max = np.empty(PERMUTATIONS)
        for t in range(PERMUTATIONS):
            order = shuffle.permutation(len(both))
            d = both[order[len(A):]].mean(0) - both[order[:len(A)]].mean(0)
            null_l1[t] = np.abs(d).sum()
            null_max[t] = np.abs(d).max()
        exceedances = int(np.count_nonzero(null_l1 >= gaps[m]))
        p = min(
            KEY_TYPE_TESTS * (exceedances + 1) / (PERMUTATIONS + 1),
            1,
        )
        adjusted = [
            min(
                KEY_TYPE_TESTS
                * (1 + int(np.count_nonzero(null_max >= abs(observed[j]))))
                / (PERMUTATIONS + 1),
                1,
            )
            for j in range(9)
        ]
        retained = {MODES[j] for j, value in enumerate(adjusted) if value <= 0.05}
        expected = set(SIDES[m]["cp"]) | set(SIDES[m]["jazz"])
        assert retained == expected, (
            f"the family-wise 5% coordinates among {m}-key works are now "
            f"{sorted(retained)}, not {sorted(expected)}")
        beyond[m] = len(retained)
        print(f"{m:8s} {len(A):6d} {len(B):5d} {gaps[m]:8.4f} {p:9.4g}"
              f"   {beyond[m]} of nine modes beyond the overall family-wise 5%")
        assert exceedances == 0, f"the {m} gap is now reached by a permutation"
        for name in SIDES[m]["cp"]:
            assert observed[MODES.index(name)] > 0, (
                f"the common practice no longer carries the more {name} mass "
                f"among {m}-key works")
        for name in SIDES[m]["jazz"]:
            assert observed[MODES.index(name)] < 0, (
                f"the jazz no longer carries the more {name} mass among "
                f"{m}-key works")

    # The size control, at the level of chords and not of works.  Standardize
    # both repertoires to the common-practice size distribution on the shared
    # support.  This avoids both extrapolation and the asymmetry of comparing a
    # standardized jazz mean with an unstandardized common-practice one.
    print()
    standardised = {}
    excluded = {}
    size_supports = {}
    size_p = {}
    for m in ("major", "minor"):
        J, C = H[("J", m)], H[("C", m)]
        observed = size_statistics(J, C)
        size_supports[m] = observed["support"]
        excluded[m] = observed["excluded"]
        assert observed["mean_size"]["J"] > observed["mean_size"]["C"], (
            f"jazz no longer has the larger duration-weighted mean chord size "
            f"among {m}-key works")
        raw = observed["raw"]
        std = observed["standardized"]
        standardised[m] = (raw, std)

        both = np.concatenate([J, C])
        shuffle = size_stream(m)
        null = np.empty(PERMUTATIONS)
        for t in range(PERMUTATIONS):
            order = shuffle.permutation(len(both))
            null[t] = size_statistics(
                both[order[:len(J)]],
                both[order[len(J):]],
            )["standardized"]
        exceedances = int(np.count_nonzero(null >= std))
        p = min(
            KEY_TYPE_TESTS * (1 + exceedances) / (PERMUTATIONS + 1),
            1,
        )
        size_p[m] = (
            f"{KEY_TYPE_TESTS}/{PERMUTATIONS + 1}"
            if exceedances == 0 else f"{p:.4f}"
        )
        print(f"   {m:8s} chord-level gap {raw:.6f}, "
              f"standardized on sizes {observed['shared']} {std:.6f}, "
              f"permutation p = {size_p[m]}; common-practice duration "
              f"excluded {100 * excluded[m]:.4f}%")
        assert std > raw, (
            f"standardising no longer widens the {m} gap, so size no longer "
            "works against the difference as Section 6.2 says")

    assert size_supports["major"] == {
        "J": list(range(2, 9)), "C": list(range(2, 7))
    }, "the major-key chord-size supports have changed"
    assert size_supports["minor"] == {
        "J": list(range(2, 8)), "C": list(range(1, 6))
    }, "the minor-key chord-size supports have changed"
    assert excluded["major"] == 0
    assert round(100 * excluded["minor"], 3) == 0.017, (
        "the excluded minor-key common-practice duration no longer rounds to "
        "0.017%")

    shared, disagree_all, kept, disagree_kept = weimar_check()
    print(f"\nWeimar: {disagree_all} of {shared} annotated-major charts heard as "
          f"minor, {disagree_kept} of {kept} after the filter")

    minor_share = {c: 100 * len(W[(c, "minor")]) / len(pooled[c]) for c in "JC"}
    export("the-two-repertoires", {
        # the checker reads the manuscript with LaTeX digit grouping stripped
        "jazz_kept":
            f"{len(pooled['J'])} of the {n_jazz} jazz lead sheets",
        "cp_kept":
            f"{len(pooled['C'])} of the {len(songs) - n_jazz} "
            "common-practice works",
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
        "size_test":
            f"both Monte Carlo p-values are "
            f"{KEY_TYPE_TESTS}/{PERMUTATIONS + 1}",
        "size_support_major_cp": "sizes 2 to 6",
        "size_support_major_jazz": "sizes 2 to 8",
        "size_support_minor_cp": "sizes 1 to 5",
        "size_support_minor_jazz": "sizes 2 to 7",
        "size_excluded_minor":
            f"accounts for only {100 * excluded['minor']:.3f}",
        "retained_major":
            f"{english_list(SIDES['major']['cp'])} in the common-practice "
            f"direction, and {english_list(SIDES['major']['jazz'])} in the "
            "jazz direction",
        "retained_minor":
            f"{english_list(SIDES['minor']['cp'])} in the common-practice "
            f"direction, and {english_list(SIDES['minor']['jazz'])} in the "
            "jazz direction",
        "permutation_floor": f"{KEY_TYPE_TESTS}/{PERMUTATIONS + 1}",
    })


if __name__ == "__main__":
    main()
