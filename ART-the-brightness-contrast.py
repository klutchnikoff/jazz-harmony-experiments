"""Article data for Section 6.3, "The brightness contrast".

Section 6.2 measures how far apart the two mean representations are.  It does not
say in which direction, and an l1 distance cannot: it is nonnegative by
construction, so its null is rejected as soon as the two repertoires use
different chords, which is certain in advance.  This subsection tests a signed
claim instead, along the one ordering the modal system already carries.

  the ordering    Section 3.2 fixes it, Lydian to Locrian, before any corpus is
                  read.  Nothing here chooses it, and no scale is placed on it.

  the claim       For every k, the common-practice mean puts at least as much
                  mass on the k brightest modes as the jazz mean does.  Writing
                  d_j for the difference of mean masses, jazz minus common
                  practice, over the seven diatonic modes renormalised within
                  each work, and T_i = -sum_{j<=i} d_j, the claim is T_i >= 0
                  for i = 1..6.  T_7 vanishes, both being distributions.

  why it matters  For any increasing weights g_1 < ... < g_7, Abel summation
                  gives sum_j g_j d_j = sum_i (g_{i+1} - g_i) T_i, so the claim
                  makes the sign of the contrast independent of every scale one
                  might place on the modes.  The script checks that identity
                  numerically as well as the inequality.

  the magnitude   Reported on one such weighting, the ranks 1..7, as a
                  difference of means with a bootstrap interval and an effect
                  size in pooled standard deviations.  The ranks carry the
                  magnitude only; the inequality above carries the direction.

The two levels of evidence differ, and Section 6.3 says so: the inequality holds
exactly on the sample, whereas its permutation test is strong among major-key
works and merely significant among minor-key ones, where it rests on far fewer
works and is bound by the Locrian tail.

Run:  LSA_LOCAL=1 .venv/bin/python ART-the-brightness-contrast.py
"""
import collections

import numpy as np
import pandas as pd

from article_data import export
from article_setup import cache_directory
from chord_scale import MODES, SYSTEM
from corpus import key_exact, load_corpus

ORDER = 0.15
PERMUTATIONS = 20_000
BOOTSTRAP = 20_000
DIATONIC = 7                      # the seven rows Section 3.2 orders
RANKS = np.arange(1, DIATONIC + 1, dtype=float)

# One generator per (purpose, key type) rather than one consumed in sequence.
# A single stream would make every number depend on how many draws the code
# before it happened to take, so adding a check or changing a replication count
# would silently move a figure the manuscript states.  These six are
# independent, and each reproduces on its own.
SEED = 20260729
STREAMS = {"abel": 1, "bootstrap": 2, "permutation": 3}
KEY_TYPES = {"major": 0, "minor": 1}


def stream(purpose, kappa):
    return np.random.default_rng([SEED, STREAMS[purpose], KEY_TYPES[kappa]])


def annotated_modes():
    out = {}
    for name in ("key_audit.csv", "common_practice_key_audit.csv"):
        table = pd.read_csv(cache_directory() / name)
        for song_id, annotated in zip(table["id"], table["annotated"]):
            minor = isinstance(annotated, str) and "min" in annotated.lower()
            out[str(song_id)] = "minor" if minor else "major"
    return out


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


def profiles():
    """Per work, its reading renormalised over the seven diatonic modes.

    Renormalising within each work and then averaging, not the reverse: the
    brightness of a work is a property of that work, and Section 6.3 averages
    brightnesses rather than reading one off an average.
    """
    songs, _titles, ids, _styles, n_jazz = load_corpus()
    keep = key_exact(ids)
    mode = annotated_modes()
    works = collections.defaultdict(list)
    for n, (song, kept, song_id) in enumerate(zip(songs, keep, ids)):
        if not kept:
            continue
        total, weighted = 0.0, np.zeros(len(MODES))
        for (root, kind), duration in song:
            weighted += duration * degree_reading(root, kind)
            total += duration
        diatonic = (weighted / total)[:DIATONIC]
        works[("J" if n < n_jazz else "C",
               mode.get(str(song_id), "major"))].append(diatonic
                                                        / diatonic.sum())
    return {k: np.array(v) for k, v in works.items()}


def cuts(jazz, common):
    """T_i for i = 1..7, the last being zero."""
    return -np.cumsum(jazz.mean(0) - common.mean(0))


def brightness(P):
    return P @ RANKS


def main():
    W = profiles()
    values = {}

    for kappa in ("major", "minor"):
        J, C = W[("J", kappa)], W[("C", kappa)]

        # every per-work profile is a distribution over the seven modes
        assert np.allclose(J.sum(1), 1) and np.allclose(C.sum(1), 1), (
            "a work profile is not a distribution over the seven diatonic modes")

        T = cuts(J, C)
        assert abs(T[-1]) < 1e-12, "the two profiles do not both sum to one"
        binding = int(np.argmin(T[:DIATONIC - 1]))
        assert T[:DIATONIC - 1].min() > 0, (
            f"among {kappa}-key works the bright end is no longer ahead at "
            f"every cut point")
        assert binding == DIATONIC - 2, (
            f"among {kappa}-key works the Locrian tail no longer binds the "
            f"inequality; cut {binding + 1} does")

        # the Abel identity Section 6.3 states, on a random increasing weighting
        d = J.mean(0) - C.mean(0)
        g = np.sort(stream("abel", kappa).uniform(0, 10, DIATONIC))
        assert np.isclose(g @ d, np.diff(g) @ T[:DIATONIC - 1]), (
            "the Abel summation identity does not hold numerically")

        a, b = brightness(J), brightness(C)
        contrast = a.mean() - b.mean()
        assert contrast > 0, (
            f"among {kappa}-key works the jazz mean is no longer the darker")

        pooled = np.concatenate([a, b])
        draw = stream("bootstrap", kappa)
        boot = np.array([draw.choice(a, len(a), True).mean()
                         - draw.choice(b, len(b), True).mean()
                         for _ in range(BOOTSTRAP)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        assert lo > 0, (
            f"among {kappa}-key works the contrast interval now contains zero")
        effect = contrast / pooled.std(ddof=1)

        # permutation test on the binding cut point
        both = np.vstack([J, C])
        shuffle = stream("permutation", kappa)
        null = np.empty(PERMUTATIONS)
        for t in range(PERMUTATIONS):
            order = shuffle.permutation(len(both))
            null[t] = cuts(both[order[:len(J)]],
                           both[order[len(J):]])[:DIATONIC - 1].min()
        # A Monte Carlo p-value is never zero, and printing 0.0000 would say it
        # is.  When no reallocation reaches the observed value, report the floor
        # of the design as Section 6.2 already does.
        # The statistic asks for all six inequalities at once, so a null
        # reallocation typically fails at least one of them and the null median
        # is negative.  Section 6.3 says so, hence the check.
        assert np.median(null) < 0, (
            f"the {kappa}-key null median is no longer negative, so the "
            "statistic is no longer the demanding one Section 6.3 describes")
        exceedances = int(np.count_nonzero(null >= T[:DIATONIC - 1].min()))
        p = (1 + exceedances) / (PERMUTATIONS + 1)
        p_text = (f"1/{PERMUTATIONS + 1}" if exceedances == 0
                  else f"{p:.4f}")

        print(f"\n=== {kappa} keys ===  jazz {len(J)}, common practice {len(C)}")
        print(f"{'mode':12s} {'jazz':>8s} {'cp':>8s} {'d':>9s} {'T_i':>9s}")
        for i, name in enumerate(MODES[:DIATONIC]):
            print(f"{name:12s} {J.mean(0)[i]:8.4f} {C.mean(0)[i]:8.4f} "
                  f"{d[i]:+9.4f} {T[i]:+9.4f}")
        print(f"  binding cut {binding + 1} ({MODES[binding]}/"
              f"{MODES[binding + 1]}), min T_i = {T[:DIATONIC - 1].min():.4f}, "
              f"permutation p = {p_text}")
        print(f"  brightness {a.mean():.3f} against {b.mean():.3f}, contrast "
              f"{contrast:+.3f} [{lo:.3f}, {hi:.3f}], {effect:.2f} sd")

        values[f"dominance_{kappa}"] = f"{T[:DIATONIC - 1].min():.4f}"
        values[f"dominance_p_{kappa}"] = p_text
        values[f"contrast_{kappa}"] = f"{contrast:.3f}"
        values[f"interval_{kappa}"] = f"[{lo:.3f}, {hi:.3f}]"
        values[f"effect_{kappa}"] = f"{effect:.2f} standard deviations"

    export("the-brightness-contrast", values)


if __name__ == "__main__":
    main()
