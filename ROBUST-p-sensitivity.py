"""Sensitivity of the corpus conclusions to nearby admissible orders p.

The grid is fixed before the results are read: every hundredth from 0.14, just
above the K_0 separation crossing at 0.1393, through 0.20, the next round order
discussed in Section 4.  No corpus result selects an endpoint or an interior
point.

For each order and annotated key type, this checks the two main corpus claims:

* the nine-mode work means differ (l1 permutation test and max-T coordinates);
* the common-practice mean dominates at all six cuts of the ordered diatonic
  profile (minimum cut, its permutation test, and a rank-scale bootstrap
  interval for the corresponding brightness contrast).

The random streams restart from the same seed at every p, so each order sees
the same label reallocations and bootstrap indices.  Results are written to
analysis-data/, not article-data/: they are versioned analysis provenance but
are not claimed by the manuscript until explicitly promoted there.

Run: LSA_LOCAL=1 pdm run python ROBUST-p-sensitivity.py
"""

import json
from pathlib import Path

import numpy as np

from article_analysis import (
    TonicModalReader,
    grouped_work_profiles,
    load_annotated_modes,
    profile_permutation_test,
)
from article_setup import cache_directory
from chord_scale import MODES, SYSTEM
from corpus import key_exact, load_corpus
from leadsheetanalyser import __version__ as LSA_VERSION

# Prespecified before running the analysis: all hundredths in the immediate
# admissible neighbourhood from the calibrated floor to the next round order.
ORDERS = (0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20)
PERMUTATIONS = 20_000
BOOTSTRAP = 20_000
KEY_TYPE_TESTS = 2
DIATONIC = 7
RANKS = np.arange(1, DIATONIC + 1, dtype=float)
SEED = 20260729
KEY_TYPES = {"major": 0, "minor": 1}
STREAMS = {"profile_test": 10, "cut_test": 11, "bootstrap": 12}

CHOCO_RELEASE = "v1.0.0"
CHOCO_SHA256 = "f50bc9e763cafd891d5934d06f2b4b8adcaee7a258c8a0514ace02561c41726d"
OUT = Path(__file__).resolve().parent / "analysis-data" / "p-sensitivity.json"


def stream(purpose, kappa):
    """A stream shared across orders but independent across tests and keys."""
    return np.random.Generator(
        np.random.PCG64([SEED, STREAMS[purpose], KEY_TYPES[kappa]])
    )


def work_profiles(songs, ids, keep, n_jazz, modes, order):
    reader = TonicModalReader(SYSTEM, order)
    return grouped_work_profiles(songs, ids, keep, n_jazz, modes, reader)


def p_text(exceedances):
    adjusted = min(
        KEY_TYPE_TESTS * (exceedances + 1) / (PERMUTATIONS + 1), 1
    )
    return (f"{KEY_TYPE_TESTS}/{PERMUTATIONS + 1}"
            if exceedances == 0 else f"{adjusted:.4f}")


def profile_test(jazz, common, kappa):
    test = profile_permutation_test(
        jazz,
        common,
        PERMUTATIONS,
        stream("profile_test", kappa),
        KEY_TYPE_TESTS,
    )
    significant = [
        MODES[j]
        for j in range(len(MODES))
        if test.adjusted_coordinate_p[j] <= 0.05
    ]
    return test.l1_gap, p_text(test.l1_exceedances), significant


def cut_test(jazz, common, kappa):
    jazz = jazz[:, :DIATONIC]
    common = common[:, :DIATONIC]
    jazz = jazz / jazz.sum(1, keepdims=True)
    common = common / common.sum(1, keepdims=True)
    cuts = np.cumsum(common.mean(0) - jazz.mean(0))
    minimum = float(cuts[:DIATONIC - 1].min())

    both = np.vstack([jazz, common])
    rng = stream("cut_test", kappa)
    null = np.empty(PERMUTATIONS)
    for t in range(PERMUTATIONS):
        allocation = rng.permutation(len(both))
        permuted = np.cumsum(
            both[allocation[len(jazz):]].mean(0)
            - both[allocation[:len(jazz)]].mean(0)
        )
        null[t] = permuted[:DIATONIC - 1].min()
    exceedances = int(np.count_nonzero(null >= minimum))

    a, b = jazz @ RANKS, common @ RANKS
    contrast = float(a.mean() - b.mean())
    rng = stream("bootstrap", kappa)
    boot = np.array([
        rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
        for _ in range(BOOTSTRAP)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "minimum_cut": round(minimum, 6),
        "all_six_cuts_positive": bool(minimum > 0),
        "cut_test_p": p_text(exceedances),
        "brightness_contrast": round(contrast, 6),
        "brightness_interval": [round(float(lo), 6), round(float(hi), 6)],
    }


def main():
    songs, _titles, ids, _styles, n_jazz = load_corpus()
    keep = key_exact(ids)
    modes = load_annotated_modes(cache_directory())
    results = {}

    for order in ORDERS:
        print(f"\np = {order:.2f}")
        works = work_profiles(songs, ids, keep, n_jazz, modes, order)
        by_key = {}
        for kappa in ("major", "minor"):
            jazz, common = works[("J", kappa)], works[("C", kappa)]
            gap, gap_p, significant = profile_test(jazz, common, kappa)
            cut = cut_test(jazz, common, kappa)
            by_key[kappa] = {
                "n_jazz": len(jazz),
                "n_common_practice": len(common),
                "l1_gap": round(gap, 6),
                "l1_test_p": gap_p,
                "max_t_significant_modes": significant,
                **cut,
            }
            print(
                f"  {kappa:5s}: l1 {gap:.4f} ({gap_p}), "
                f"min cut {cut['minimum_cut']:.4f} ({cut['cut_test_p']}), "
                f"brightness {cut['brightness_contrast']:.4f} "
                f"[{cut['brightness_interval'][0]:.4f}, "
                f"{cut['brightness_interval'][1]:.4f}]"
            )
        results[f"{order:.2f}"] = by_key

    # Independent implementation must reproduce the article at its chosen p.
    chosen = results["0.15"]
    assert round(chosen["major"]["l1_gap"], 3) == 0.093
    assert round(chosen["minor"]["l1_gap"], 3) == 0.233
    assert round(chosen["major"]["minimum_cut"], 4) == 0.0065
    assert round(chosen["minor"]["minimum_cut"], 4) == 0.0067
    assert round(chosen["major"]["brightness_contrast"], 3) == 0.125
    assert round(chosen["minor"]["brightness_contrast"], 3) == 0.383

    key_types = ("major", "minor")
    summary = {
        "l1_significant_at_every_order": all(
            results[p][k]["l1_test_p"] != "1.0000"
            and (results[p][k]["l1_test_p"].startswith("2/")
                 or float(results[p][k]["l1_test_p"]) <= 0.05)
            for p in results for k in key_types
        ),
        "cut_test_significant_at_every_order": all(
            results[p][k]["cut_test_p"].startswith("2/")
            or float(results[p][k]["cut_test_p"]) <= 0.05
            for p in results for k in key_types
        ),
        "six_cuts_positive_at_every_order": all(
            results[p][k]["all_six_cuts_positive"]
            for p in results for k in key_types
        ),
        "brightness_interval_positive_at_every_order": all(
            results[p][k]["brightness_interval"][0] > 0
            for p in results for k in key_types
        ),
        "max_t_mode_sets_stable": all(
            results[p][k]["max_t_significant_modes"]
            == chosen[k]["max_t_significant_modes"]
            for p in results for k in key_types
        ),
    }
    payload = {
        "design": {
            "orders": list(ORDERS),
            "permutations": PERMUTATIONS,
            "bootstrap_replicates": BOOTSTRAP,
            "seed": SEED,
            "same_resamples_at_each_order": True,
            "choco_release": CHOCO_RELEASE,
            "choco_archive_sha256": CHOCO_SHA256,
            "leadsheetanalyser_version": LSA_VERSION,
        },
        "results": results,
        "summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nsummary: {summary}")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
