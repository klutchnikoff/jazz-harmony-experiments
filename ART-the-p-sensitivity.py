"""Promote the reported p-sensitivity results into the article-data pipeline.

ROBUST-p-sensitivity.py owns the complete prespecified grid and all numerical
results.  This producer reads that versioned record, asserts the qualitative
claims made in Section 6.3, and exports exactly the phrases and boundary values
stated in the manuscript.

Run ROBUST-p-sensitivity.py first, then this script.
"""

import json
from pathlib import Path

from article_data import export


SOURCE = Path(__file__).resolve().parent / "analysis-data" / "p-sensitivity.json"


def main():
    if not SOURCE.exists():
        raise RuntimeError(f"{SOURCE} is missing; run ROBUST-p-sensitivity.py")
    analysis = json.loads(SOURCE.read_text())
    assert analysis["design"]["orders"] == [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]

    summary = analysis["summary"]
    assert summary["l1_significant_at_every_order"]
    assert summary["max_t_mode_sets_stable"]
    assert summary["six_cuts_positive_at_every_order"]
    assert summary["brightness_interval_positive_at_every_order"]
    assert not summary["cut_test_significant_at_every_order"]

    result = analysis["results"]
    major_modes = result["0.15"]["major"]["max_t_significant_modes"]
    minor_modes = result["0.15"]["minor"]["max_t_significant_modes"]
    assert len(major_modes) == 4 and len(minor_modes) == 5
    assert result["0.18"]["minor"]["cut_test_p"] == "0.0470"
    assert result["0.19"]["minor"]["cut_test_p"] == "0.0512"

    export("the-p-sensitivity", {
        "grid_start": "p=0.14",
        "grid_end": "p=0.20",
        "profile_tests": "remain significant at all seven orders",
        "max_t_sets": "the same four major-key and five minor-key coordinates",
        "cut_direction": "All six cumulative differences remain positive",
        "bootstrap_direction": "every brightness bootstrap interval excludes zero",
        "major_cut_upper": result["0.20"]["major"]["cut_test_p"],
        "minor_cut_lower": result["0.14"]["minor"]["cut_test_p"],
        "minor_cut_upper": result["0.20"]["minor"]["cut_test_p"],
        "minor_last_significant": result["0.18"]["minor"]["cut_test_p"],
        "minor_first_nonsignificant": result["0.19"]["minor"]["cut_test_p"],
    })


if __name__ == "__main__":
    main()
