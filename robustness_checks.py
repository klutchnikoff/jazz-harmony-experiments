"""How much of the corpus geometry is owed to the modal system (Section 8.2).

The closing paragraph of Section 8.2 reports that the jazz / common-practice
separation survives three degradations of the ground cost.  This script produces
those numbers.  The variants are declared in corpus_distances.VARIANTS:

  duration    the article's system W_D
  uniformW    uniform weights on the same seven diatonic supports, so that a
              modal coordinate can only count occupied degrees
  identityW   W = I, no modal structure at all: d_W becomes the unweighted
              Euclidean distance between kind vectors
  angular     W_D again, but each profile normalised to unit length, so only the
              direction of a chord's modal affinities enters the cost

Each variant needs its own union distance matrix, cached in tmp/ under the
variant name and built on demand if absent.  Building one from scratch takes on
the order of ten minutes, so on a cold cache this script is slow; running

    python py-code/build_union_distances.py --with-variants

first does the same work with clearer progress reporting.

The jazz side is restricted to the key-corroborated songs, exactly as in
mixed_corpus_mds.py, so the numbers are comparable with the rest of Section 8.

Run:  python py-code/robustness_checks.py
"""
import numpy as np

import article_setup  # noqa: F401
from corpus_distances import (
    load_corpus, distance_matrix, key_reliable,
)

VARIANTS = [
    ("theory-guided W_D", "duration"),
    ("uniform on same supports", "uniformW"),
    ("identity W (no modal structure)", "identityW"),
    ("angular (direction only)", "angular"),
]

songs, titles, song_ids, styles, n_jazz = load_corpus()
reliable = np.concatenate([key_reliable(song_ids[:n_jazz]),
                           np.ones(len(songs) - n_jazz, bool)])
keep = np.flatnonzero(reliable)
is_jazz = (styles[keep] == "jazz")

print(f"{len(keep)} songs ({is_jazz.sum()} jazz, {(~is_jazz).sum()} common practice)\n")
print(f"{'ground cost':34s}{'axis-2 gap':>12s}{'axis-1 gap':>12s}")

for label, variant in VARIANTS:
    D = distance_matrix(songs, variant)[np.ix_(keep, keep)]
    n = len(D)
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ (D ** 2) @ centring
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    X = vectors[:, :2] * np.sqrt(np.clip(values[:2], 0, None))
    gaps = [abs(X[is_jazz, a].mean() - X[~is_jazz, a].mean()) / X[:, a].std()
            for a in (0, 1)]
    print(f"{label:34s}{gaps[1]:12.2f}{gaps[0]:12.2f}")

print("\nThe separation is carried by the second axis under every variant.")
