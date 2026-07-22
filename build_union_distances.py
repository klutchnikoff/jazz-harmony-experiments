"""Compute and cache the union song-distance matrix (Sections 8.1 and 8.2).

The article uses the duration-weighted matrix, which is what this script builds
by default; mixed_corpus_mds.py reads a block of it.  It
also build it on demand, so running this script is only a convenience when one
wants the (slow) computation done up front.

    python py-code/build_union_distances.py                  # duration only
    python py-code/build_union_distances.py --with-token     # + token weighting
    python py-code/build_union_distances.py --with-variants  # + the three
                                                             #   degraded costs

The token-weighted matrix is not used by any article figure.  It is the material
for weighting_robustness.py, kept in case a referee asks whether the findings
depend on the weighting convention.  The three degraded ground costs -- uniform
weights, the identity system, and the direction-only embedding -- are what
robustness_checks.py reads for the closing paragraph of Section 8.2.
"""
import sys
import time

from corpus_distances import load_corpus, distance_matrix

t0 = time.time()
songs, titles, song_ids, styles, n_jazz = load_corpus()
print(f"union corpus: {len(songs)} songs "
      f"({n_jazz} jazz, {len(songs) - n_jazz} common practice)")

variants = ["duration"]
if "--with-token" in sys.argv:
    variants.append("token")
if "--with-variants" in sys.argv:
    variants += ["uniformW", "identityW", "angular"]

for variant in variants:
    print(f"\n=== {variant} ===")
    distance_matrix(songs, variant)

print(f"\ntotal {time.time() - t0:.0f}s")
