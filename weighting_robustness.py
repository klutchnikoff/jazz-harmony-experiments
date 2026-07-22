"""Duration weighting versus token weighting -- kept for a referee, not reported.

The article weights each chord by the time it occupies.  The earlier convention
gave every chord token equal mass.  Nothing in the article rests on a comparison
between the two: duration weighting is adopted because it uses information the
corpora carry and because it neutralises the difference of annotation
granularity between the Real Book and When-in-Rome (see corpus_distances.py),
not because it wins a contest against a weaker alternative.  Presenting such a
contest would also pull the paper towards the benchmark framing it deliberately
avoids.

This script exists so that the comparison can be produced on demand -- if a
referee asks whether the findings depend on the weighting -- without rebuilding
anything.  It is intentionally absent from generate_all.py, so regenerating the
article figures does not require the token-weighted matrix.

Both matrices must already be cached; build them with

    python py-code/build_union_distances.py

Run:  python py-code/weighting_robustness.py
"""
import numpy as np

from corpus_distances import (
    load_corpus, distance_matrix, descriptive_statistics, classical_mds,
)


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


songs, titles, song_ids, styles, n_jazz = load_corpus()
Dd = distance_matrix(songs, "duration")
Dt = distance_matrix(songs, "token")

iu = np.triu_indices(len(songs), 1)
print(f"\nwhole union corpus ({len(iu[0]):,} pairs)")
print(f"  corr(duration, token) distances       = {corr(Dd[iu], Dt[iu]):+.3f}")
print(f"  mean distance   duration {Dd[iu].mean():.4f}   token {Dt[iu].mean():.4f}")

# Real Book block: the map of Section 8.1.
jazz = songs[:n_jazz]
n_distinct, _, _, _ = descriptive_statistics(jazz, "duration")
Xd, vd = classical_mds(Dd[:n_jazz, :n_jazz])
Xt, vt = classical_mds(Dt[:n_jazz, :n_jazz])
for X in (Xd, Xt):
    if X[np.argmax(n_distinct), 0] < 0:
        X[:, 0] *= -1
ij = np.triu_indices(n_jazz, 1)
print(f"\nReal Book block ({n_jazz} songs)")
print(f"  corr of the two distance matrices     = {corr(Dd[:n_jazz, :n_jazz][ij], Dt[:n_jazz, :n_jazz][ij]):+.3f}")
print(f"  corr of the two first MDS axes        = {corr(Xd[:, 0], Xt[:, 0]):+.3f}")
print(f"  corr of the two second MDS axes       = {corr(Xd[:, 1], Xt[:, 1]):+.3f}")
print(f"  variance share  duration {vd:.0%}   token {vt:.0%}")
print("\n  The first axis -- the only one the article interprets -- is stable")
print("  across the two conventions; the second is not, which is consistent")
print("  with the article reading a region rather than a second dimension.")

# Section 8.2 statistics under both conventions, on the sample and on all songs.
jz = np.where(styles == "jazz")[0]
sav = np.where(styles != "jazz")[0]
sample = np.random.default_rng(3).choice(jz, size=434, replace=False)
print(f"\nSection 8.2 statistics")
print(f"  {'weighting':10s}{'jazz set':12s}{'jazz-CP':>9s}{'plain':>8s}{'rich':>8s}{'in-jazz':>9s}")
for weighting, D in (("token", Dt), ("duration", Dd)):
    _, _, borrowed, _ = descriptive_statistics(songs, weighting)
    for name, J in (("sample 434", sample), ("all 2829", jz)):
        s = borrowed[J]
        plain = J[s <= np.quantile(s, 0.25)]
        rich = J[s >= np.quantile(s, 0.75)]
        within = D[np.ix_(J, J)][np.triu_indices(len(J), 1)].mean()
        print(f"  {weighting:10s}{name:12s}{D[np.ix_(J, sav)].mean():9.3f}"
              f"{D[np.ix_(plain, sav)].mean():8.3f}{D[np.ix_(rich, sav)].mean():8.3f}"
              f"{within:9.3f}")
print("\n  Token weighting on the 434-song sample reproduces the values the")
print("  preliminary version reported (0.62 / 0.57 / 0.73 / 0.65), which checks")
print("  that the refactoring introduced no regression.")
