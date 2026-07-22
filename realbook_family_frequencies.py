"""Count the diagnostic chord families in the Real Book corpus.

The denominator is the set of valid chord tokens in the 2,846-song processed
Real Book partition used by the article.

Run:  python py-code/realbook_family_frequencies.py
"""

from collections import Counter

import pandas as pd

from article_setup import PACKAGE_ROOT
from diagnostic_vocabulary import FAMILY_ORDER, VOCABULARY


DATA = PACKAGE_ROOT / "data" / "music_realbook.pkl"
KIND_TO_FAMILY = {
    tuple(kind): family
    for _, kind, family in VOCABULARY
}

corpus = pd.read_pickle(DATA)
counts = Counter()
invalid = 0
kinds_seen = set()

for progression in corpus["chord_progression"]:
    for chord in progression:
        if len(chord) < 12 or chord[0] is None or any(value is None for value in chord[1:12]):
            invalid += 1
            continue
        kind = tuple(int(value) for value in chord[1:12])
        kinds_seen.add(kind)
        counts[KIND_TO_FAMILY.get(kind, "Other kinds")] += 1

total = sum(counts.values())
selected = sum(counts[family] for family in FAMILY_ORDER)
selected_kinds_seen = kinds_seen.intersection(KIND_TO_FAMILY)

print(f"songs: {len(corpus)}")
print(f"valid chord tokens: {total}")
print(f"invalid chord tokens excluded: {invalid}")
print(f"distinct kinds: {len(kinds_seen)}")
print(f"selected kinds present: {len(selected_kinds_seen)}")
print(f"other kinds present: {len(kinds_seen) - len(selected_kinds_seen)}")
print()
print(f"{'family':16s} {'tokens':>10s} {'corpus share':>14s}")
for family in [*FAMILY_ORDER, "Other kinds"]:
    count = counts[family]
    print(f"{family:16s} {count:10d} {100 * count / total:13.2f}%")
print(f"{'Selected total':16s} {selected:10d} {100 * selected / total:13.2f}%")

# ---------------------------------------------------------------------------
# How the diagnostic vocabulary relates to sheer frequency (Section 7.1).
# The vocabulary spans five intervallic families rather than listing the
# commonest kinds, and the article states where the two criteria diverge.
per_kind = Counter()
for progression in corpus["chord_progression"]:
    for chord in progression:
        if len(chord) < 12 or chord[0] is None or any(v is None for v in chord[1:12]):
            continue
        per_kind[tuple(int(v) for v in chord[1:12])] += 1

NAME = {tuple(kind): name for name, kind, _ in VOCABULARY}
NOTE = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]
top20 = per_kind.most_common(20)
in_vocabulary = sum(1 for kind, _ in top20 if kind in NAME)

print()
print(f"{in_vocabulary} of the 20 commonest kinds are in the diagnostic vocabulary")
print("  vocabulary kinds outside the 20 commonest, rarest first:")
outside = sorted(((NAME[k], per_kind.get(k, 0)) for k in NAME
                  if k not in {kind for kind, _ in top20}), key=lambda kv: kv[1])
for name, n in outside:
    print(f"    {name:12s}{n:8d}{n / total:8.3%}")
print("  frequent kinds outside the vocabulary:")
for kind, n in top20:
    if kind not in NAME:
        intervals = " ".join(NOTE[j + 1] for j in range(11) if kind[j])
        print(f"    [1 {intervals}]{n:8d}{n / total:8.3%}")
