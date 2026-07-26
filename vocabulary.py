"""The chord-kind vocabulary, fixed by rules stated before the data is read.

Three rules, and nothing else, decide which kinds the article works with.

  Coverage.    Kinds are ranked by token frequency in the jazz corpus and the
               list is cut at 95 % coverage.  The jazz corpus alone: it is the
               repertoire the construction is built for, and the common-practice
               works are then read with a vocabulary they did not shape, which
               makes them a test rather than a party to the design.

               All of it, with no key filter.  Counting which kinds are played
               does not require the annotated key to be trustworthy, and the
               filter would import a bias of its own: it discards the songs whose
               annotated key is not corroborated, which are the harmonically
               adventurous ones, and those are exactly the songs that play the
               unusual chords.  Filtering costs the vocabulary 7#9, played in 218
               songs, along with two minor elevenths.

  Families.    Each kind falls to the first applicable clause: 3 and 6 present,
               Diminished; else 4 and 8 present and 7 absent, Augmented; else 4
               present, Major-third; else 3 present, Minor-third; otherwise
               Suspended.  Indices are semitones above the root.

  Completion.  Each family gains, in one step, the kinds a member generates by
               adding or removing the ninth, by removing a seventh it contains,
               or by adding a seventh of a quality another member already uses.
               Of those only the kinds played in at least two distinct songs are
               kept, which guards against annotation accidents, not rarity.

Also reported: the kinds the common-practice corpus plays and this vocabulary
does not contain -- the out-of-vocabulary set that Section 6 examines.

Run:  LSA_LOCAL=1 .venv/bin/python vocabulary.py
"""
import numpy as np
from collections import Counter

from corpus import load_corpus

COVERAGE = 0.95
MIN_SONGS = 2
NINTH = 2
SEVENTHS = (10, 11)
FAMILIES = ["Major-third", "Minor-third", "Suspended", "Diminished", "Augmented"]

NAMES = {
    (0,0,0,1,0,0,1,0,0,0,0): "maj",      (0,0,1,0,0,0,1,0,0,0,0): "min",
    (0,0,0,1,0,0,1,0,0,1,0): "7",        (0,0,1,0,0,0,1,0,0,1,0): "min7",
    (0,0,0,1,0,0,1,0,0,0,1): "maj7",     (0,0,1,0,0,0,1,0,1,0,0): "min6",
    (0,0,0,1,0,0,1,0,1,0,0): "6",        (0,1,1,0,0,0,1,0,0,1,0): "min9",
    (0,1,0,1,0,0,1,0,0,1,0): "9",        (0,1,1,0,0,0,1,0,1,0,0): "min69",
    (0,1,0,1,0,0,1,0,1,0,0): "69",       (0,0,1,0,0,1,0,0,0,1,0): "min7b5",
    (0,1,0,1,0,0,1,0,0,0,1): "maj9",     (0,0,1,0,0,1,0,0,1,0,0): "dim7",
    (0,0,0,1,0,0,0,1,0,1,0): "aug7",     (0,0,1,0,0,1,0,0,0,0,0): "dim",
    (0,0,0,1,0,0,0,1,0,0,0): "aug",      (0,0,0,0,1,0,1,0,0,1,0): "7sus4",
    (1,0,0,1,0,0,1,0,0,1,0): "7b9",      (0,1,0,0,1,0,1,0,0,1,0): "7sus2sus4",
    (0,1,0,0,0,0,1,0,0,1,0): "7sus2",    (0,1,0,1,0,0,1,0,0,0,0): "add9",
    (0,0,0,0,1,0,1,0,0,0,0): "sus4",     (0,1,1,0,0,0,1,0,0,0,0): "min(add9)",
    (0,1,0,1,1,0,1,0,1,1,0): "13",       (0,1,0,1,0,0,0,1,0,1,0): "9#5",
    (0,0,0,1,0,0,1,0,1,0,1): "maj7(13)", (0,0,0,1,0,0,1,0,1,1,0): "7(13)",
    (0,0,1,0,0,0,1,0,1,1,0): "min7(13)", (0,1,0,1,1,0,1,0,1,0,1): "maj13",
    (0,0,1,1,0,0,1,0,0,1,0): "7#9",      (0,1,1,0,1,0,1,0,0,1,0): "min11",
    (0,0,1,0,1,0,1,0,0,1,0): "min7(11)",
}


def name(kind):
    k = tuple(kind)
    return NAMES.get(k) or "{" + ",".join(
        str(i + 1) for i, x in enumerate(k) if x) + "}"


def family(kind):
    has = lambda i: kind[i - 1] == 1
    if has(3) and has(6):
        return "Diminished"
    if has(4) and has(8) and not has(7):
        return "Augmented"
    if has(4):
        return "Major-third"
    if has(3):
        return "Minor-third"
    return "Suspended"


def one_step(seed):
    """The kinds a member of each family generates in a single step."""
    qualities = {}
    for k in seed:
        qualities.setdefault(family(k), set()).update(
            {s for s in SEVENTHS if k[s - 1]})
    out = set()
    for k in seed:
        fam, v = family(k), list(k)
        for bit in (0, 1):
            w = v[:]; w[NINTH - 1] = bit
            if any(w):
                out.add(tuple(w))
        for s in SEVENTHS:
            if v[s - 1]:
                w = v[:]; w[s - 1] = 0
                if any(w):
                    out.add(tuple(w))
        for s in qualities.get(fam, ()):
            if not v[s - 1]:
                w = v[:]; w[s - 1] = 1
                out.add(tuple(w))
    return out - set(seed)


def corpus_counts():
    """Token counts and song counts per kind, for each repertoire."""
    songs, titles, song_ids, styles, n_jazz = load_corpus()
    tokens = {"jazz": Counter(), "cp": Counter()}
    in_songs = {"jazz": Counter(), "cp": Counter()}
    for i in range(len(songs)):
        g = "jazz" if styles[i] == "jazz" else "cp"
        for k in {tuple(k) for (r, k), d in songs[i]}:
            in_songs[g][k] += 1
        for (r, k), d in songs[i]:
            tokens[g][tuple(k)] += 1
    return tokens, in_songs


def build():
    """The vocabulary, plus what the common-practice corpus plays outside it."""
    tokens, in_songs = corpus_counts()
    total = sum(tokens["jazz"].values())
    seed, cum = [], 0
    for k, n in tokens["jazz"].most_common():
        cum += n
        seed.append(k)
        if cum / total >= COVERAGE:
            break
    added = {k for k in one_step(seed) if in_songs["jazz"][k] >= MIN_SONGS}
    vocab = list(seed) + sorted(added, key=lambda k: -tokens["jazz"][k])
    outside = {k for k in tokens["cp"] if k not in set(vocab)
               and in_songs["cp"][k] >= MIN_SONGS}
    return vocab, set(seed), added, outside, tokens, in_songs


def main():
    vocab, seed, added, outside, tokens, in_songs = build()
    tj, tc = sum(tokens["jazz"].values()), sum(tokens["cp"].values())
    cov_j = sum(tokens["jazz"][k] for k in vocab) / tj
    cov_c = sum(tokens["cp"][k] for k in vocab) / tc
    print(f"\nseed {len(seed)} kinds at {COVERAGE:.0%} of the jazz corpus, "
          f"+{len(added)} by completion = {len(vocab)} kinds")
    print(f"covering {cov_j:.2%} of jazz tokens and {cov_c:.2%} of "
          f"common-practice tokens\n")
    for fam in FAMILIES:
        ks = [k for k in sorted(vocab, key=lambda k: -tokens["jazz"][k])
              if family(k) == fam]
        print(f"  {fam} ({len(ks)})")
        for k in ks:
            print(f"    {'+' if k in added else ' '} {name(k):12s} "
                  f"jazz {tokens['jazz'][k]:7,d} in {in_songs['jazz'][k]:5d} songs"
                  f"   cp {tokens['cp'][k]:7,d}")
    print(f"\nplayed by the common-practice corpus, outside the vocabulary "
          f"({len(outside)} kinds, {sum(tokens['cp'][k] for k in outside)/tc:.2%} "
          f"of its tokens)")
    for k in sorted(outside, key=lambda k: -tokens["cp"][k]):
        print(f"    {name(k):14s} cp {tokens['cp'][k]:6,d} in "
              f"{in_songs['cp'][k]:4d} works   jazz {tokens['jazz'][k]:6,d}")


if __name__ == "__main__":
    main()
