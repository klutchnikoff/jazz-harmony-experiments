"""Article data for Section 2.3, "A vocabulary fixed by rule".

Produces every number that subsection states, and nothing else, including the
table: the kinds of each family, their token counts, and their share.

Small integers are deliberately not exported.  check_article_numbers.py searches
the manuscript for each value, and a number like 5 or 32 occurs in too many
places for finding it to prove anything; only figures distinctive enough for the
search to mean something are exported.

Run:  LSA_LOCAL=1 .venv/bin/python ART-a-vocabulary-fixed-by-rule.py
"""
from article_data import export
from vocabulary import build, corpus_counts, family, name, FAMILIES, COVERAGE, one_step


def families_at(coverage, tokens, in_songs):
    """How many of the five families a given coverage cut retains."""
    total, cum, seed = sum(tokens["jazz"].values()), 0, []
    for k, n in tokens["jazz"].most_common():
        cum += n
        seed.append(k)
        if cum / total >= coverage:
            break
    added = {k for k in one_step(seed) if in_songs["jazz"][k] >= 2}
    return {family(k) for k in seed} | {family(k) for k in added}, len(seed) + len(added)


def main():
    vocab, seed, added, outside, tokens, in_songs = build()
    jazz_total = sum(tokens["jazz"].values())
    cp_total = sum(tokens["cp"].values())

    kept = sum(tokens["jazz"][k] for k in vocab)
    values = {
        # a phrase, not a bare count: "32" alone would be found anywhere
        "vocabulary_size": f"{len(vocab)} kinds",
        "vocabulary_tokens": kept,
        # exported as the article writes them: a trailing zero is significant
        "jazz_coverage": f"{100 * kept / jazz_total:.2f}",
        "cp_coverage": f"{100 * sum(tokens['cp'][k] for k in vocab) / cp_total:.2f}",
    }

    print(f"\n{'family':12s} {'#':>3s} {'tokens':>9s} {'share':>7s}   kinds")
    for fam in FAMILIES:
        ks = [k for k in sorted(vocab, key=lambda k: -tokens["jazz"][k])
              if family(k) == fam]
        n = sum(tokens["jazz"][k] for k in ks)
        values[f"tokens_{fam.lower().replace('-', '_')}"] = n
        print(f"{fam:12s} {len(ks):3d} {n:9,d} {n/jazz_total:7.2%}   "
              + ", ".join(name(k) for k in ks))
    print(f"{'total':12s} {len(vocab):3d} {kept:9,d} {kept/jazz_total:7.2%}")

    for cov in (0.90, COVERAGE):
        fams, size = families_at(cov, tokens, in_songs)
        missing = [f for f in FAMILIES if f not in fams]
        print(f"\nat {cov:.0%} coverage: {size} kinds, "
              f"{len(fams)} of the five families"
              + (f", missing {' and '.join(missing)}" if missing else ""))

    export("a-vocabulary-fixed-by-rule", values)


if __name__ == "__main__":
    main()
