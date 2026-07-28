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

    assert all(family(k) in FAMILIES for k in vocab), (
        "the vocabulary now contains an unclassified kind")

    # Section 2.1 prints the kind of a dominant seventh in full, which is a
    # claim about the interval convention and not an illustration: the ones sit
    # at the major third, the perfect fifth, and the minor seventh.
    # Section 2.3 says the thirty-two carry thirty-two distinct symbols, which
    # is what lets it speak of a kind's ninth or seventh without ambiguity: a
    # coordinate carries no spelling, and several named intervals share one.
    assert len({name(k) for k in vocab}) == len(vocab), (
        "two kinds of the vocabulary now share a symbol")

    dominant = next(k for k in vocab if name(k) == "7")
    assert dominant == (0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0), (
        f"the dominant seventh is now {dominant}, and Section 2.1 prints it")
    jazz_total = sum(tokens["jazz"].values())
    cp_total = sum(tokens["cp"].values())

    kept = sum(tokens["jazz"][k] for k in vocab)
    values = {
        # a phrase, not a bare count: "32" alone would be found anywhere
        "vocabulary_size": f"{len(vocab)} kinds",
        "dominant_seventh": ",".join(str(x) for x in dominant),
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

    # Section 2.3 says a 90 % cut would leave no suspended kind, and that
    # thirteen of the thirty-two come from the completion.  Both were computed
    # here and neither was exported, so neither was checked.
    spelled_small = {13: "thirteen", 19: "Nineteen", 20: "Twenty",
                     21: "Twenty-one"}

    # Section 2.3 makes both claims of the segment itself, before completion,
    # which is the stronger form: at 95 % the twenty kinds already hold every
    # family, at 90 % the thirteen hold no suspended one.
    def segment(coverage):
        total, cum, out = sum(tokens["jazz"].values()), 0, []
        for k, n in tokens["jazz"].most_common():
            cum += n
            out.append(k)
            if cum / total >= coverage:
                return out
        return out

    at_ninety = {family(k) for k in segment(0.90)}
    assert "Suspended" not in at_ninety, (
        "a 90 % segment now holds a suspended kind, which Section 2.3 denies")
    assert set(FAMILIES) == {family(k) for k in segment(COVERAGE)}, (
        "the 95 % segment no longer holds every family")
    values["segment_size"] = f"{spelled_small[len(segment(COVERAGE))]} kinds meet"
    values["segment_at_ninety"] = f"{spelled_small[len(segment(0.90))]} kinds and no"
    spelled = {12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen"}
    values["from_completion"] = f"{spelled[len(added)]} of the thirty-two"

    export("a-vocabulary-fixed-by-rule", values)


if __name__ == "__main__":
    main()
