"""Article data for Section 2.3, "A vocabulary fixed by rule".

Produces every value that subsection states, including its rule parameters and
the table.  Small integers remain unambiguous because the manuscript references
export keys directly rather than searching for their rendered values.

Run:  LSA_LOCAL=1 .venv/bin/python ART-a-vocabulary-fixed-by-rule.py
"""
from article_data import export
from corpus import MIN_CHORDS
from vocabulary import (
    build, corpus_counts, family, name, FAMILIES, COVERAGE, MIN_SONGS, one_step,
)


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
        "vocabulary_size": len(vocab),
        "minimum_chords": MIN_CHORDS,
        "coverage_target": f"{COVERAGE:.2f}",
        "coverage_target_percent": f"{100 * COVERAGE:.0f}",
        "diagnostic_coverage_percent": "90",
        "family_count": len(FAMILIES),
        "minimum_completion_songs": MIN_SONGS,
        "dominant_seventh": list(dominant),
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
        stem = fam.lower().replace("-", "_")
        values[f"tokens_{stem}"] = n
        values[f"share_{stem}"] = f"{100 * n / jazz_total:.2f}"
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
    values["segment_size"] = len(segment(COVERAGE))
    values["segment_at_ninety"] = len(segment(0.90))
    values["completion_size"] = len(added)

    export("a-vocabulary-fixed-by-rule", values)


if __name__ == "__main__":
    main()
