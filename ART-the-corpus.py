"""Article data for Section 2.2, "The corpus".

Produces every number that subsection states, and nothing else.  Values are
printed for reading and written to results/data/ for ART-check.py, which
verifies that each of them appears in the manuscript.

Run:  LSA_LOCAL=1 .venv/bin/python ART-the-corpus.py
"""
from article_data import export
from corpus import load_corpus
from vocabulary import corpus_counts


def main():
    songs, titles, song_ids, styles, n_jazz = load_corpus()
    tokens, _ = corpus_counts()

    values = {
        "jazz_songs": n_jazz,
        "jazz_tokens": sum(tokens["jazz"].values()),
        "jazz_kinds": len(tokens["jazz"]),
        "cp_works": len(songs) - n_jazz,
        "cp_tokens": sum(tokens["cp"].values()),
        "cp_kinds": len(tokens["cp"]),
    }
    # The clause on editorial granularity rounds this ratio to "some four times".
    per_jazz = values["jazz_tokens"] / values["jazz_songs"]
    per_cp = values["cp_tokens"] / values["cp_works"]

    print(f"\nReal Book     {values['jazz_songs']:5,d} lead sheets  "
          f"{values['jazz_tokens']:8,d} symbols  {values['jazz_kinds']:4d} kinds  "
          f"{per_jazz:6.1f} per sheet")
    print(f"When-in-Rome  {values['cp_works']:5,d} works        "
          f"{values['cp_tokens']:8,d} symbols  {values['cp_kinds']:4d} kinds  "
          f"{per_cp:6.1f} per work")
    print(f"\nsymbols per work over symbols per sheet: {per_cp / per_jazz:.2f}")

    export("the-corpus", values)


if __name__ == "__main__":
    main()
