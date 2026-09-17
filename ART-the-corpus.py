"""Corpus sizes used in Section 2.2, "A vocabulary fixed by rule".

Values are printed for reading and written to article-data/ for
check_article_numbers.py, which verifies that each appears in the manuscript.

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
    print(f"\nReal Book     {values['jazz_songs']:5,d} lead sheets  "
          f"{values['jazz_tokens']:8,d} symbols  "
          f"{values['jazz_kinds']:4d} kinds")
    print(f"When-in-Rome  {values['cp_works']:5,d} works        "
          f"{values['cp_tokens']:8,d} symbols  "
          f"{values['cp_kinds']:4d} kinds")

    export("the-corpus", values)


if __name__ == "__main__":
    main()
