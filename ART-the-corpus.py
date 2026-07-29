"""Article data for Section 2.2, "Corpora".

Produces every number that subsection states, and nothing else.  Values are
printed for reading and written to article-data/ for
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
    # Section 2.2 states both densities and their ratio, where it used to say
    # "some four times" and leave the figure out of the pipeline entirely
    per_jazz = values["jazz_tokens"] / values["jazz_songs"]
    per_cp = values["cp_tokens"] / values["cp_works"]
    values["per_sheet"] = f"{per_jazz:.1f}"
    values["per_work"] = f"{per_cp:.1f}"
    values["density_ratio"] = f"{per_cp / per_jazz:.1f}"

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
