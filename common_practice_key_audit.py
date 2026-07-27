"""Audit of the When-in-Rome key annotations against chord-based key estimation.

The jazz counterpart of `key_audit.py`, producing the same columns so that the
filters in `corpus.py` can read one table across both corpora.  Two scripts
rather than one because the corpora arrive by different routes: the Real Book
through a pickle of parsed progressions, When-in-Rome through JAMS files.

Not to be confused with `common_practice_audit.py`, which asks a different and
softer question -- whether the estimate names the same diatonic *collection*,
and how much of a work's duration sits under a key other than its opening one.
That one exists to justify reading a single key per work.  This one exists to
say whether that key's centre and mode can be trusted, which is what Section 6
needs and what the collection-level test was never designed to answer.

What it finds, on the 434 works: the annotations are corroborated on tonic and
mode 85.5 % of the time, against 67.9 % for the Real Book, and the minor share
survives filtering nearly intact, 37.6 % before and 36.9 % after.  ChoCo records
these annotations as `expert_human` and the jazz ones as `crowdsource`, so the
gap is in the provenance rather than in the repertoire.

Run:  LSA_LOCAL=1 .venv/bin/python common_practice_key_audit.py
"""
import warnings

import jams
import pandas as pd
from music21 import chord as m21chord
from music21 import stream

from article_setup import cache_directory, cache_is_fresh
from corpus import DATA, _annotation, classify, era
from leadsheetanalyser.chords import map_chord
from leadsheetanalyser.constants import NOTE_TO_PC

warnings.filterwarnings("ignore")

OUT = cache_directory() / "common_practice_key_audit.csv"
OUT.parent.mkdir(exist_ok=True)


def parse_annotation(label):
    """(pitch class, mode) of an annotated key string, or None."""
    if ":" not in str(label):
        return None
    note, mode = str(label).split(":")[0], str(label).split(":")[1].strip().lower()
    pc = NOTE_TO_PC.get(note)
    if pc is None:
        return None
    return pc, ("minor" if mode.startswith("min") else "major")


def estimate_key(annotation):
    """Krumhansl-Schmuckler over the chords, as key_audit.py does for the jazz."""
    s = stream.Stream()
    for obs in annotation.data:
        try:
            c = map_chord(obs.value)
        except Exception:
            continue
        if c is None or c[0] is None or any(x is None for x in c[1:12]):
            continue
        root = int(c[0])
        s.append(m21chord.Chord(sorted({root % 12}
                                       | {(root + i) % 12
                                          for i in range(1, 12) if c[i]})))
    if len(s) < 3:
        return None
    k = s.analyze("key")
    return k.tonic.pitchClass, k.mode, float(k.correlationCoefficient)


def build():
    meta = pd.read_csv(DATA / "meta.csv")
    wir = meta[meta["id"].astype(str).str.startswith("when-in-rome")]
    rows = []
    for n, (_, row) in enumerate(wir.iterrows()):
        if n % 100 == 0:
            print(f"  work {n}/{len(wir)}", flush=True)
        if era(row["composers"]) is None:
            continue
        try:
            j = jams.load(str(DATA / "jams_files" / f"{row['id']}.jams"),
                          validate=False)
        except Exception:
            continue
        chords, keys = _annotation(j, "chord_harte"), _annotation(j, "key_mode")
        if chords is None or keys is None or not len(keys.data):
            continue
        annot = parse_annotation(keys.data[0].value)
        est = estimate_key(chords)
        if est is None:
            continue
        rows.append({
            "id": str(row["id"]),
            "title": str(row.get("title", row["id"])),
            "annotated": str(keys.data[0].value) if annot else None,
            "est_pc": est[0], "est_mode": est[1], "est_corr": round(est[2], 3),
            "category": "no-annot" if annot is None else classify(annot, est),
        })
    return pd.DataFrame(rows)


if cache_is_fresh(OUT, DATA / "meta.csv", __file__):
    audit = pd.read_csv(OUT)
    print(f"loaded cached audit {OUT.name} ({len(audit)} works)")
else:
    audit = build()
    audit.to_csv(OUT, index=False)

print(f"\n[common-practice key audit]  {len(audit)} works analysed")
counts = audit["category"].value_counts()
for cat in ["exact", "relative", "parallel", "fifth", "other", "no-annot"]:
    n = int(counts.get(cat, 0))
    print(f"  {cat:9s} {n:5d}  ({n / len(audit):5.1%})")

mode = audit["annotated"].map(
    lambda v: "minor" if isinstance(v, str) and "min" in v.lower() else "major")
kept = audit["category"] == "exact"
print(f"\n  annotated minor: {int((mode == 'minor').sum())} of {len(audit)} "
      f"({(mode == 'minor').mean():.1%})")
print(f"  after the exact filter: {int((mode[kept] == 'minor').sum())} of "
      f"{int(kept.sum())} ({(mode[kept] == 'minor').mean():.1%})")
print(f"  mean estimation confidence r = {audit['est_corr'].mean():.3f}")
print(f"  written to {OUT}")
