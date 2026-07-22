"""
Audit of the Real Book key annotations against chord-based key estimation.

For every song we estimate a global key from the pitch-class content of its
chord symbols with music21's default key-finding algorithm (Krumhansl-Schmuckler
correlation with Aarden-Essen profiles), then classify its relation to the
ChoCo key annotation:

  exact        same tonic pitch class, same mode
  relative     annotated key is the relative major/minor of the estimated key
  parallel     same tonic pitch class, opposite mode
  fifth        tonics a perfect fifth apart, any mode
  other        anything else
  no-annot     no parseable key annotation

Results are written to tmp/key_audit.csv (id, title, annotated, estimated,
correlation, category); the id is what downstream code matches on, since
thirteen Real Book titles are shared by two or more entries and a summary is printed.

The categories above compare tonics.  What Section 8.1 actually rests on is the
weaker comparison of diatomic COLLECTIONS, minor mapped to its relative major on
both sides, and the summary reports that separately: the share of songs whose
annotation is corroborated, the distribution of the residual disagreements by
interval, and the two groups the section retains.  The distinction matters --
an annotation and an estimate can name different tonics of the same collection,
which for our purposes is agreement.

Run:  python py-code/key_audit.py
"""
import numpy as np
import pandas as pd
from music21 import chord as m21chord
from music21 import stream

from article_setup import DATA_ROOT, cache_directory, cache_is_fresh
from leadsheetanalyser.constants import NOTE_TO_PC

OUT = cache_directory() / "key_audit.csv"
OUT.parent.mkdir(exist_ok=True)


def parse_annotation(key_field):
    if not (isinstance(key_field, list) and key_field and isinstance(key_field[0], str)
            and ":" in key_field[0]):
        return None
    note, mode = key_field[0].split(":")[0], key_field[0].split(":")[1]
    pc = NOTE_TO_PC.get(note)
    if pc is None or mode not in ("major", "minor"):
        return None
    return pc, mode


def estimate_key(prog):
    s = stream.Stream()
    for c in prog:
        if c is None or c[0] is None or any(x is None for x in c[1:12]):
            continue
        root = int(c[0])
        pcs = sorted({root % 12} | {(root + i) % 12 for i in range(1, 12) if c[i]})
        s.append(m21chord.Chord(pcs))
    if len(s) < 3:
        return None
    k = s.analyze("key")
    return k.tonic.pitchClass, k.mode, float(k.correlationCoefficient)


def classify(annot, est):
    (apc, amode), (epc, emode, _) = annot, est
    if apc == epc and amode == emode:
        return "exact"
    if emode == "minor" and amode == "major" and apc == (epc + 3) % 12:
        return "relative"          # annotation is the relative major
    if emode == "major" and amode == "minor" and apc == (epc + 9) % 12:
        return "relative"          # annotation is the relative minor
    if apc == epc:
        return "parallel"
    if (apc - epc) % 12 in (5, 7):
        return "fifth"
    return "other"


SOURCE = DATA_ROOT / "music_realbook.pkl"

if cache_is_fresh(OUT, SOURCE, __file__):
    audit = pd.read_csv(OUT)
    print(f"loaded cached audit {OUT.name} ({len(audit)} songs); "
          f"pass --force to recompute")
else:
    df = pd.read_pickle(SOURCE)
    rows = []
    for n, (_, row) in enumerate(df.iterrows()):
        if n % 200 == 0:
            print(f"  song {n}/{len(df)}", flush=True)
        annot = parse_annotation(row["key"])
        est = estimate_key(row["chord_progression"])
        if est is None:
            continue
        category = "no-annot" if annot is None else classify(annot, est)
        rows.append({
            "id": str(row["id"]),
            "title": str(row["title"]),
            "annotated": row["key"][0] if annot else None,
            "est_pc": est[0], "est_mode": est[1], "est_corr": round(est[2], 3),
            "category": category,
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT, index=False)

print(f"\n[key audit]  {len(audit)} songs analysed")
counts = audit["category"].value_counts()
for cat in ["exact", "relative", "parallel", "fifth", "other", "no-annot"]:
    n = int(counts.get(cat, 0))
    print(f"  {cat:9s} {n:5d}  ({n / len(audit):5.1%})")
print(f"\n  mean estimation confidence r = {audit['est_corr'].mean():.3f}")
low = audit[audit["est_corr"] < 0.7]
print(f"  songs with r < 0.7: {len(low)} ({len(low) / len(audit):.1%})")
print(f"  written to {OUT}")

# ---------------------------------------------------------------------------
# Collection-level agreement: the figures quoted in Section 8.1.
from corpus_distances import key_reliable, normalised_tonic, SUBDOMINANT_PULL


def annotated_collection(label):
    if not isinstance(label, str) or ":" not in label:
        return None
    return normalised_tonic(label.split(":")[0], label.split(":")[1])


gaps = []
for annotated, pc, mode in zip(audit["annotated"], audit["est_pc"], audit["est_mode"]):
    ann = annotated_collection(annotated)
    if ann is None:
        gaps.append(None)
        continue
    gaps.append((((int(pc) + 3) % 12 if str(mode) == "minor" else int(pc)) - ann) % 12)
gaps = pd.Series(gaps, dtype="Int64")
usable = gaps.notna()

print(f"\n[collections]  {int(usable.sum())} songs with a parseable annotation")
print(f"  annotation corroborated (same collection): "
      f"{(gaps == 0).sum()} ({(gaps == 0).sum() / usable.sum():.1%})")
print("  residual disagreements, by interval from annotation to estimate:")
for g in range(1, 12):
    n = int((gaps == g).sum())
    if n:
        note = ""
        if g == SUBDOMINANT_PULL:
            note = "  <- subdominant pull, annotation trusted"
        elif g in (3, 8, 10):
            note = "  <- a third or a tone: manufactures borrowed degrees"
        print(f"    +{g:2d} semitones {n:5d}{note}")

keep = key_reliable(list(audit["id"]))
print(f"\n  retained: {keep.sum()} of {int(usable.sum())} "
      f"({keep.sum() / usable.sum():.1%}) -- corroborated, plus the "
      f"{(gaps == SUBDOMINANT_PULL).sum()} songs pulled to the subdominant")
print(f"  discarded: {int(usable.sum()) - keep.sum()}")
