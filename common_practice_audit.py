"""
Audit of the common-practice key annotations (Section 8.2, other side).

Section 8.1 establishes which Real Book annotations we can build on.  The
common-practice side of the mixed corpus rests on annotations too, and this
script subjects them to the same scrutiny, plus one test the jazz side cannot
support.

Two questions:

  agreement    the test of key_audit.py, transposed: does the Krumhansl-
               Schmuckler estimate designate the same diatonic collection as the
               annotation the loader uses?

  modulation   the When-in-Rome files annotate the whole succession of keys, not
               just the opening one, and load_common_practice() reads only the
               first.  So we can measure exactly what that costs: the share of a
               work's duration spent under a key whose collection differs from
               the opening one, and by how far.  A jazz lead sheet carries a
               single key annotation, so no such measurement exists there.

The second is the sharper question.  A global transposition is only meaningful
for a work that stays put; a sonata movement whose second group sits in the
dominant is being read, for half its length, against a tonic it has left.

Run:  python py-code/common_practice_audit.py
"""
import jams
import numpy as np
import pandas as pd
from music21 import chord as m21chord
from music21 import stream

from article_setup import DATA_ROOT, cache_directory, cache_is_fresh
from corpus_distances import (
    DATA, era, map_chord, normalised_tonic, _annotation, SUBDOMINANT_PULL,
)

def estimate_key(prog):
    """The estimator of key_audit.py, inlined: that module is a script."""
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


OUT = cache_directory() / "common_practice_audit.csv"
OUT.parent.mkdir(exist_ok=True)
PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

def build_audit():
    meta = pd.read_csv(DATA / "meta.csv")
    wir = meta[meta["id"].astype(str).str.startswith("when-in-rome")]

    rows = []
    for n, (_, row) in enumerate(wir.iterrows()):
        if n % 200 == 0:
            print(f"  work {n}/{len(wir)}", flush=True)
        style = era(row["composers"])
        if style is None:
            continue
        try:
            j = jams.load(str(DATA / "jams_files" / f"{row['id']}.jams"), validate=False)
        except Exception:
            continue
        chords = _annotation(j, "chord_harte")
        keys = _annotation(j, "key_mode")
        if chords is None or keys is None or not len(keys.data):
            continue
        opening = normalised_tonic(*str(keys.data[0].value).split(":")[:2]) \
            if ":" in str(keys.data[0].value) else None
        if opening is None:
            continue

        # What the loader transposes, and what the annotations say it should be.
        total = away = 0.0
        far = 0.0
        for obs in keys.data:
            label = str(obs.value)
            d = float(obs.duration or 0.0)
            if d <= 0 or ":" not in label:
                continue
            col = normalised_tonic(*label.split(":")[:2])
            if col is None:
                continue
            total += d
            if col != opening:
                away += d
                if (col - opening) % 12 in (3, 8, 10):
                    far += d
        if total <= 0:
            continue

        progression = []
        for obs in chords.data:
            try:
                progression.append(map_chord(obs.value))
            except Exception:
                progression.append(None)
        est = estimate_key([c for c in progression if c is not None])
        if est is None:
            continue
        est_col = (est[0] + 3) % 12 if est[1] == "minor" else est[0]

        rows.append({
            "id": str(row["id"]),
            "style": style,
            "n_keys": len(keys.data),
            "opening": PC_NAMES[opening],
            "estimated": PC_NAMES[est_col],
            "gap": (est_col - opening) % 12,
            "est_corr": round(est[2], 3),
            "away_share": away / total,
            "far_share": far / total,
        })

    return pd.DataFrame(rows)


if cache_is_fresh(OUT, DATA / "meta.csv", __file__):
    audit = pd.read_csv(OUT)
    print(f"loaded cached audit {OUT.name} ({len(audit)} works); "
          f"pass --force to recompute")
else:
    audit = build_audit()
    audit.to_csv(OUT, index=False)

print(f"\n[common practice]  {len(audit)} works audited, written to {OUT}")

# ---------------------------------------------------------------------------
print("\n-- agreement with the Krumhansl estimate (the test of Section 8.1)")
corroborated = (audit["gap"] == 0).sum()
print(f"  opening annotation corroborated: {corroborated} ({corroborated / len(audit):.1%})")
print("  residual disagreements, by interval:")
for g in range(1, 12):
    n = int((audit["gap"] == g).sum())
    if n:
        note = ""
        if g == SUBDOMINANT_PULL:
            note = "  <- subdominant pull"
        elif g in (3, 8, 10):
            note = "  <- a third or a tone"
        print(f"    +{g:2d} semitones {n:5d}{note}")
print(f"  mean estimation confidence r = {audit['est_corr'].mean():.3f}")

print("\n-- exposure to modulation (no counterpart on the jazz side)")
print(f"  works with a single annotated key: "
      f"{(audit['n_keys'] == 1).sum()} ({(audit['n_keys'] == 1).mean():.1%})")
print(f"  median number of annotated keys: {audit['n_keys'].median():.0f}")
print(f"  duration spent away from the opening collection: "
      f"mean {audit['away_share'].mean():.1%}, median {audit['away_share'].median():.1%}")
print(f"  duration spent a third or a tone away: "
      f"mean {audit['far_share'].mean():.1%}")
for q in (0.5, 0.75, 0.9):
    print(f"    {q:.0%} of works spend at most {audit['away_share'].quantile(q):.0%} away")

print("\n-- by style")
print(f"  {'style':12s}{'n':>6s}{'corrob.':>10s}{'away':>9s}{'far':>8s}")
for style, g in audit.groupby("style"):
    print(f"  {style:12s}{len(g):6d}{(g['gap'] == 0).mean():9.1%}"
          f"{g['away_share'].mean():9.1%}{g['far_share'].mean():8.1%}")

print("\n-- both tests together")
strict = (audit["gap"] == 0) & (audit["away_share"] <= 0.25)
print(f"  corroborated opening AND at most a quarter of the duration away: "
      f"{strict.sum()} of {len(audit)} ({strict.mean():.1%})")

# ---------------------------------------------------------------------------
# What the single-key transposition costs the geometry (Section 8.2).
#
# Reading a work under a tonic it has left spreads its roots over many degrees,
# which records it as chromatic.  Splitting the common-practice side at the
# median of away_share isolates that effect on the axis-2 separation.
from corpus_distances import load_corpus, distance_matrix, key_reliable

songs, titles, song_ids, styles, n_jazz = load_corpus()
reliable = np.concatenate([key_reliable(song_ids[:n_jazz]),
                           np.ones(len(songs) - n_jazz, bool)])
keep = np.flatnonzero(reliable)
D = distance_matrix(songs, "duration")[np.ix_(keep, keep)]
is_jazz = styles[keep] == "jazz"

n = len(D)
centring = np.eye(n) - np.ones((n, n)) / n
gram = -0.5 * centring @ (D ** 2) @ centring
values, vectors = np.linalg.eigh(gram)
order = np.argsort(values)[::-1]
X = vectors[:, order[:2]] * np.sqrt(np.clip(values[order[:2]], 0, None))
axis2, sd = X[:, 1], X[:, 1].std()

cp_rows = audit.set_index("id").reindex(np.array(song_ids)[keep][~is_jazz])
away = cp_rows["away_share"].to_numpy()
cp_idx = np.flatnonzero(~is_jazz)
jazz_mean = axis2[is_jazz].mean()

print("\n-- effect of the single-key transposition on the axis-2 separation")
print(f"  corr(away share, axis 2) = "
      f"{np.corrcoef(away, axis2[cp_idx])[0, 1]:+.3f}   (jazz lies on the + side)")
median = np.median(away)
for label, sel in [("all common-practice works", np.ones(len(away), bool)),
                   ("half closest to opening key", away <= median),
                   ("half furthest from it", away > median)]:
    gap = abs(jazz_mean - axis2[cp_idx[sel]].mean()) / sd
    print(f"    {label:30s} n={sel.sum():4d}   gap = {gap:.2f} sd")
