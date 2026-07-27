"""Loading the two corpora, and the key filter that defines the working set.

Carried over from the v0.2 pipeline, with one change: songs are transposed to
their annotated tonic and no longer to the tonic of its collection.  The parsing
is delicate -- durations read alongside the chords, the
annotated key corroborated against a Krumhansl estimate -- and rewriting it would
risk a silently different corpus.  Everything downstream of the loading is new.

`key_reliable` and `key_exact` read cache/key_audit.csv, produced by key_audit.py.
"""
from pathlib import Path

import numpy as np
import jams
import pandas as pd

from article_setup import DATA_ROOT, cache_directory
from leadsheetanalyser.chords import map_chord
from leadsheetanalyser.constants import NOTE_TO_PC

DATA = DATA_ROOT
CACHE_DIR = cache_directory()

BORROWED_DEGREES = {3, 8, 10}  # bIII, bVI, bVII relative to the tonic
MIN_CHORDS = 3


def annotated_tonic(note, mode):
    """Pitch class of the annotated tonic, or None.

    The tonic itself, not the tonic of its collection: a piece in A minor has
    tonic A.  Until v0.3 this returned the relative major for minor keys, which
    was harmless while only kinds were read, a kind being transposition
    invariant.  Section 5 reads a chord from the tonic, and the seven diatonic
    modes are the seven rotations of one collection, so folding minor onto its
    relative major would put Aeolian beyond reach by construction.
    """
    del mode  # the mode no longer moves the tonic; kept for call-site symmetry
    return NOTE_TO_PC.get(note)


def collection_tonic(note, mode):
    """Pitch class of the tonic of the collection, minor mapped to its relative
    major.  What the collection-level agreement of key_reliable() compares."""
    tonic = NOTE_TO_PC.get(note)
    if tonic is None:
        return None
    if str(mode).strip().lower().startswith("min"):
        tonic = (tonic + 3) % 12
    return tonic


def _rooted_with_durations(progression, durations, tonic):
    """[(rooted chord, duration)] for the chords that parse cleanly."""
    out = []
    for chord, duration in zip(progression, durations):
        if chord is None or chord[0] is None or any(x is None for x in chord[1:12]):
            continue
        if duration is None or not np.isfinite(duration) or duration <= 0:
            continue
        root = int(chord[0])
        if 0 <= root <= 11:
            kind = tuple(1 if int(x) else 0 for x in chord[1:12])
            out.append((((root - tonic) % 12, kind), float(duration)))
    return out


def era(composer):
    c = str(composer)
    if any(x in c for x in ["Bach", "Monteverdi", "Purcell"]):
        return "baroque"
    if any(x in c for x in ["Beethoven", "Haydn", "Mozart"]):
        return "classical"
    if any(x in c for x in ["Schubert", "Schumann", "Wolf", "Reichardt", "Hensel",
                            "Lang", "Mahler", "Coleridge", "Mayer", "Chaminade",
                            "White", "Brahms", "Mendelssohn"]):
        return "romantic"
    return None


def _annotation(j, namespace):
    for a in j.annotations:
        if a.namespace == namespace:
            return a
    return None


def load_real_book():
    """Real Book songs with durations, collection-normalised."""
    df = pd.read_pickle(DATA / "music_realbook.pkl")
    if "duration_progression" not in df.columns:
        raise RuntimeError(
            "music_realbook.pkl carries no durations; regenerate it with "
            "leadsheetanalyser/scripts/build_corpus.py"
        )
    songs, titles, ids, n_minor = [], [], [], 0
    for _, row in df.iterrows():
        key = row["key"]
        if not (isinstance(key, list) and key and isinstance(key[0], str) and ":" in key[0]):
            continue
        note, mode = key[0].split(":")[0], key[0].split(":")[1]
        tonic = annotated_tonic(note, mode)
        if tonic is None:
            continue
        if str(mode).strip().lower().startswith("min"):
            n_minor += 1
        song = _rooted_with_durations(
            row["chord_progression"], row["duration_progression"], tonic
        )
        if len(song) >= MIN_CHORDS:
            songs.append(song)
            titles.append(str(row["title"]))
            ids.append(str(row["id"]))
    print(f"  Real Book: {len(songs)} songs "
          f"({n_minor} of them annotated in a minor key, read from that tonic)")
    return songs, titles, ids


def load_common_practice():
    """When-in-Rome songs with durations, transposed to their annotated tonic."""
    meta = pd.read_csv(DATA / "meta.csv")
    wir = meta[meta["id"].astype(str).str.startswith("when-in-rome")]
    songs, titles, styles, ids = [], [], [], []
    for _, row in wir.iterrows():
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
        key0 = str(keys.data[0].value)
        if ":" not in key0:
            continue
        note, mode = key0.split(":")[0], key0.split(":")[1]
        tonic = annotated_tonic(note, mode)
        if tonic is None:
            continue
        progression, durations = [], []
        for obs in chords.data:
            try:
                progression.append(map_chord(obs.value))
            except Exception:
                progression.append(None)
            durations.append(obs.duration)
        song = _rooted_with_durations(progression, durations, tonic)
        if len(song) >= MIN_CHORDS:
            songs.append(song)
            titles.append(str(row.get("title", row["id"])))
            styles.append(style)
            ids.append(str(row["id"]))
    print(f"  common practice: {len(songs)} songs")
    return songs, titles, styles, ids


SUBDOMINANT_PULL = 5  # semitones: estimate a fourth above the annotation


def _annotated_collection(label):
    """Collection tonic of an annotated key string, minor mapped to relative major."""
    if not isinstance(label, str) or ":" not in label:
        return None
    note, mode = label.split(":")[0], label.split(":")[1]
    return collection_tonic(note, mode)


def classify(annot, est):
    """How an annotated key stands to an estimated one.

    Lives here rather than in key_audit.py because both producers need it and
    that module does its whole analysis at import time.

      exact     same tonic, same mode -- what the key-dependent sections require
      relative  the annotation names the relative major or minor of the estimate,
                so the collection agrees and only the centre is in dispute
      parallel  same tonic, opposite mode
      fifth     tonics a fourth or fifth apart
      other     anything else
    """
    (apc, amode), (epc, emode, _) = annot, est
    if apc == epc and amode == emode:
        return "exact"
    if emode == "minor" and amode == "major" and apc == (epc + 3) % 12:
        return "relative"
    if emode == "major" and amode == "minor" and apc == (epc + 9) % 12:
        return "relative"
    if apc == epc:
        return "parallel"
    if (apc - epc) % 12 in (5, 7):
        return "fifth"
    return "other"


AUDITS = {
    "key_audit.csv": "key_audit.py",                       # the Real Book
    "common_practice_key_audit.csv": "common_practice_key_audit.py",
}


def _audit_table(required=False):
    """The key audits of both corpora, concatenated, with their guards.

    The two producers run separately, so a missing table is not fatal unless
    every one is missing; a song absent from what was loaded is simply not
    corroborated, which errs towards excluding it.  The freshness guard is not
    optional, a filter resting on an audit older than its producer being worse
    than no filter at all.
    """
    frames = []
    for name, producer_name in AUDITS.items():
        audit_path = CACHE_DIR / name
        if not audit_path.exists():
            if required:
                raise RuntimeError(
                    f"{audit_path} is missing; run {producer_name} first"
                )
            continue
        producer = Path(__file__).resolve().parent / producer_name
        if producer.exists() and producer.stat().st_mtime > audit_path.stat().st_mtime:
            raise RuntimeError(
                f"{audit_path} is older than {producer_name}; re-run it, "
                "or the filters below rest on a superseded analysis"
            )
        frame = pd.read_csv(audit_path)
        if "id" not in frame.columns:
            raise RuntimeError(
                f"{audit_path} predates the switch to stable ids; delete it "
                f"and re-run {producer_name}"
            )
        frames.append(frame)
    if not frames:
        raise RuntimeError("no key audit found; run the producers first")
    return pd.concat(frames, ignore_index=True)


def key_reliable(song_ids):
    """Mask over `song_ids`: is the annotated key corroborated? (Section 8.1)

    Kept are the songs whose annotated and Krumhansl-estimated keys designate
    the same diatonic collection, plus those whose estimate lies a fourth above
    the annotation.  That second group is the subdominant pull a prevalent
    flattened seventh exerts on any profile-matching method; in blues-inflected
    repertoire the annotation is the reading to trust, so those songs stay.

    Everything else is discarded, because a tonic wrong by a third relabels the
    tonic, dominant and supertonic as bVI, bIII and bVII -- manufacturing the
    borrowed-degree share rather than merely perturbing it.

    Reads tmp/key_audit.csv, produced by key_audit.py.
    """
    audit = _audit_table()
    gap = {}
    for song_id, annotated, pc, mode in zip(audit["id"], audit["annotated"],
                                            audit["est_pc"], audit["est_mode"]):
        ann = _annotated_collection(annotated)
        if ann is None:
            continue
        est = (int(pc) + 3) % 12 if str(mode) == "minor" else int(pc)
        gap[str(song_id)] = (est - ann) % 12
    return np.array([gap.get(i) in (0, SUBDOMINANT_PULL) for i in song_ids])


def key_exact(song_ids):
    """Mask over `song_ids`: does the Krumhansl estimate confirm tonic and mode?

    Stricter than key_reliable(), which corroborates the collection only and so
    passes the songs whose annotation names the relative major of a minor tune.
    That was harmless while only kinds were read.  Section 5 reads a chord from
    the tonic, so the centre has to be right and not merely the collection.

    The Real Book key annotations are crowd-sourced -- ChoCo records them as
    such -- and no authoritative source for them exists.  Measured against the
    Weimar Jazz Database, the one expert annotation with a usable overlap, this
    filter cuts the share of annotated-major songs that Weimar hears as minor
    from 11 of 114 to 1 of 60.  The sample is too small to rank this filter
    against the alternatives tried, and the figure is agreement between
    annotators, not accuracy.

    Reads cache/key_audit.csv, produced by key_audit.py.
    """
    audit = _audit_table()
    exact = {str(i): c == "exact"
             for i, c in zip(audit["id"], audit["category"])}
    return np.array([exact.get(str(i), False) for i in song_ids])


# Ground-cost variants.  "duration" and "token" are the article's system W_D and
# differ only in how the song measure is weighted; the other three exist for the
# robustness statement at the end of Section 8.2.  The tags double as cache
# names, so the matrices already computed stay valid.

def load_corpus():
    """The union corpus: songs, titles, stable ids, style labels, jazz count.

    Titles are for display only -- thirteen of them are shared by two or more
    Real Book entries, some carrying different keys ("Mr" occurs six times,
    a casualty of titles truncated at a full stop).  Anything that has to match
    a song against external data, key_reliable() above all, must use the ids.
    """
    print("loading corpora...")
    jazz, jazz_titles, jazz_ids = load_real_book()
    cp, cp_titles, cp_styles, cp_ids = load_common_practice()
    songs = jazz + cp
    titles = jazz_titles + cp_titles
    ids = jazz_ids + cp_ids
    styles = np.array(["jazz"] * len(jazz) + cp_styles)
    return songs, titles, ids, styles, len(jazz)



def descriptive_statistics(songs, weighting="duration"):
    """Per-song descriptors, weighted consistently with the song measure.

    Returns (n_distinct, seventh share, borrowed share, mean chord size).  Chord
    size counts NOTES, root included, so a plain triad measures 3: the kind
    vector records the intervals above the root, hence the +1.
    """
    n_distinct, seventh, borrowed, mean_size = [], [], [], []
    for song in songs:
        w = np.array([d if weighting == "duration" else 1.0 for _, d in song])
        w = w / w.sum()
        kinds = [k for (_, k), _ in song]
        roots = [r for (r, _), _ in song]
        n_distinct.append(len({c for c, _ in song}))
        seventh.append(float(np.sum(w * np.array([1.0 if (k[9] or k[10]) else 0.0
                                                  for k in kinds]))))
        borrowed.append(float(np.sum(w * np.array([1.0 if r in BORROWED_DEGREES else 0.0
                                                   for r in roots]))))
        mean_size.append(float(np.sum(w * np.array([sum(k) + 1.0 for k in kinds]))))
    return (np.array(n_distinct), np.array(seventh),
            np.array(borrowed), np.array(mean_size))


