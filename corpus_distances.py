"""Shared corpus loading and song-distance computation for Sections 8.1 and 8.2.

Both corpus studies rest on one distance matrix, computed here once over the
union of the Real Book and the When-in-Rome partitions of ChoCo.  Section 8.1
reads the Real Book block, Section 8.2 reads the (jazz sample) x (common
practice) block.  Merging them is exact, not an approximation: optimal transport
between two songs only ever consults the sub-cost on the chords those two songs
actually use, so enlarging the shared chord vocabulary leaves every individual
distance unchanged.  It also guarantees the two figures cannot drift apart.

Song measure.  A song is read as a distribution over its rooted chords, each
chord weighted by the *time it occupies* rather than by the number of times it
is written:

    mu_S(c) = (total notated duration of c in S) / (total duration of S).

Normalising within the song makes this free of tempo and of the unit in which
durations happen to be expressed.  It matters most for Section 8.2, where the
two corpora are annotated at different rhythmic granularities -- the Real Book
on a coarse chart grid (median 2 beats), When-in-Rome by fine-grained harmonic
analysis (median 1.5, down to 0.04).  Token counting would let that difference
of editorial convention masquerade as a difference of harmonic style; time
shares are convention-free.

Key normalisation is the collection normalisation of Section 8.1, applied
identically to both corpora: every minor key is mapped to its relative major
before transposition, so pitch class 0 always denotes the tonic of the annotated
diatonic collection.

The distance matrix is cached in tmp/ and reused.
"""
import time
from collections import defaultdict

import numpy as np
import ot
import jams
import pandas as pd

from article_setup import DATA_ROOT, cache_directory
from leadsheetanalyser.chords import map_chord
from leadsheetanalyser.constants import W_DIATONIC, NOTE_TO_PC

DATA = DATA_ROOT
CACHE_DIR = cache_directory()

WD = np.asarray(W_DIATONIC, dtype=float)
BORROWED_DEGREES = {3, 8, 10}  # bIII, bVI, bVII relative to the tonic
MIN_CHORDS = 3


def normalised_tonic(note, mode):
    """Pitch class of the tonic after collection normalisation, or None."""
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
        tonic = normalised_tonic(note, mode)
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
          f"({n_minor} minor annotations mapped to their relative major)")
    return songs, titles, ids


def load_common_practice():
    """When-in-Rome songs with durations, collection-normalised."""
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
        tonic = normalised_tonic(note, mode)
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
    return normalised_tonic(note, mode)


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
    audit_path = CACHE_DIR / "key_audit.csv"
    if not audit_path.exists():
        raise RuntimeError(
            f"{audit_path} is missing; run key_audit.py first "
            "(generate_all.py already orders it before the corpus figures)"
        )
    audit = pd.read_csv(audit_path)
    if "id" not in audit.columns:
        raise RuntimeError(
            f"{audit_path} predates the switch to stable ids; delete it and "
            "re-run key_audit.py"
        )
    gap = {}
    for song_id, annotated, pc, mode in zip(audit["id"], audit["annotated"],
                                            audit["est_pc"], audit["est_mode"]):
        ann = _annotated_collection(annotated)
        if ann is None:
            continue
        est = (int(pc) + 3) % 12 if str(mode) == "minor" else int(pc)
        gap[str(song_id)] = (est - ann) % 12
    return np.array([gap.get(i) in (0, SUBDOMINANT_PULL) for i in song_ids])


# Ground-cost variants.  "duration" and "token" are the article's system W_D and
# differ only in how the song measure is weighted; the other three exist for the
# robustness statement at the end of Section 8.2.  The tags double as cache
# names, so the matrices already computed stay valid.
_SUPPORT = (WD > 0).astype(float)
W_UNIFORM = _SUPPORT / _SUPPORT.sum(axis=1, keepdims=True)

VARIANTS = {
    # tag          (system,    embedding,  song weighting)
    "duration":    (WD,        "linear",   "duration"),
    "token":       (WD,        "linear",   "token"),
    "uniformW":    (W_UNIFORM, "linear",   "duration"),
    "identityW":   (np.eye(11), "linear",  "duration"),
    "angular":     (WD,        "angular",  "duration"),
}


def _reinterpretations(rooted):
    """R[i, r] = the 11-vector of chord i heard as rooted on r."""
    R = np.zeros((len(rooted), 12, 11), dtype=np.int8)
    for i, (root, kind) in enumerate(rooted):
        pcs = {(root + j) % 12 for j in [0] + [j for j, v in enumerate(kind, 1) if v]}
        for rp in range(12):
            for j in range(1, 12):
                if (rp + j) % 12 in pcs:
                    R[i, rp, j - 1] = 1
    return R


def _embed(V, W, mode):
    """V: (..., 11) -> (..., m).  "angular" keeps only the profile's direction."""
    P = V @ W.T
    if mode == "linear":
        return P
    norms = np.linalg.norm(P, axis=-1, keepdims=True)
    return np.divide(P, norms, out=np.zeros_like(P), where=norms > 1e-12)


def chord_cost_matrix(songs, system=WD, embedding="linear", block=256):
    """Ground cost D over the union chord vocabulary.

    Vectorised over the whole vocabulary.  Writing A[i, r] for chord i embedded
    after reinterpretation at root r, and noting that reinterpreting a chord at
    its own root is the identity,

        D(i,j) = || A[i,r_j] - A[j,r_j] || + || A[j,r_i] - A[i,r_i] ||,

    which is the definition of Section 4 with both directional discrepancies
    summed.  Agreement with the package's reference implementation is checked in
    rooted_chord_diagnostics.py.
    """
    chords = sorted({c for song in songs for c, _ in song})
    index = {c: i for i, c in enumerate(chords)}
    roots = np.array([r for r, _ in chords])
    A = _embed(_reinterpretations(chords).astype(float), system, embedding)
    own = A[np.arange(len(chords)), roots]
    C = np.zeros((len(chords), len(chords)))
    for s in range(0, len(chords), block):
        e = min(s + block, len(chords))
        first = np.linalg.norm(A[s:e][:, roots, :] - own[None, :, :], axis=2)
        second = np.linalg.norm(
            A[:, roots[s:e], :].transpose(1, 0, 2) - own[s:e][:, None, :], axis=2)
        C[s:e] = first + second
    np.fill_diagonal(C, 0.0)
    return index, C


def song_measures(songs, index, weighting="duration"):
    """(support, weights) per song, weighted by duration or by token count."""
    vecs = []
    for song in songs:
        mass = defaultdict(float)
        for chord, duration in song:
            mass[index[chord]] += duration if weighting == "duration" else 1.0
        support = np.fromiter(mass.keys(), int)
        weights = np.fromiter(mass.values(), float)
        vecs.append((support, weights / weights.sum()))
    return vecs


def pairwise_songdist(vecs, C, label=""):
    N = len(vecs)
    D = np.zeros((N, N))
    start = time.time()
    for i in range(N):
        ii, wi = vecs[i]
        for j in range(i + 1, N):
            jj, wj = vecs[j]
            D[i, j] = D[j, i] = ot.emd2(wi, wj, C[np.ix_(ii, jj)])
        if i % 200 == 0:
            done = i * N - i * (i + 1) / 2
            total = N * (N - 1) / 2
            elapsed = time.time() - start
            eta = elapsed * (total / done - 1) if done else float("nan")
            print(f"  {label}row {i}/{N}  {elapsed:6.0f}s elapsed, ETA {eta:6.0f}s",
                  flush=True)
    return D


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


def distance_matrix(songs, variant="duration"):
    """Cached pairwise song-distance matrix over the whole union corpus.

    `variant` selects one of VARIANTS and names the cache file.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of "
                         f"{sorted(VARIANTS)}")
    system, embedding, weighting = VARIANTS[variant]
    cache = CACHE_DIR / f"union_songdist_{variant}_{len(songs)}.npy"
    if cache.exists():
        print(f"loading cached distance matrix {cache.name}")
        return np.load(cache)
    print(f"computing {len(songs)}x{len(songs)} song dissimilarities "
          f"(variant {variant}: {embedding} embedding, {weighting} weighting)...")
    index, C = chord_cost_matrix(songs, system, embedding)
    print(f"  ground cost over {len(index)} distinct chords done")
    vecs = song_measures(songs, index, weighting)
    D = pairwise_songdist(vecs, C)
    np.save(cache, D)
    print(f"  saved {cache.name}")
    return D


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


def classical_mds(D):
    N = len(D)
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    X = eigvecs[:, :2] * np.sqrt(np.clip(eigvals[:2], 0, None))
    return X, eigvals[:2].sum() / eigvals[eigvals > 0].sum()
