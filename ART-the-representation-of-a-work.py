"""Article data for Section 6.1, "The representation of a work".

The subsection defines the representation and then says what its one
simplification costs.  Both are checked here.

  the simplification   The definition carries a local tonic q_t.  A lead sheet
                       annotates one key for the whole chart, so the section
                       takes q_t = q, and the common-practice partition -- which
                       does annotate the succession -- says what that is worth:
                       the median number of keys per work, the share of works
                       carrying one, the share of a work's duration spent under
                       another collection, and how much of that is a distant
                       modulation rather than a neighbouring one.

  the local tonic      What q_t = q moves, read directly: the common-practice
                       partition times its key annotations, so each chord can be
                       read under the tonic in force at its own onset and the two
                       representations compared.  Half the works do not move at
                       all and the corpus mean shifts by 0.029 in l1, which is a
                       fifth of the gap Section 6.2 measures between repertoires.

  the weighting        Duration against chord count.  Section 6 claims the
                       choice is immaterial, and the correlation between the two
                       mean representations is asserted rather than exported,
                       0.999 being a number the manuscript rounds.

Reads the modulation figures from cache/common_practice_audit.csv, which is a
different audit from cache/common_practice_key_audit.csv: that one asks whether
a work's opening key can be trusted, this one how long the work stays in it.

Run:  LSA_LOCAL=1 .venv/bin/python ART-the-representation-of-a-work.py
"""
import warnings

import jams
import numpy as np
import pandas as pd

from article_data import export
from article_setup import cache_directory
from chord_scale import SYSTEM
from corpus import DATA, _annotation, era, key_exact, load_corpus
from leadsheetanalyser.chords import map_chord
from leadsheetanalyser.constants import NOTE_TO_PC

warnings.filterwarnings("ignore")

ORDER = 0.15
MODULATION = cache_directory() / "common_practice_audit.csv"


def degree_reading(root, kind, cache={}):
    key = (root, kind)
    if key not in cache:
        content = {(root + i) % 12
                   for i in (0,) + tuple(j + 1 for j in range(11) if kind[j])}
        content |= {0}
        intervals = [i - 1 for i in range(1, 12) if i in content]
        m = np.mean(SYSTEM[:, intervals] ** ORDER, axis=1) ** (1 / ORDER)
        cache[key] = m / m.sum()
    return cache[key]


def represent(song, by_duration=True):
    total, weighted = 0.0, np.zeros(9)
    for (root, kind), duration in song:
        w = duration if by_duration else 1.0
        weighted += w * degree_reading(root, kind)
        total += w
    return weighted / total


def local_tonic_shift(read):
    """(share of works unchanged, l1 shift of the corpus mean) under local keys.

    Each chord is read twice: under the opening tonic, as the section does, and
    under the tonic annotated at its own onset.  A chord falling in no annotated
    span keeps the opening tonic, which happens 24 times in the whole partition.
    """
    def tonic_of(value):
        if ":" not in str(value):
            return None
        return NOTE_TO_PC.get(str(value).split(":")[0])

    meta = pd.read_csv(DATA / "meta.csv")
    wir = meta[meta["id"].astype(str).str.startswith("when-in-rome")]
    opening, local = [], []
    for _, row in wir.iterrows():
        if str(row["id"]) not in read or era(row["composers"]) is None:
            continue
        try:
            j = jams.load(str(DATA / "jams_files" / f"{row['id']}.jams"),
                          validate=False)
        except Exception:
            continue
        chords, keys = _annotation(j, "chord_harte"), _annotation(j, "key_mode")
        if chords is None or keys is None or not len(keys.data):
            continue
        spans = [(o.time, o.time + (o.duration or 0), tonic_of(o.value))
                 for o in keys.data]
        spans = [s for s in spans if s[2] is not None]
        if not spans:
            continue
        q0 = spans[0][2]
        a, b, w = np.zeros(9), np.zeros(9), 0.0
        for o in chords.data:
            try:
                c = map_chord(o.value)
            except Exception:
                continue
            if c is None or c[0] is None or any(x is None for x in c[1:12]):
                continue
            d = float(o.duration or 0)
            if d <= 0:
                continue
            root = int(c[0]) % 12
            kind = tuple(1 if int(x) else 0 for x in c[1:12])
            here = next((t for s, e, t in spans if s <= o.time < e), q0)
            a += d * degree_reading((root - q0) % 12, kind)
            b += d * degree_reading((root - here) % 12, kind)
            w += d
        if w > 0:
            opening.append(a / w)
            local.append(b / w)
    A, B = np.array(opening), np.array(local)
    unmoved = (np.abs(A - B).sum(axis=1) < 1e-9).mean()
    return 100 * unmoved, np.abs(A.mean(0) - B.mean(0)).sum()


def main():
    if not MODULATION.exists():
        raise RuntimeError(f"{MODULATION} is missing; run common_practice_audit.py")
    songs, _titles, ids, _styles, n_jazz = load_corpus()
    keep = key_exact(ids)
    read = {str(i) for i, k in zip(ids, keep) if k}

    # on the works Section 6 actually reads, not on all 434: the filter keeps the
    # tonally plainer works, and their modulation figures are the lower ones
    audit = pd.read_csv(MODULATION)
    audit = audit[audit["id"].astype(str).isin(read)]
    keys = audit["n_keys"].dropna()
    away = audit["away_share"].dropna()
    far = audit["far_share"].dropna()

    # the manuscript spells small numbers out, as it does everywhere in prose
    words = {5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    values = {
        "median_keys":
            f"median of {words[int(keys.median())]} annotated key spans",
        "single_key": f"{100 * (keys == 1).mean():.1f}",
        "away_share": f"{100 * away.mean():.1f}",
        "far_share": f"{100 * far.mean():.1f}",
        "no_far": f"{100 * (far < 1e-9).mean():.1f}",
    }
    print(f"\n{len(audit)} common-practice works, those Section 6 reads")
    print(f"   keys per work: median {keys.median():.0f}, mean {keys.mean():.1f}, "
          f"max {int(keys.max())}")
    print(f"   carrying a single key            {100 * (keys == 1).mean():5.1f}%")
    print(f"   duration under another collection {100 * away.mean():5.1f}% on average")
    print(f"   of which a third or a tone away   {100 * far.mean():5.1f}%")
    print(f"   works with no distant modulation  {100 * (far < 1e-9).mean():5.1f}%")

    by_duration, by_count = {"jazz": [], "cp": []}, {"jazz": [], "cp": []}
    for n, (song, k) in enumerate(zip(songs, keep)):
        if not k:
            continue
        where = "jazz" if n < n_jazz else "cp"
        by_duration[where].append(represent(song, True))
        by_count[where].append(represent(song, False))

    print()
    for where in ("jazz", "cp"):
        a = np.array(by_duration[where]).mean(axis=0)
        b = np.array(by_count[where]).mean(axis=0)
        r = np.corrcoef(a, b)[0, 1]
        print(f"   {where:4s} {len(by_duration[where]):5d} works, duration against "
              f"count r = {r:.4f}")
        assert round(r, 3) >= 0.999, (
            f"the two weightings no longer agree to 0.999 in the {where} corpus, "
            "which Section 6 says they do")

    unmoved, shift = local_tonic_shift(read)
    print(f"\n   under locally annotated tonics: {unmoved:.1f}% of works "
          f"unchanged, corpus mean shifted {shift:.4f} in l1")
    values["works_unmoved"] = f"{unmoved:.1f}"
    values["local_shift"] = f"{shift:.3f}"

    export("the-representation-of-a-work", values)


if __name__ == "__main__":
    main()
