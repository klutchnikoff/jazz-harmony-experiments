"""Pure construction and aggregation of the article's modal representations."""

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from leadsheetanalyser.chord_dissimilarities import tonic_modal_profile


def interval_kind(intervals: Iterable[int]) -> np.ndarray:
    """Build an eleven-coordinate binary kind from semitone intervals 1--11."""
    kind = np.zeros(11, dtype=int)
    for interval in intervals:
        if not isinstance(interval, (int, np.integer)) or not 1 <= int(interval) <= 11:
            raise ValueError("Chord intervals must be integers in range 1-11")
        kind[int(interval) - 1] = 1
    return kind


class TonicModalReader:
    """Cached modal reading for one fixed system, order, and reference tonic.

    A reader owns its cache, so cached values can never leak between different
    orders or modal systems.  This replaces the mutable default dictionaries
    formerly repeated in the executable scripts.
    """

    def __init__(self, system: np.ndarray, order: float, tonic: int = 0):
        self.system = np.asarray(system, dtype=float)
        self.order = float(order)
        self.tonic = int(tonic)
        self._cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    def __call__(self, root: int, kind: Sequence[int]) -> np.ndarray:
        key = (int(root), tuple(int(bit) for bit in kind))
        if key not in self._cache:
            profile = tonic_modal_profile(
                key[0],
                np.asarray(key[1], dtype=int),
                self.system,
                self.order,
                self.tonic,
            )
            if profile is None:
                raise ValueError("Phi_p(0) is undefined")
            self._cache[key] = profile
        return self._cache[key]

    def from_intervals(self, root: int, intervals: Iterable[int]) -> np.ndarray:
        """Read a chord whose kind is given as semitone interval numbers."""
        return self(root, interval_kind(intervals))

    @property
    def cached_kinds(self) -> int:
        """Number of distinct rooted kinds currently cached."""
        return len(self._cache)


def work_profile(
    song,
    reader: TonicModalReader,
    by_duration: bool = True,
) -> np.ndarray:
    """Weighted mean modal profile of one work."""
    total = 0.0
    weighted = np.zeros(reader.system.shape[0], dtype=float)
    for (root, kind), duration in song:
        weight = float(duration) if by_duration else 1.0
        weighted += weight * reader(root, kind)
        total += weight
    if total <= 0:
        raise ValueError("A work must contain at least one positively weighted chord")
    return weighted / total


def grouped_work_profiles(
    songs,
    ids,
    keep,
    n_jazz: int,
    modes: dict[str, str],
    reader: TonicModalReader,
) -> dict[tuple[str, str], np.ndarray]:
    """Per-work profiles grouped by corpus (J/C) and annotated key mode."""
    works = defaultdict(list)
    for index, (song, retained, song_id) in enumerate(zip(songs, keep, ids)):
        if not retained:
            continue
        corpus = "J" if index < n_jazz else "C"
        mode = modes.get(str(song_id), "major")
        works[(corpus, mode)].append(work_profile(song, reader))
    return {key: np.asarray(value) for key, value in works.items()}
