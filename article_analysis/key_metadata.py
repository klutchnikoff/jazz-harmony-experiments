"""Read the versioned key-audit tables used to stratify article analyses."""

from pathlib import Path

import pandas as pd


AUDIT_TABLES = ("key_audit.csv", "common_practice_key_audit.csv")


def load_annotated_modes(cache_root: Path) -> dict[str, str]:
    """Map work identifiers to ``major`` or ``minor`` from both audit tables."""
    modes = {}
    for name in AUDIT_TABLES:
        table = pd.read_csv(Path(cache_root) / name)
        for song_id, annotated in zip(table["id"], table["annotated"]):
            minor = isinstance(annotated, str) and "min" in annotated.lower()
            modes[str(song_id)] = "minor" if minor else "major"
    return modes
