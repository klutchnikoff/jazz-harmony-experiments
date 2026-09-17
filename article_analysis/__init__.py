"""Reusable analysis primitives shared by the article's executable scripts."""

import article_setup  # noqa: F401  (select package source before submodule imports)

from .key_metadata import load_annotated_modes
from .representations import (
    TonicModalReader,
    grouped_work_profiles,
    interval_kind,
    work_profile,
)
from .repertoire_comparison import (
    profile_permutation_test,
    profile_stream,
)

__all__ = [
    "TonicModalReader",
    "grouped_work_profiles",
    "interval_kind",
    "load_annotated_modes",
    "profile_permutation_test",
    "profile_stream",
    "work_profile",
]
