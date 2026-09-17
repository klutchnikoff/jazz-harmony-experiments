"""Shared work-level permutation inference for the repertoire comparison."""

from typing import NamedTuple

import numpy as np


PROFILE_SEED = 20260729
KEY_TYPES = {"major": 0, "minor": 1}


class ProfilePermutationResult(NamedTuple):
    observed: np.ndarray
    l1_gap: float
    l1_exceedances: int
    adjusted_l1_p: float
    adjusted_coordinate_p: np.ndarray


def profile_stream(key_type: str) -> np.random.Generator:
    """Deterministic label-allocation stream fixed by the article protocol."""
    return np.random.Generator(
        np.random.PCG64([PROFILE_SEED, KEY_TYPES[key_type]])
    )


def profile_permutation_test(
    jazz: np.ndarray,
    common_practice: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
    multiplicity: int = 1,
) -> ProfilePermutationResult:
    """Joint l1 and max-T permutation test on two sets of work profiles.

    One label reallocation supplies both null statistics, preserving their
    dependence and the exact random-number consumption of the article analysis.
    ``multiplicity`` applies the between-group Bonferroni correction.
    """
    jazz = np.asarray(jazz, dtype=float)
    common_practice = np.asarray(common_practice, dtype=float)
    if jazz.ndim != 2 or common_practice.ndim != 2:
        raise ValueError("Both profile samples must be two-dimensional")
    if jazz.shape[1] != common_practice.shape[1]:
        raise ValueError("The two samples must use the same coordinates")
    if permutations < 1 or multiplicity < 1:
        raise ValueError("permutations and multiplicity must be positive")

    observed = common_practice.mean(0) - jazz.mean(0)
    gap = float(np.abs(observed).sum())
    both = np.vstack([jazz, common_practice])
    null_l1 = np.empty(permutations)
    null_max = np.empty(permutations)
    for index in range(permutations):
        allocation = rng.permutation(len(both))
        difference = (
            both[allocation[len(jazz):]].mean(0)
            - both[allocation[:len(jazz)]].mean(0)
        )
        null_l1[index] = np.abs(difference).sum()
        null_max[index] = np.abs(difference).max()

    l1_exceedances = int(np.count_nonzero(null_l1 >= gap))
    adjusted_l1 = min(
        multiplicity
        * (1 + l1_exceedances)
        / (permutations + 1),
        1,
    )
    adjusted_coordinates = np.array([
        min(
            multiplicity
            * (1 + int(np.count_nonzero(null_max >= abs(value))))
            / (permutations + 1),
            1,
        )
        for value in observed
    ])
    return ProfilePermutationResult(
        observed,
        gap,
        l1_exceedances,
        adjusted_l1,
        adjusted_coordinates,
    )
