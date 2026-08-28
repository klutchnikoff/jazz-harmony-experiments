"""Tests for the shared repertoire permutation statistic."""

import unittest

import numpy as np

from article_analysis.repertoire_comparison import (
    profile_permutation_test,
    profile_stream,
)


class TestRepertoireComparison(unittest.TestCase):
    def test_stream_restarts_for_each_key_type(self):
        first = profile_stream("major").integers(0, 100, 8)
        second = profile_stream("major").integers(0, 100, 8)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, profile_stream("minor").integers(0, 100, 8)))

    def test_result_contains_l1_and_max_t_adjustments(self):
        jazz = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
        common = np.array([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
        result = profile_permutation_test(
            jazz,
            common,
            permutations=99,
            rng=np.random.default_rng(123),
            multiplicity=2,
        )
        np.testing.assert_allclose(result.observed, [-0.6, 0.6])
        self.assertAlmostEqual(result.l1_gap, 1.2)
        self.assertGreaterEqual(result.adjusted_l1_p, 2 / 100)
        self.assertEqual(result.adjusted_coordinate_p.shape, (2,))


if __name__ == "__main__":
    unittest.main()
