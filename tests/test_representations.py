"""Unit tests for shared representation construction."""

import unittest

import numpy as np

from article_analysis.representations import (
    TonicModalReader,
    grouped_work_profiles,
    interval_kind,
    work_profile,
)


class TestRepresentations(unittest.TestCase):
    def setUp(self):
        # Positive entries exercise the same power-mean path as the article
        # without depending on its vocabulary or corpus caches.
        self.system = np.array([
            np.arange(1, 12, dtype=float),
            np.arange(11, 0, -1, dtype=float),
        ])
        self.system /= self.system.sum(axis=1, keepdims=True)
        self.reader = TonicModalReader(self.system, 0.15)
        self.major = tuple(interval_kind((4, 7)))
        self.minor = tuple(interval_kind((3, 7)))

    def test_reader_matches_the_legacy_formula(self):
        root, kind = 7, self.major
        content = {(root + i) % 12 for i in (0, 4, 7)} | {0}
        columns = [i - 1 for i in range(1, 12) if i in content]
        legacy = np.mean(self.system[:, columns] ** 0.15, axis=1) ** (1 / 0.15)
        legacy /= legacy.sum()
        np.testing.assert_allclose(self.reader(root, kind), legacy, atol=1e-12)

    def test_cache_is_owned_by_one_fixed_reader(self):
        first = self.reader(0, self.major)
        second = self.reader(0, self.major)
        self.assertIs(first, second)
        self.assertEqual(self.reader.cached_kinds, 1)

        another_order = TonicModalReader(self.system, 0.20)
        another_order(0, self.major)
        self.assertEqual(another_order.cached_kinds, 1)
        self.assertEqual(self.reader.cached_kinds, 1)

    def test_work_profile_respects_the_selected_weighting(self):
        song = [((0, self.major), 3.0), ((2, self.minor), 1.0)]
        duration = work_profile(song, self.reader, by_duration=True)
        count = work_profile(song, self.reader, by_duration=False)
        expected_duration = (
            3 * self.reader(0, self.major) + self.reader(2, self.minor)
        ) / 4
        expected_count = (
            self.reader(0, self.major) + self.reader(2, self.minor)
        ) / 2
        np.testing.assert_allclose(duration, expected_duration)
        np.testing.assert_allclose(count, expected_count)

    def test_grouping_preserves_work_level_rows(self):
        songs = [
            [((0, self.major), 1.0)],
            [((2, self.minor), 1.0)],
            [((7, self.major), 1.0)],
        ]
        grouped = grouped_work_profiles(
            songs,
            ["j1", "j2", "c1"],
            [True, False, True],
            2,
            {"j1": "major", "c1": "minor"},
            self.reader,
        )
        self.assertEqual(grouped[("J", "major")].shape, (1, 2))
        self.assertEqual(grouped[("C", "minor")].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
