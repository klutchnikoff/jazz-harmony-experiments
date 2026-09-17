"""Formatting guards for the generated manuscript snapshot."""

import re
import unittest

from article_data import render_tex_values


class TestArticleData(unittest.TestCase):
    def setUp(self):
        self.rendered = render_tex_values({
            "example": {
                "count": 2829,
                "small": 7,
                "modes": ["Lydian", "Ionian", "Dorian"],
                "vector": [0, 1, -2],
                "label": "A&B_1",
            },
        })

    def test_snapshot_has_a_deterministic_digest(self):
        match = re.search(r"article-data-sha256: ([0-9a-f]{64})", self.rendered)
        self.assertIsNotNone(match)
        self.assertEqual(self.rendered, render_tex_values({
            "example": {
                "label": "A&B_1",
                "vector": [0, 1, -2],
                "modes": ["Lydian", "Ionian", "Dorian"],
                "small": 7,
                "count": 2829,
            },
        }))

    def test_numbers_have_prose_table_and_word_forms(self):
        self.assertIn("article@example@count\\endcsname{2{,}829}", self.rendered)
        self.assertIn("article-thin@example@count\\endcsname{2\\,829}", self.rendered)
        self.assertIn("article-word@example@small\\endcsname{seven}", self.rendered)
        self.assertIn("article-word-cap@example@small\\endcsname{Seven}", self.rendered)

    def test_lists_and_tex_special_characters_are_rendered(self):
        self.assertIn("{Lydian, Ionian, and Dorian}", self.rendered)
        self.assertIn("{0,1,-2}", self.rendered)
        self.assertIn(r"{A\&B\_1}", self.rendered)


if __name__ == "__main__":
    unittest.main()
