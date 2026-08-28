"""Tests for the shared key-audit table reader."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from article_analysis.key_metadata import AUDIT_TABLES, load_annotated_modes


class TestKeyMetadata(unittest.TestCase):
    def test_both_audits_are_combined_by_identifier(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"id": [1, 2], "annotated": ["C:major", "A:minor"]}).to_csv(
                root / AUDIT_TABLES[0], index=False
            )
            pd.DataFrame({"id": ["w1"], "annotated": ["E:min"]}).to_csv(
                root / AUDIT_TABLES[1], index=False
            )
            self.assertEqual(
                load_annotated_modes(root),
                {"1": "major", "2": "minor", "w1": "minor"},
            )


if __name__ == "__main__":
    unittest.main()
