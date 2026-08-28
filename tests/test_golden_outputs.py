"""Byte-for-byte guards for the scientific JSON consumed by the manuscript."""

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "article-data/a-vocabulary-fixed-by-rule.json": "3e733537acb7ffe9c1410740c7f075c9f9187e5adb6030429e7400b18b65c1a8",
    "article-data/borrowed-degrees.json": "994eb2123d92a1d41a32e0aee3b668f14b0fd18e9734d8df7e870dd1914845e9",
    "article-data/reading-from-the-tonic.json": "f66e72d2cc1dab6a347269590f58b1a0ea6132ddba9cb3dc87a85ea9ff56da3e",
    "article-data/the-brightness-contrast.json": "7ce71313692520b6e1bbb4b0d13001069f05c76bc08f1efcb6331ec8832b8249",
    "article-data/the-corpus.json": "484a9c9cdaca83c01b1eb4d86d20f381bf3042bd10fd04512bc35439c4e3c125",
    "article-data/the-degrees-of-a-key.json": "679c3ea24717d729fa47bfab4a47eab596bbdd9b489eb5eabbd31398fd861c94",
    "article-data/the-order-used-below.json": "b28b5b4e63f3c887bbcc917b0805c03a52eeb412d1366a3c35423388c25e1727",
    "article-data/the-p-sensitivity.json": "03f05f14ad282df2288c09d39452823a22fc4c6aa677c8b368d55cdd393b11de",
    "article-data/the-representation-of-a-work.json": "299c56ccd438533fa1a11451e6db05ca1d6f8a0e48f4c00a47e54e3d4ccb65c2",
    "article-data/the-system-used-below.json": "43555e09009e11684b8f6ff9be150e70f2b484019ce437824af817f9b2cd7140",
    "article-data/the-two-repertoires.json": "32b579efe0fe495b5904c58e8e5a663c9478a10ce528bf2aaa035acb96f38c3e",
    "article-data/what-separates-them.json": "7f7ec8871a4fe1e9d53547d49c0b2bf0506749679c9109c9e86e0489e75b0f73",
    "article-data/what-the-reading-separates.json": "5dc8250cd8894e9dcbd33f7e718bec528f2075e9bd6eb1d452289d8f944628b9",
    "article-data/what-the-two-readings-cost.json": "95795e98cad9f63bd7280e38aa5297e3818d57b751141d68227ca71a670c72ad",
    "analysis-data/p-sensitivity.json": "a61c3760030ba22175d9c0d32cdb1e4875aa7ad36c2831272701c7697a7aad47",
}


class TestGoldenOutputs(unittest.TestCase):
    def test_scientific_json_has_not_changed(self):
        for relative, expected in EXPECTED_SHA256.items():
            with self.subTest(path=relative):
                digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                self.assertEqual(digest, expected)


if __name__ == "__main__":
    unittest.main()
