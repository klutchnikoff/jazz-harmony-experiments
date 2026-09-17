"""Byte-for-byte guards for the scientific JSON consumed by the manuscript."""

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "article-data/a-vocabulary-fixed-by-rule.json": "4ef163f342160a522e64b12856a87776dc09fb01fbc04e1f50247bd62806e08c",
    "article-data/borrowed-degrees.json": "9e744605599657101638a919591ebd952eeca6272712e1384331113b0170a781",
    "article-data/reading-from-the-tonic.json": "e87e88860d4bdb5b83ead1ec8ec94fb8050f6f45ec715aeb35af22865a752122",
    "article-data/the-brightness-contrast.json": "3540147eed3c3440d240848e442dc3ca4a52d22c077f6daf6c595e387eda2a61",
    "article-data/the-corpus.json": "484a9c9cdaca83c01b1eb4d86d20f381bf3042bd10fd04512bc35439c4e3c125",
    "article-data/the-degrees-of-a-key.json": "9f428811a89c6f6bff9f48f9e68aaa5f858a44c2947c40949b3470e9a81de4aa",
    "article-data/the-order-used-below.json": "83dd76c1e7c97c47367ee66af754241dad7b2a8feaa197716afbf56b35fde3d4",
    "article-data/the-p-sensitivity.json": "db2bdd026474c227b52d3f59c3ae808e5d39f411e87bdf6d61c819776a0070d7",
    "article-data/the-representation-of-a-work.json": "c6a010ee8595166c742be463685339f2e642e62ebdba3952bee09b48e60941af",
    "article-data/the-system-used-below.json": "7e1d1d2d2cc2c8143af44e7f36644809bc03cdf82ab6621dd3a0b162c726b79c",
    "article-data/the-two-repertoires.json": "19cb5f3308fc9b09222a3013bb021d29964b660a01931ceb1d7bb8e675bb268c",
    "article-data/what-separates-them.json": "6484699caa70235865f1c3396f42b4c285b2395bf7424c6ed6e04d31e604c88d",
    "article-data/what-the-reading-separates.json": "ead55656ea9b424d82bd13dc42d740dd507b49233f548bc2fbcc153c9a9a3238",
    "article-data/what-the-two-readings-cost.json": "c46b2698a0c86c41752986f32dd8876c2b92025138812754c7ec2ff5f51bdc87",
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
