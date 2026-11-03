import unittest

from qreason.raw_boolean_multi_seed import run_raw_multi_seed


class RawBooleanMultiSeedTests(unittest.TestCase):
    def test_parameter_matched_report(self) -> None:
        report = run_raw_multi_seed(seeds=(2, 3))
        self.assertEqual(report["seeds"], 2)
        self.assertEqual(report["parameter_delta"], -3)
        self.assertGreaterEqual(report["qreason"]["accuracy"]["mean"], 0.0)
        self.assertLessEqual(report["qreason"]["accuracy"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
