import unittest

from qreason.multi_seed import run_multi_seed


class MultiSeedTests(unittest.TestCase):
    def test_multi_seed_report_has_expected_shape(self) -> None:
        report = run_multi_seed(seeds=(2, 3))
        result = report.as_dict()
        self.assertEqual(result["seeds"], 2)
        self.assertEqual(len(report.qreason_accuracy), 2)
        self.assertEqual(len(report.real_accuracy), 2)
        self.assertEqual(report.qreason_parameters, 665)
        self.assertEqual(report.real_parameters, 668)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in report.qreason_accuracy))


if __name__ == "__main__":
    unittest.main()
