import unittest

from qreason.hard_boolean import closest_real_width, make_hard_boolean_task
from qreason.structured_boolean import StructuredRealControl
from qreason.trainable import parameter_count
from qreason.unique_boolean import verify_unique


class HardBooleanTests(unittest.TestCase):
    def test_near_miss_generator_has_unique_solution(self) -> None:
        task = make_hard_boolean_task(
            samples=32, candidate_count=8, variable_count=8, clause_count=12, seed=61
        )
        self.assertTrue(verify_unique(task))

    def test_real_width_is_close_to_target(self) -> None:
        width = closest_real_width(37, 324)
        count = parameter_count(StructuredRealControl(37, 8, width))
        self.assertLessEqual(abs(count - 324), 40)


if __name__ == "__main__":
    unittest.main()
