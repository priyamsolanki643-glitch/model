import unittest

from qreason.unique_boolean import make_unique_boolean_task, verify_unique


class UniqueBooleanTests(unittest.TestCase):
    def test_every_example_has_exactly_one_valid_candidate(self) -> None:
        task = make_unique_boolean_task(
            samples=64,
            candidate_count=7,
            variable_count=5,
            seed=77,
        )
        self.assertTrue(verify_unique(task))

    def test_generation_is_reproducible(self) -> None:
        first = make_unique_boolean_task(samples=12, seed=78)
        second = make_unique_boolean_task(samples=12, seed=78)
        self.assertTrue((first.candidates == second.candidates).all())
        self.assertTrue((first.clauses == second.clauses).all())
        self.assertTrue((first.labels == second.labels).all())


if __name__ == "__main__":
    unittest.main()
