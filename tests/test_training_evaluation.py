import unittest

from src.evaluation import context_recall, exact_match
from src.training import TrainingExample, split_dataset


class TrainingAndEvaluationTests(unittest.TestCase):
    def test_split_dataset(self):
        dataset = [
            TrainingExample("q1", "a1"),
            TrainingExample("q2", "a2"),
            TrainingExample("q3", "a3"),
            TrainingExample("q4", "a4"),
        ]
        train, test = split_dataset(dataset, train_ratio=0.5)
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)

    def test_exact_match_and_context_recall(self):
        self.assertEqual(exact_match("Resposta", "resposta"), 1.0)
        self.assertEqual(context_recall(["A", "B"], ["B", "C"]), 0.5)


if __name__ == "__main__":
    unittest.main()
