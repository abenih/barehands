import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "train_gesture_classifier.py"
SPEC = importlib.util.spec_from_file_location("trainer", MODULE_PATH)
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class ClassifierTrainerTests(unittest.TestCase):
    def test_sigmoid_is_bounded(self):
        self.assertAlmostEqual(TRAINER.sigmoid(0), 0.5)
        self.assertTrue(math.isfinite(TRAINER.sigmoid(1000)))
        self.assertTrue(math.isfinite(TRAINER.sigmoid(-1000)))

    def test_training_separates_simple_rows(self):
        rows = [
            {"label": "pinch", "x": [-1.0]},
            {"label": "pinch", "x": [-0.8]},
            {"label": "relaxed", "x": [0.8]},
            {"label": "relaxed", "x": [1.0]},
        ]
        model = TRAINER.train(rows, "pinch", epochs=300)
        positive = TRAINER.sigmoid(model["bias"] - model["weights"][0])
        negative = TRAINER.sigmoid(model["bias"] + model["weights"][0])
        self.assertGreater(positive, 0.8)
        self.assertLess(negative, 0.2)


if __name__ == "__main__":
    unittest.main()
