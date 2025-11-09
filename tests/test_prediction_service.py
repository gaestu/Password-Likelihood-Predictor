import os
import tempfile
import unittest

import numpy as np

from app_code.prediction_service import PredictionService


class DummyModel:
    def predict_proba(self, data):
        length = len(data)
        return np.column_stack([np.zeros(length), np.ones(length)])


class PredictionServiceTests(unittest.TestCase):
    def test_predict_returns_scores_for_traditional_model(self):
        service = PredictionService()
        service.loaded_models["dummy.pkl"] = DummyModel()

        predictions = service.predict(["Aa1!", "password"], "dummy.pkl")

        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(isinstance(score, float) for score in predictions))

    def test_list_available_models_only_returns_supported_ext(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "model.pkl"), "w").close()
            open(os.path.join(tmpdir, "model.keras"), "w").close()
            open(os.path.join(tmpdir, "notes.txt"), "w").close()

            service = PredictionService(model_dir=tmpdir)
            models = service.list_available_models()

            self.assertEqual(models, ["model.keras", "model.pkl"])


if __name__ == "__main__":
    unittest.main()
