import unittest
from unittest.mock import patch

from app import app, prediction_service


class AppRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_accepts_json_payload(self):
        with patch.object(prediction_service, "predict", return_value=[0.1, 0.2]):
            response = self.client.post(
                "/predict",
                json={"model": "demo.pkl", "words": ["one", "two"]},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["predictions"]), 2)
        self.assertEqual(payload["predictions"][0]["word"], "one")

    def test_predict_missing_model_returns_400(self):
        response = self.client.post("/predict", json={"words": ["one"]})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
