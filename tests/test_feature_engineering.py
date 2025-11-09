import os
import tempfile
import unittest
from unittest import mock

import joblib
import pandas as pd

from app_code.feature_engineering import extract_features
from app_code.scale_features import scale_features_from_stored


class ShiftScaler:
    def __init__(self, shift):
        self.shift = shift

    def transform(self, data):
        return data.to_numpy() - self.shift


class IdentityScaler:
    def transform(self, data):
        return data


class FeatureEngineeringTests(unittest.TestCase):
    def test_extract_features_counts_and_ratios(self):
        features = extract_features("Aa1!")

        self.assertEqual(features["digit_count"], 1)
        self.assertEqual(features["uppercase_count"], 1)
        self.assertEqual(features["special_char_count"], 1)
        self.assertEqual(features["unique_char_count"], 4)
        self.assertAlmostEqual(features["digit_ratio"], 0.25)
        self.assertAlmostEqual(features["uppercase_proportion"], 0.25)
        self.assertAlmostEqual(features["special_char_ratio"], 0.25)

    def test_extract_features_empty_string_defaults(self):
        features = extract_features("")

        self.assertEqual(features["digit_count"], 0)
        self.assertEqual(features["entropy"], 0)
        self.assertEqual(features["unique_char_ratio"], 0)
        self.assertEqual(features["max_consecutive_repeats"], 1)


class ScaleFeaturesTests(unittest.TestCase):
    def test_scale_features_uses_available_scaler(self):
        df = pd.DataFrame({"digit_count": [5], "digit_ratio": [0.5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, "digit_count_scaler.pkl")

            joblib.dump(ShiftScaler(5), scaler_path)

            scaled = scale_features_from_stored(df, scaler_path=tmpdir)

            self.assertIn("digit_count", scaled.columns)
            self.assertIn("digit_ratio", scaled.columns)
            self.assertAlmostEqual(scaled.loc[df.index[0], "digit_count"], 0.0, places=6)
            self.assertEqual(scaled.loc[df.index[0], "digit_ratio"], 0.5)

    def test_scale_features_uses_cache_for_scalers(self):
        df = pd.DataFrame({"digit_count": [1], "digit_ratio": [0.1]})

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, "digit_count_scaler.pkl")

            joblib.dump(IdentityScaler(), scaler_path)

            with mock.patch("app_code.scale_features.joblib.load", wraps=joblib.load) as mocked_load:
                scale_features_from_stored(df, scaler_path=tmpdir)
                scale_features_from_stored(df, scaler_path=tmpdir)

                self.assertEqual(mocked_load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
