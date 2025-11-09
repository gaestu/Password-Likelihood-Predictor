import os
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json

from app_code.feature_engineering import extract_features
from app_code.scale_features import scale_features_from_stored


class PredictionService:
    """
    Central orchestration layer for loading models, preparing inputs, and
    generating predictions in batch. Keeps inference fast by caching models,
    tokenizers, and scalers.
    """

    def __init__(
        self,
        model_dir: str = "model",
        scaler_dir: str = "model/scalers",
        tokenizer_vocab_size: int = 50000,
    ) -> None:
        self.model_dir = model_dir
        self.scaler_dir = scaler_dir
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.loaded_models = {}
        self.tokenizer_cache = {}

    def list_available_models(self) -> List[str]:
        if not os.path.isdir(self.model_dir):
            return []
        candidates: List[str] = []
        for filename in os.listdir(self.model_dir):
            if not os.path.isfile(os.path.join(self.model_dir, filename)):
                continue
            if filename.endswith(".pkl") or filename.endswith(".keras"):
                candidates.append(filename)
        return sorted(candidates)

    def predict(self, words: Sequence[str], model_name: str) -> List[float]:
        if not words:
            return []

        model_payload = self._load_model(model_name)
        model = self._extract_core_model(model_payload)

        if isinstance(model, tf.keras.Model):
            return self._predict_with_keras_model(words, model_name, model_payload, model)

        return self._predict_with_traditional_model(words, model)

    # ---- internal helpers -------------------------------------------------

    def _load_model(self, model_name: str):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")

        if model_name.endswith(".pkl"):
            model = joblib.load(model_path)
        elif model_name.endswith(".keras"):
            model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model extension for {model_name}")

        self.loaded_models[model_name] = model
        return model

    def _predict_with_traditional_model(self, words: Sequence[str], model) -> List[float]:
        feature_rows = [extract_features(word) for word in words]
        features_df = pd.DataFrame(feature_rows)
        scaled_features = scale_features_from_stored(features_df, scaler_path=self.scaler_dir)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_features)[:, 1]
        elif hasattr(model, "predict"):
            proba = model.predict(scaled_features)
        else:
            raise AttributeError("Model must implement predict_proba or predict.")

        return [float(score) for score in np.asarray(proba).flatten()]

    def _predict_with_keras_model(
        self,
        words: Sequence[str],
        model_name: str,
        model_payload,
        model: tf.keras.Model,
    ) -> List[float]:
        tokenizer = self._resolve_tokenizer(model_name, model_payload, words)
        expected_length = self._resolve_sequence_length(model)

        sequences = tokenizer.texts_to_sequences(words)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=expected_length,
            padding="post",
            truncating="post",
        )

        if isinstance(model.input_shape, list) and len(model.input_shape) == 2:
            feature_rows = [extract_features(word) for word in words]
            features_df = pd.DataFrame(feature_rows)
            scaled_features = scale_features_from_stored(
                features_df, scaler_path=self.scaler_dir
            )
            network_input = [padded_sequences, scaled_features.to_numpy()]
        else:
            network_input = padded_sequences

        predictions = model.predict(network_input, verbose=0)
        return [float(score) for score in np.asarray(predictions).flatten()]

    def _resolve_tokenizer(
        self,
        model_name: str,
        model_payload,
        words: Sequence[str],
    ) -> Tokenizer:
        """
        Returns a tokenizer for the given model. Prefers persisted tokenizers,
        falls back to fitting on the incoming batch for backward compatibility.
        """
        if model_name in self.tokenizer_cache:
            return self.tokenizer_cache[model_name]

        tokenizer = None

        if isinstance(model_payload, dict) and "tokenizer" in model_payload:
            tokenizer = model_payload["tokenizer"]
        else:
            base_name = os.path.splitext(model_name)[0]
            pkl_path = os.path.join(self.model_dir, f"{base_name}_tokenizer.pkl")
            json_path = os.path.join(self.model_dir, f"{base_name}_tokenizer.json")

            if os.path.exists(pkl_path):
                tokenizer = joblib.load(pkl_path)
            elif os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as fh:
                    tokenizer = tokenizer_from_json(fh.read())

        if tokenizer is None:
            tokenizer = Tokenizer(char_level=True, num_words=self.tokenizer_vocab_size)
            tokenizer.fit_on_texts(words)

        self.tokenizer_cache[model_name] = tokenizer
        return tokenizer

    @staticmethod
    def _extract_core_model(model_payload):
        if isinstance(model_payload, dict) and "model" in model_payload:
            return model_payload["model"]
        return model_payload

    @staticmethod
    def _resolve_sequence_length(model: tf.keras.Model) -> int:
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            sequence_shape = input_shape[0]
        else:
            sequence_shape = input_shape

        if not sequence_shape or len(sequence_shape) < 2:
            raise ValueError("Could not determine expected sequence length from model.")

        maxlen = sequence_shape[1]
        if maxlen is None:
            raise ValueError("Model input shape must define sequence length.")
        return int(maxlen)
