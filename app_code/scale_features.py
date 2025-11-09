import os

import joblib
import pandas as pd

_SCALER_CACHE = {}


def scale_features_from_stored(data_frame, scaler_path='model/scalers'):
    """
    Scales the features of the input DataFrame using stored scalers.

    Parameters:
    data_frame (pd.DataFrame): The input data frame with raw features.
    scaler_path (str, optional): Path where the scaler files are stored. Default is 'model/scalers'.

    Returns:
    pd.DataFrame: DataFrame with scaled features.
    """
    normalized_data = pd.DataFrame(index=data_frame.index)  # Create an empty DataFrame with the same index

    for feature_name in data_frame.columns:
        scaler = _get_scaler(scaler_path, feature_name)

        if scaler is not None:
            feature_data = data_frame[[feature_name]]  # Retain the feature name as a DataFrame
            normalized_feature = scaler.transform(feature_data)  # Transform the feature
            normalized_data[feature_name] = normalized_feature  # Add the normalized feature
        else:
            # If there's no scaler, just copy the original feature
            normalized_data[feature_name] = data_frame[feature_name]

    return normalized_data


def _get_scaler(scaler_path: str, feature_name: str):
    normalized_path = os.path.abspath(scaler_path)
    cache_key = (normalized_path, feature_name)

    if cache_key in _SCALER_CACHE:
        return _SCALER_CACHE[cache_key]

    scaler_file = os.path.join(normalized_path, f"{feature_name}_scaler.pkl")
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        _SCALER_CACHE[cache_key] = scaler
    else:
        _SCALER_CACHE[cache_key] = None
    return _SCALER_CACHE[cache_key]
