import os
import pandas as pd
import joblib

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
        scaler_file = os.path.join(scaler_path, f"{feature_name}_scaler.pkl")
        
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            feature_data = data_frame[[feature_name]]  # Retain the feature name as a DataFrame
            normalized_feature = scaler.transform(feature_data)  # Transform the feature
            normalized_data[f'{feature_name}'] = normalized_feature  # Add the normalized feature
        else:
            # If there's no scaler, just copy the original feature
            normalized_data[feature_name] = data_frame[feature_name]

    print("Original data:\n", data_frame)
    print("Final normalized data:\n", normalized_data)
    print("\n\n")
    return normalized_data
