# Password-Likelihood-Predictor

## Overview
Password-Likelihood-Predictor is a Flask-based web application that creates a score on strings to determine how likely they are to be a password.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Password-Likelihood-Predictor.git
    cd Password-Likelihood-Predictor
    ```

2. Build the Docker image:
    ```sh
    docker build -t password-likelihood-predictor .
    ```

3. Run the Docker container:
    ```sh
    docker run -p 5000:5000 password-likelihood-predictor
    ```

## Usage

1. Open your web browser and navigate to `http://localhost:5000`.
2. Enter a string and select a model to get a score.

## Models

The application uses several machine learning models stored in the [`model/`] directory:
- AdaBoost
- HistGradientBoosting (with and without oversampling)
- LightGBM (with and without oversampling)
- Logistic Regression (with and without oversampling)
- Random Forest (with and without oversampling)

## Feature Engineering

The feature engineering is handled in the [`app_code/feature_engineering.py`] file, which includes functions like:
- `average_unicode_value`
- `process_word_column`
- `ratio_unique_chars_to_length`
- `special_char_count`
- `unique_char_count`
- `uppercase_count`

## Scaling Features

The feature scaling is managed in the [`app_code/scale_features.py] file.

## License

This project is licensed under the Apache License, Version 2.0. See the [`LICENSE`] file for more details.