from flask import Flask, jsonify, render_template, request

from app_code.prediction_service import PredictionService

app = Flask(__name__)
prediction_service = PredictionService()
MAX_WORDS = 5000


@app.route('/', methods=['GET'])
def index():
    model_files = prediction_service.list_available_models()
    return render_template('index.html', models=model_files)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name, words = _parse_prediction_payload(request)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    predictions = prediction_service.predict(words, model_name)

    response = [
        {'word': word, 'score': score}
        for word, score in zip(words, predictions)
    ]
    return jsonify({'predictions': response})


def _parse_prediction_payload(req):
    if req.is_json:
        payload = req.get_json(silent=True) or {}
        model_name = payload.get('model')
        words = payload.get('words', [])
    else:
        model_name = req.form.get('model')
        words = _split_text_block(req.form.get('words', ''))
        uploaded_file = req.files.get('words_file')
        if uploaded_file and uploaded_file.filename:
            file_content = uploaded_file.read()
            decoded = file_content.decode('utf-8', errors='ignore')
            words.extend(_split_text_block(decoded))

    if not model_name:
        raise ValueError('Model name is required.')

    normalized_words = _normalize_words(words)
    if not normalized_words:
        raise ValueError('No words provided.')

    if len(normalized_words) > MAX_WORDS:
        raise ValueError(f'Too many words submitted (max {MAX_WORDS}).')

    return model_name, normalized_words


def _normalize_words(words):
    normalized = []
    for entry in words:
        if isinstance(entry, str):
            normalized.extend(_split_text_block(entry))
        elif entry is None:
            continue
        else:
            normalized.append(str(entry).strip())
    return [word for word in normalized if word]


def _split_text_block(text_block):
    if not text_block:
        return []
    normalized = text_block.replace('\r\n', '\n')
    return [line.strip() for line in normalized.split('\n') if line.strip()]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
