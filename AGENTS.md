# Repository Guidelines

## Project Structure & Module Organization
- `app.py` hosts the Flask entrypoint, HTTP routes, and model-loading logic.
- `app_code/` contains reusable ML utilities: `feature_engineering.py` for derived features and `scale_features.py` for persisted scalers.
- `model/` stores serialized estimators (`.pkl`, `.keras`); keep filenames descriptive (e.g., `lightgbm_oversampled.pkl`).
- `templates/index.html` and `static/styles.css` drive the UI; update both when changing form fields or model selectors.

## Setup, Build & Run
```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py  # serves http://localhost:5000 in debug mode
```
- Use `FLASK_ENV=development python app.py` for auto-reload.
- Docker workflow: `docker build -t password-likelihood-predictor .` then `docker run -p 5000:5000 password-likelihood-predictor`.
- Regenerate models externally and copy artifacts into `model/` before committing.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and `snake_case` for functions, `PascalCase` for classes.
- Keep model-specific helpers pure and stateless; prefer module-level functions over global variables (besides the `loaded_models` cache).
- Place shared logic in `app_code/` and avoid duplicating feature or scaling steps inside view functions.
- Document non-obvious heuristics with inline comments or docstrings; mirror existing wording.

## Testing Guidelines
- Automated tests are not yet present; add `pytest` suites under `tests/` when modifying feature engineering or scaling logic.
- When touching Keras models, include a lightweight inference smoke test that loads the artifact from `model/` and scores a short sample list.
- Manually verify UI changes by posting through `/predict` with `curl` or inspecting the form in a browser (ensure model dropdown updates).

## Commit & Pull Request Guidelines
- Recent history favors concise, lowercase, imperative subjects (e.g., `add preview version`). Match that tone and keep under ~60 characters.
- Squash local commits before opening a PR; describe model or data updates clearly, including training sources and parameter changes.
- PRs should link related issues, summarize functional impacts, note new dependencies, and attach screenshots or JSON snippets for UI/API changes.
