# AI Agent Contribution Guide

## Repository Snapshot
- Flask application (`app.py`) serves HTML templates from `templates/` and static assets from `static/`.
- Feature engineering helpers live in `app_code/`; they must stay deterministic and pure for reuse inside model pipelines.
- Serialized models and scalers are stored in `model/`; do not regenerate them inside CI or tests.
- Existing documentation sits in `README.md` (user focused) and `AGENTS.md` (human contributor guide); keep messaging consistent.

## Workspace Conventions
- Target Python 3.10+ with virtual environments: `python -m venv .venv && source .venv/bin/activate`.
- Install dependencies from `requirements.txt` only; avoid adding heavy training libraries unless essential.
- Preserve ASCII encoding and 4-space indentation across Python modules; prefer functional patterns over global state.
- When adding files, place automation-usable artifacts under `app_code/` or `tests/`; keep experiment notebooks out of the repo.

## Implementation Playbook
- Reuse `extract_features` and `scale_features_from_stored` rather than duplicating logic; extend them with guarded helper functions if new features are required.
- Any new model loaders should integrate with `load_model` in `app.py` using the same caching dictionary; document additional file extensions.
- UI changes require synchronized updates to `templates/index.html` and `static/styles.css`; confirm model dropdowns reflect new artifacts.
- For data migrations or scaler updates, script them externally, then commit the resulting `.pkl` files with clear naming.

## Testing Requirements
- Unit tests must run via `python -m unittest`; keep suites lightweight and deterministic.
- Place new tests inside `tests/` and mirror the module structure (e.g., `tests/test_feature_engineering.py`).
- Use temporary directories for filesystem-dependent checks; never mutate `model/` during tests.
- Ensure added tests cover edge cases like empty strings, high-entropy inputs, and missing scaler files.

## Delivery Checklist
- Run `python -m unittest` before handing off changes; capture and share failures with stack traces.
- Update documentation when behaviour or usage changes, especially `README.md` and this guide.
- Keep commits atomic with imperative subject lines (e.g., `add gru inference helper`).
- Provide reproduction steps for bugs, including inputs and the observed API or UI response.
- Confirm Docker builds (`docker build -t password-likelihood-predictor .`) after dependency updates or runtime changes.
