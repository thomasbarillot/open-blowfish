# GitHub Actions

| Workflow | Purpose |
|----------|---------|
| [`ci.yml`](ci.yml) | Python **3.10–3.12** on Ubuntu: `pip install -e .` + `pytest` (core deps only; no SHAP). |
| [`ci-explain.yml`](ci-explain.yml) | Same tests with **`pip install -e ".[explain]"`**; **`continue-on-error: true`** if `shap` has no wheel for the runner. |

Local parity:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
pytest tests/ -q
```

With SHAP / `FeedbackDecider`:

```bash
pip install -e ".[explain]"
```