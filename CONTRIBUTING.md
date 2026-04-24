# Contributing to PredMaint AI

Thank you for your interest in contributing.

## Getting Started

```bash
git clone https://github.com/SumedhPatil1507/industrial-predmaint-ai.git
cd industrial-predmaint-ai
pip install -r requirements.txt
cp .env.example .env
```

## Development Workflow

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Check syntax: `python -m py_compile backend/*.py frontend/*.py`
5. Commit with a clear message: `git commit -m "feat: add your feature"`
6. Push and open a Pull Request

## Commit Message Format

```
feat:     New feature
fix:      Bug fix
docs:     Documentation only
refactor: Code restructure without behavior change
test:     Adding tests
chore:    Build/config changes
```

## Areas to Contribute

- **New machine types** — add to `backend/iot_simulator.py` and `backend/health_score.py`
- **New ML models** — add to `backend/ml_engine.py`
- **New chart types** — add to `frontend/charts.py`
- **Bug fixes** — check open issues
- **Documentation** — improve docstrings and README

## Code Style

- Follow PEP 8
- Add docstrings to all functions
- Keep functions under 50 lines
- No unused imports

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## Questions

Open an issue or reach out via GitHub.
