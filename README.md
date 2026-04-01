# ratemyhuman

[![License: MIT](https://img.shields.io/badge/License-MIT-16697A.svg)](LICENSE)

Facial valence detection from human face images.  
Classifies facial expressions as **Negative**, **Neutral**, or **Positive** using a pretrained Vision Transformer and MTCNN face detection.

## Architecture

The pipeline has three decoupled stages (concept note §4.3):

1. **Face detection** — MTCNN (via `facenet-pytorch`) locates and crops the face
2. **Emotion inference** — ViT (`trpakov/vit-face-expression`, 85.8M params) predicts 7-class emotion probabilities
3. **Valence mapping** — Probabilities are aggregated into 3 valence classes:
   - Negative = Angry + Disgust + Fear + Sad
   - Neutral = Neutral
   - Positive = Happy + Surprise

## Quickstart

```bash
# Install dependencies (requires NVIDIA GPU with CUDA 12.8)
uv sync

# Classify a single image
uv run ratemyhuman classify path2realface.png
uv run ratemyhuman classify path2realface.png --json

# Explore the FER2013 dataset (distribution plots, sample grids)
uv run ratemyhuman explore

# Validate on the FER2013 test set (metrics + confusion matrix)
uv run ratemyhuman validate

# Start web UI
uv run ratemyhuman demo --port 8888
```

## CLI reference

| Command | Description |
|---|---|
| `uv run ratemyhuman classify IMAGE` | Classify one image (supports `--json`, `--device`) |
| `uv run ratemyhuman explore` | Run dataset exploration (supports `--data-dir`, `--output-dir`) |
| `uv run ratemyhuman validate` | Run validation on labelled data (supports `--data-dir`, `--split`) |
| `uv run ratemyhuman demo` | Launch Gradio web UI (supports `--share`, `--port`) |
| `uv run ratemyhuman push` | DVC + git commit + push workflow (supports `-m`, `--dry-run`) |

Global: `uv run ratemyhuman -v <command>` enables debug logging.

## Validation results (FER2013 test set)

| Metric | Value |
|---|---|
| Accuracy | 82.86% |
| F1 (macro) | 0.803 |
| F1 (weighted) | 0.828 |
| MCC | 0.728 |
| Random baseline | 33.3% |
| Majority baseline | 44.2% |

2,999 of 3,589 images evaluated (590 skipped — MTCNN cannot detect faces in some 48×48 images).

## Project structure

```
src/ratemyhuman/
  model.py       # ValenceDetector, ValenceResult, valence mapping
  validate.py    # ValidationRunner, ValidationReport, metrics + plots
  explore.py     # Dataset exploration: distributions, sample grids, integrity
  app.py         # Gradio web UI: predict, build_app, launch
  cli.py         # Click CLI: classify, explore, validate, demo, push
tests/
  test_model.py      # Valence mapping + mocked pipeline tests
  test_validate.py   # Metrics, runner, plot, convenience function tests
  test_explore.py    # Count, integrity, plotting, run_exploration tests
  test_cli.py        # CLI helper functions + Click command tests
  test_app.py        # Gradio predict + build_app tests
docs/
  concept_note.md    # Full project concept note
  presentation.ipynb # Jupyter notebook presentation
  *.png              # Generated plots
data/                # FER2013 dataset (DVC-tracked, not in git)
```

## Data setup

The FER2013 dataset is managed with [DVC](https://dvc.org/) and stored on S3.

```bash
# Pull data from the configured DVC remote (requires AWS credentials)
dvc pull
```

Alternatively, download [FER-2013 from Kaggle](https://www.kaggle.com/datasets/pankaj4321/fer-2013-facial-expression-dataset) and extract into `data/` with subdirectories `train/`, `val/`, `test/` containing one folder per emotion class.  
See `.env.example` for the Kaggle credential placeholders.

## GPU requirements

- NVIDIA GPU with CUDA 12.8 (tested on RTX 5070 Ti, Blackwell sm_120)
- PyTorch nightly build from `https://download.pytorch.org/whl/nightly/cu128`
- `uv` manages the nightly index and dependency overrides via `pyproject.toml`

## Testing

```bash
# Unit tests (no GPU required) — 109 tests, ~90% coverage
uv run pytest tests/ -m "not slow"

# Integration tests (requires GPU + model weights) — 3 tests
uv run pytest tests/ -m "slow"

# With coverage report
uv run pytest tests/ -m "not slow" --cov=ratemyhuman --cov-report=term-missing
```
