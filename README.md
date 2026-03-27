# ratemyhuman

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
ratemyhuman explore

# Validate on the FER2013 test set (metrics + confusion matrix)
ratemyhuman validate
```

## CLI reference

| Command | Description |
|---|---|
| `ratemyhuman classify IMAGE` | Classify one image (supports `--json`, `--device`) |
| `ratemyhuman explore` | Run dataset exploration (supports `--data-dir`, `--output-dir`) |
| `ratemyhuman validate` | Run validation on labelled data (supports `--data-dir`, `--split`) |
| `ratemyhuman push` | DVC + git commit + push workflow (supports `-m`, `--dry-run`) |

Global: `ratemyhuman -v <command>` enables debug logging.

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
  cli.py         # Click CLI: classify, explore, validate, push
tests/
  test_model.py      # 29 unit + 3 integration tests
  test_validate.py   # 10 unit tests
docs/
  concept_note.md    # Full project concept note
  notebook_steps.txt # Jupyter notebook runbook
  *.png              # Generated plots
data/                # FER2013 dataset (DVC-tracked, not in git)
```

## GPU requirements

- NVIDIA GPU with CUDA 12.8 (tested on RTX 5070 Ti, Blackwell sm_120)
- PyTorch nightly build from `https://download.pytorch.org/whl/nightly/cu128`
- `uv` manages the nightly index and dependency overrides via `pyproject.toml`

## Testing

```bash
# Unit tests (no GPU required)
uv run pytest tests/ -m "not slow"

# Integration tests (requires GPU + model weights)
uv run pytest tests/ -m "slow"
```
