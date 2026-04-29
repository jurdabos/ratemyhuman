# ratemyhuman

[![License: MIT](https://img.shields.io/badge/License-MIT-16697A.svg)](LICENSE)

Facial valence detection from human face images.  
Classifies facial expressions as **Negative**, **Neutral**, or **Positive** using a pretrained Vision Transformer and MTCNN face detection.

## For the marketing department

This tool was built for the **marketing department** to gauge viewer reactions to advertisement content (concept note §1).  
Instead of granular emotions (anger, fear, joy, …), it returns a **threefold valence signal** — `Negative` / `Neutral` / `Positive` — which is the polarity score most useful for ad-testing analytics.

### What the valence labels mean in the ad-viewing context

| Label    | Underlying emotions             | What you usually see on the face                                | What it tells you about the ad                                                                                                                                                                                                                                                                              |
|----------|---------------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Positive | `happy`, `surprise`             | Smile, raised eyebrows, mouth slightly open, engaged eyes       | The viewer is reacting favourably. `happy` typically signals delight or amusement; `surprise` signals an unexpected hook — which is usually *desirable* in advertising, even when the underlying emotion is mild shock rather than joy, because surprise drives ad recall and re-watch behaviour.            |
| Neutral  | `neutral`                       | Relaxed face, no dominant expression                            | No clear affective response. This may mean (a) the ad isn't resonating, (b) the viewer is in observational mode at this moment, or (c) the captured frame sits between two stronger expressive moments. Neutral isn't automatically a failure — informational ads (e.g. tech specs) can legitimately score Neutral. |
| Negative | `angry`, `disgust`, `fear`, `sad` | Frown, brow furrow, lip curl, eye avert, downturned mouth     | The viewer is reacting unfavourably — *but the underlying emotion matters*. `disgust` or `anger` at a creative choice is a clear warning. `sad` may be the **intended** reaction to an emotional charity-style or storyline ad and is therefore not necessarily a problem.                                  |

The 7-class emotion breakdown (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) is exposed alongside every prediction so analysts can drill into the *why*. Two `Negative` readings of the same magnitude can mean very different things — *disgust* at a creative is a problem to fix; *sadness* can be the desired emotional response.

### How to interpret confidence scores

The **confidence** is the probability mass on the winning valence class. With three classes, the **random baseline is 33.3 %** — anything in that neighbourhood is barely better than a coin flip on a three-sided coin.

| Confidence | Interpretation                                                              | Recommended action                                                                                                                                                       |
|------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ≥ 80 %    | Strong reading; the runner-up class is far behind.                          | Treat as a reliable single-frame data point.                                                                                                                             |
| 60–80 %    | Moderate reading; the top class is most likely but a runner-up has mass.    | Inspect the 3-class valence breakdown — if two classes are within ~10 percentage points of each other, treat as **ambiguous**.                                          |
| 40–60 %    | Weak reading; the model is hesitant.                                        | Cross-reference with adjacent frames from the same session before drawing conclusions.                                                                                   |
| < 40 %     | Very weak reading; close to chance.                                         | Discard or re-check. The input is likely poorly lit, partially occluded, mid-blink, or transitional.                                                                     |

The most common borderline cases in practice are:

- **Neutral vs. Positive** — a faint smile is hard to distinguish from a relaxed face (often a ~50 / 45 / 5 valence split).
- **Neutral vs. Negative** — a slight frown or pursed lips (often ~45 / 50 / 5).
- **Positive vs. Negative** *(rare)* — wide-eyed surprise can occasionally be misread as fear; the 7-class emotion breakdown is the tiebreaker.

### When to trust the output

The model is most reliable when **all** of the following hold:

- The face is **frontal**, well-lit, and unobstructed.
- The image resolution is **at least ~96 × 96 pixels** (the model's native input is 224 × 224, so higher is better).
- The expression is **moderate** — naturally shaped, not exaggerated to the point of distortion.
- The viewer is naturally engaged with the screen (eyes roughly forward).
- You're **aggregating across multiple frames** from the same session rather than reading a single moment in isolation.

Under these conditions, the headline accuracy on the FER-2013 test split (82.86 %, see *Validation results* below) is a reasonable **lower bound** — production webcam frames at 720p+ typically perform better than the 48 × 48 grayscale thumbnails FER-2013 was scored on.

### When to distrust the output

Be cautious when **any** of the following apply:

- **MTCNN reports "No face detected"** — the demo shows a warning card; the CLI exits with an error. This usually indicates extreme angles, heavy occlusion (mask, hand, hair across the face), or insufficient resolution.
- **Confidence is below ~60 %** — see the interpretation table above.
- **The 7-class emotion breakdown is evenly spread** — e.g. `happy = 0.30, neutral = 0.28, surprise = 0.25` with the rest below 0.10. The model is genuinely uncertain about *which* emotion is present, not just which valence; aggregating into 3 classes hides this.
- **The frame catches a transient state** — mid-blink, mid-speech, the on-ramp of a laugh, the moment a smile is fading.
- **Lighting is harsh** (single side-light, deep shadows on one half of the face) **or very dim**.
- **The viewer is wearing glasses with strong reflections, a face mask, or heavy makeup** that obscures the eyes, brows, or mouth — the muscle groups the model relies on are partially hidden.
- **You're inferring from a single frame in isolation** — real consumer reactions are temporally smoothed; one moment rarely tells the whole story.
- **The viewer is from a demographic significantly under-represented in FER-2013** (the training set is skewed toward Western, adult, frontal-portrait imagery). Consider a small private validation set of representative viewers before trusting aggregate numbers in production.

### Workflow recommendations

The end-to-end loop for an ad-testing session:

1. **Capture** — collect still frames from a viewing session (webcam snapshot, uploaded photo, or frames extracted from video) while a subject watches the ad. A **frontal, 720p+ webcam at 1–4 frames per second** is the practical sweet spot.
2. **Classify** — run each frame through the demo (`uv run ratemyhuman demo`) for an interactive review, or the CLI (`uv run ratemyhuman classify <image>`) for batch / scripted processing.
3. **Aggregate** — tally `Positive` / `Neutral` / `Negative` counts and confidences across the session to produce a per-ad valence profile. **Discard** low-confidence (< 60 %) and no-face frames *before* aggregation; they add noise without adding signal.
4. **Compare** — contrast valence profiles across creative variants (A/B test, regional cuts, length variants) to identify which version resonates.

Beyond the loop:

- **Report with the emotion breakdown, not just the headline.** "60 % Positive, driven mostly by *surprise*" is more actionable for a creative team than "60 % Positive". Likewise, `Negative` driven by *sad* tells a different story than `Negative` driven by *disgust*.
- **Validate periodically with a known reference clip.** If you have a baseline ad with established team consensus on its valence, classify it at the start of each session as a drift check.
- **Respect privacy and consent.** Viewer footage must be obtained with explicit consent and stored per your organisation's data-protection policy. The pipeline performs **no identity recognition** — it reads only facial expressions — but the source images themselves remain personal data.

## Architecture

The pipeline has three decoupled stages (concept note §4.3):

1. **Face detection** — MTCNN (via `facenet-pytorch`) locates and crops the face
2. **Emotion inference** — ViT (`trpakov/vit-face-expression`, 85.8M params) predicts 7-class emotion probabilities
3. **Valence mapping** — Probabilities are aggregated into 3 valence classes:
   - Negative = Angry + Disgust + Fear + Sad
   - Neutral = Neutral
   - Positive = Happy + Surprise

## Installation

This section provisions a fresh machine end-to-end — from "OS + internet only" to "valence detection demo serving live predictions on http://127.0.0.1:8888". On a typical broadband connection it takes about **5–10 minutes** of wall-clock time, most of which is dependency download.

### Prerequisites

| Component   | Minimum                                                                | Recommended                       | Notes                                                                                              |
|-------------|------------------------------------------------------------------------|-----------------------------------|----------------------------------------------------------------------------------------------------|
| OS          | Windows 10/11 or Ubuntu 22.04+                                         | Either                            | macOS works on CPU only (no CUDA wheels for Apple Silicon).                                        |
| GPU         | Any NVIDIA GPU with a CUDA-capable driver                              | RTX 30/40/50-series               | A CPU fallback exists (`--device cpu`) but classification is ~5× slower.                          |
| GPU driver  | Supports CUDA 12.8 (Linux ≥ 550.x, Windows ≥ 551.x)                    | Latest NVIDIA stable              | Verified with `nvidia-smi`.                                                                        |
| Disk        | ~3 GB free                                                             | ~5 GB                             | PyTorch nightly (~2.5 GB) + ViT weights (~340 MB) + FER-2013 (~80 MB).                             |
| RAM         | 8 GB                                                                   | 16 GB+                            |                                                                                                    |
| Python      | 3.12                                                                   | 3.12.11                           | uv installs this for you if missing; system Python is not used.                                    |
| Internet    | Required during install                                                |                                   | Downloads PyTorch nightly, ViT weights, and (optionally) FER-2013.                                 |

### Step 1 — Install uv

[`uv`](https://docs.astral.sh/uv/) is the package manager for the project; it provisions Python interpreters, the `.venv/`, and pinned dependencies in one go.

**Windows (PowerShell):**

```powershell
winget install --id=astral-sh.uv -e
```

**Linux / macOS (bash):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verify:**

```bash
uv --version
```

Expected: `uv 0.6.x` or newer.

### Step 2 — Verify the NVIDIA driver

```bash
nvidia-smi
```

Expected output: a table showing the GPU model, driver version, and a `CUDA Version` field. For this project that field must read **12.8 or higher** (it indicates the *maximum runtime* the driver supports; the actual CUDA runtime ships inside the PyTorch wheel).

If `nvidia-smi` is not found:

- **Windows** — install the latest *NVIDIA Game Ready Driver* or *Studio Driver* from https://www.nvidia.com/drivers/ and reboot.
- **Linux** — install the proprietary driver via your distro (e.g. `sudo ubuntu-drivers autoinstall` on Ubuntu 22.04+) and reboot.
- **No NVIDIA GPU** — skip to *Troubleshooting → Running on CPU*.

### Step 3 — Clone the repository

```bash
git clone https://github.com/acidvuca/ratemyhuman.git
cd ratemyhuman
```

### Step 4 — Sync the project

```bash
uv sync
```

This single command:

1. Installs Python 3.12 into uv's interpreter cache (if missing).
2. Creates a project-local `.venv/`.
3. Resolves every dependency exactly as pinned in `uv.lock`, including PyTorch nightly with CUDA 12.8.
4. Downloads ~2.5 GB of wheels from PyPI and the PyTorch nightly index (`https://download.pytorch.org/whl/nightly/cu128`).

The sync may print `warning: prerelease versions selected for ...` — this is intentional. `pyproject.toml` enables `prerelease = "allow"` and `index-strategy = "unsafe-best-match"` under `[tool.uv]` so the nightly PyTorch builds (required for Blackwell sm_120 / RTX 50-series support) resolve correctly.

### Step 5 — Verify the install

Confirm the CLI is wired up:

```bash
uv run ratemyhuman --help
```

Expected: a Click help screen listing `classify`, `explore`, `validate`, `demo`, and `push`.

Confirm CUDA is visible to PyTorch:

```bash
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected (on a CUDA-equipped machine):

```text
CUDA available: True
Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

Run the unit tests as a final sanity check (~10 s, no GPU required):

```bash
uv run pytest tests/ -m "not slow"
```

Expected: `110 passed, 3 deselected, 1 warning`. The deselected tests are the GPU integration tests (run them separately with `-m "slow"`); the single warning is an upstream PyTorch nightly `FutureWarning` and is documented in `docs/concept_note.md`.

### Step 6 — Obtain the FER-2013 dataset (optional)

The FER-2013 dataset is needed only for `uv run ratemyhuman validate` and `uv run ratemyhuman explore`. The `classify` and `demo` commands work without it (they classify arbitrary uploaded faces, not a labelled split).

**Option A — DVC pull (project maintainers / authorised collaborators):**

```bash
dvc pull
```

Requires AWS credentials with read access to the project's configured S3 DVC remote. Pulls ~80 MB of PNGs into `data/{train,val,test}/` based on the `.dvc` pointer files already tracked in git.

**Option B — Manual Kaggle download (everyone else):**

1. Create a Kaggle account at https://www.kaggle.com/ and a new API token at *Account → API → Create New Token* (downloads `kaggle.json`).
2. Copy `.env.example` to `.env` and populate the placeholders:

   ```bash
   cp .env.example .env
   # then edit .env to set KAGGLE_USERNAME and KAGGLE_KEY
   ```

3. Download and extract the dataset:

   ```bash
   uv run kaggle datasets download -d pankaj4321/fer-2013-facial-expression-dataset -p data/ --unzip
   ```

4. Verify the layout:

   ```text
   data/
     train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
     val/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
     test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
   ```

Either path produces an identical on-disk layout that `explore` and `validate` can consume.

### Step 7 — First run

**Classify a single image (no dataset needed):**

```bash
uv run ratemyhuman classify path/to/face.png
uv run ratemyhuman classify path/to/face.png --json   # machine-readable output
```

The first invocation downloads ~340 MB of ViT weights (`trpakov/vit-face-expression`) into your local HuggingFace cache. Subsequent calls reuse the cache and run in well under a second on GPU.

**Launch the Gradio demo:**

```bash
uv run ratemyhuman demo --port 8888
```

Open http://127.0.0.1:8888. Drop in a face image (or grant webcam permission) and the UI returns a `Positive` / `Neutral` / `Negative` valence label, a confidence score, the 3-class valence breakdown, and the underlying 7-class emotion breakdown.

**Run validation on FER-2013** (requires Step 6):

```bash
uv run ratemyhuman validate
```

Batch-classifies all images in `data/test/`, computes accuracy / F1 / MCC, saves a confusion matrix and a misclassified-samples grid into `docs/`, and prints a summary to stdout.

**Generate the dataset exploration plots** (requires Step 6):

```bash
uv run ratemyhuman explore
```

Produces class/valence distribution plots and per-emotion sample grids under `docs/`.

## Troubleshooting

### `uv sync` fails with "no compatible distribution"

uv could not find a PyTorch nightly wheel for your platform. Confirm:

- Python 3.12 is the active interpreter (`uv python install 3.12`, then re-run `uv sync`).
- `pyproject.toml` retains `prerelease = "allow"` and `index-strategy = "unsafe-best-match"` under `[tool.uv]` (these flags are committed to the repo; do not remove them).
- Your platform is `win_amd64` or `linux_x86_64`. PyTorch nightly currently does not publish Apple Silicon CUDA builds.

### `torch.cuda.is_available()` returns `False` on a GPU machine

- Re-run `nvidia-smi` and confirm `CUDA Version: 12.8` (or higher) appears in the header.
- On Windows, ensure the driver came from NVIDIA (Game Ready / Studio) and not the generic Microsoft Basic Display driver.
- On Linux, ensure the `nvidia` kernel module is loaded (`lsmod | grep nvidia`) and that you are not inside a container without `--gpus all` (Docker) or `nvidia-runtime`.

### Running on CPU (no NVIDIA GPU)

`classify` and `validate` accept a `--device cpu` flag; `demo` auto-detects:

```bash
uv run ratemyhuman classify path/to/face.png --device cpu
uv run ratemyhuman demo --port 8888    # falls back to CPU automatically
```

Classification on CPU takes ~2–3 seconds per image instead of < 1 s on GPU; the demo remains responsive.

### MTCNN cannot detect a face

Some inputs fail face detection — typically very small, heavily occluded, or strongly profile-angled faces. The CLI exits with a clear error; the demo shows a styled "No face detected" warning card. Try a higher-resolution, frontal image. (For the FER-2013 test split, ~16 % of the 48×48 images fail this step; this is documented in the validation report.)

### Gradio "AttributeError: 'str' object has no attribute 'name'"

This was an upstream Gradio bug in `Font.__eq__` triggered when our font list collided in length with `gr.themes.Glass()._font` (a 5-string list). The fix is committed in `src/ratemyhuman/app.py` (canonical 4-entry `[Font, str, str, str]` shape, see the comment block around the `THEME` definition). If you see this error after pinning a different Gradio version, re-check that comment.

### First demo request takes 30+ seconds

The first `predict()` call lazily loads MTCNN + the ViT model into the GPU and downloads the ViT weights if not cached. The Gradio button shows a loading spinner during this period. Subsequent calls are sub-second.

### `dvc pull` returns "403 Forbidden" or "NoCredentialsError"

The DVC remote lives in a private S3 bucket. If you don't have AWS credentials for it, use *Option B* (Kaggle download) in Step 6 instead — the resulting on-disk layout is identical.

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

## Testing

```bash
# Unit tests (no GPU required) — 109 tests, ~90% coverage
uv run pytest tests/ -m "not slow"

# Integration tests (requires GPU + model weights) — 3 tests
uv run pytest tests/ -m "slow"

# With coverage report
uv run pytest tests/ -m "not slow" --cov=ratemyhuman --cov-report=term-missing
```

## Future work

The project is intentionally decoupled — face detection, emotion inference, valence aggregation, the CLI, and the Gradio UI are each independently swappable — so the extensions below can be picked up without rewriting the core pipeline. Exact priorities and timelines are subject to further discussion between the marketing, IT, and engineering teams.

### 1. Fine-tuning on company-specific ad data

The current emotion model (`trpakov/vit-face-expression`) is fine-tuned on FER-2013, which has noisy labels (~65 % inter-annotator agreement; see concept note §2.5) and is skewed toward Western, adult, frontal-portrait imagery. Collecting a labelled corpus of viewer reactions to *actual* company ads (with explicit consent and proper data-protection handling) and fine-tuning the ViT head on it would:

- improve in-distribution accuracy on the demographics that actually watch your ads,
- reduce systematic bias on under-represented groups,
- and let the valence mapping be re-tuned per business definition (e.g. weighting `surprise` differently for shock-style vs. comfort-style creative).

Integration point: only `MODEL_ID` and the weights loaded by `ValenceDetector.__init__` change; the rest of the pipeline (face detection, valence aggregation, validation) is identical. AffectNet is a strong intermediate option — it carries continuous valence-arousal annotations for ~420 k images and aligns directly with the project's valence-based framing.

### 2. Video stream support

Today the CLI and demo classify **one still frame per request**. A first-class video mode would directly serve the Capture → Classify → Aggregate → Compare workflow described above:

- new CLI subcommand `uv run ratemyhuman classify-video <file.mp4> --fps 4` that extracts frames, batches them through `predict_emotion()` (GPU-efficient), and emits a per-frame JSON timeline plus aggregate Positive/Neutral/Negative ratios;
- a streaming Gradio component (`gr.Video` + `gr.LinePlot`) in `app.py` showing a live valence curve over the duration of the ad;
- a `--smooth` flag that applies a small temporal moving average so single-frame anomalies (mid-blink, mid-laugh-onset) don't dominate the per-second readout.

Implementation: add OpenCV or `torchvision.io.read_video` for frame extraction, route batches through the existing `ValenceDetector.predict_emotion()` (it already accepts numpy arrays), and persist timelines as JSON for downstream analytics.

### 3. Multi-face batch processing

MTCNN is currently configured with `keep_all=False` and `select_largest=True` (`model.py:ValenceDetector.__init__`), so only the largest face in a frame is classified. For focus-group-style scenes with multiple viewers, switching to `keep_all=True` would yield N faces per frame; the pipeline would then return a `list[ValenceResult]` keyed by face bounding-box centroid (so the same viewer can be tracked across frames).

Integration point: a small change to `detect_face()` plus a new `classify_array_multi()` method that preserves per-face confidence. The existing `ValidationRunner` already aggregates lists of predictions, so most of the harness carries over unchanged.

### 4. Alternative face detectors (RetinaFace, InsightFace)

MTCNN — while well-tested and PyTorch-native (concept note §4.5) — struggles with small or strongly-angled faces; ~16 % of FER-2013's 48×48 test images fail detection today. Plug-compatible alternatives worth benchmarking:

- [**RetinaFace**](https://github.com/serengil/retinaface) — single-stage, accurate on small faces, exposes bounding-box + 5-point landmarks (same interface as MTCNN).
- [**InsightFace**](https://github.com/deepinsight/insightface) (`SCRFD` / `RetinaFace` variants) — ONNX-ready, often the best speed/accuracy trade-off on modern GPUs.
- [**MediaPipe Face Detection**](https://developers.google.com/mediapipe) — CPU-friendly fallback for environments without CUDA.

Integration point: only `ValenceDetector.face_detector` and `detect_face()` change. A short benchmark on the FER-2013 test split (skip rate, latency, peak VRAM) would let the team pick the right trade-off rather than guess.

### 5. Deploying as a REST API or containerised service

The current entry points are a Click CLI and a Gradio web UI — both run synchronously in a single Python process, which is fine for ad-hoc analyst use but not for integration into upstream systems. Two natural deployment targets:

- **REST API** — a thin FastAPI wrapper around `ValenceDetector.classify_array()` exposing `POST /classify` (multipart upload or base64 JSON) and `POST /classify-batch`. The detector is loaded once at startup and re-used across requests (the same lazy singleton pattern already used in `app.py:_get_detector()`). Add token-based auth, rate limiting, and structured request/response logging for production hygiene.
- **Containerised service** — a `Dockerfile` based on a CUDA-enabled image (e.g. `nvcr.io/nvidia/pytorch:24.09-py3`) plus a `compose.yml` for local orchestration. Deployment becomes "pull image, mount HF cache volume, run with `--gpus all`" on any host with the NVIDIA Container Toolkit. The DVC remote and the FER-2013 dataset are *not* bundled into the image — they're only needed for `validate` / `explore`, and the runtime image is leaner without them.

The REST + container pair is what moves the project from "research artefact running locally" to "service the ad-testing infrastructure can call". The exact integration shape — sync vs. async, JSON vs. multipart, on-prem vs. cloud, retention/consent policy for incoming frames — is the part that needs inter-departmental sign-off.
