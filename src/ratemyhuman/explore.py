"""
Explores and validates the FER-2013 dataset.

Produces class/valence distribution stats, checks image integrity,
and generates visualisation plots saved to the project's docs/ folder.
"""
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from ratemyhuman.model import (
    EMOTION_LABELS as EMOTION_CLASSES,
    EMOTION_PALETTE,
    VALENCE_COLOURS,
    VALENCE_MAP,
)

logger = logging.getLogger(__name__)

SPLITS: list[str] = ["train", "val", "test"]


def _find_project_root() -> Path:
    """
    Locates the project root by walking up to find pyproject.toml.

    Falls back to the module's own directory if no marker is found,
    keeping the helper safe to call from arbitrary import locations.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def count_images(data_dir: Path) -> dict[str, dict[str, int]]:
    """
    Counts images per emotion class for each split.

    Returns a nested dict: {split: {emotion: count}}.
    """
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        counts[split] = {}
        for emotion in EMOTION_CLASSES:
            emotion_dir = split_dir / emotion
            if emotion_dir.is_dir():
                counts[split][emotion] = len(list(emotion_dir.glob("*.png")))
            else:
                counts[split][emotion] = 0
    return counts


def valence_distribution(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """
    Aggregates emotion counts into valence classes per split.

    Returns {split: {"Negative": n, "Neutral": n, "Positive": n}}.
    """
    valence_counts: dict[str, dict[str, int]] = {}
    for split, emotions in counts.items():
        agg: dict[str, int] = Counter()
        for emotion, count in emotions.items():
            agg[VALENCE_MAP[emotion]] += count
        valence_counts[split] = dict(agg)
    return valence_counts


def check_integrity(data_dir: Path) -> dict[str, list[str]]:
    """
    Verifies every image can be opened and has the expected properties.

    Returns a dict of issues: {"corrupt": [...], "wrong_size": [...], "wrong_mode": [...]}.
    """
    issues: dict[str, list[str]] = {"corrupt": [], "wrong_size": [], "wrong_mode": []}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for img_path in split_dir.rglob("*.png"):
            rel = str(img_path.relative_to(data_dir))
            try:
                with Image.open(img_path) as img:
                    if img.size != (48, 48):
                        issues["wrong_size"].append(f"{rel}: {img.size}")
                    if img.mode != "L":
                        issues["wrong_mode"].append(f"{rel}: {img.mode}")
            except Exception as exc:
                issues["corrupt"].append(f"{rel}: {exc}")
    return issues


def plot_class_distribution(counts: dict[str, dict[str, int]], output_dir: Path) -> Path:
    """
    Plots per-split emotion class distribution as a grouped bar chart.

    Saves the figure as ``class_distribution.png`` under ``output_dir``
    and returns the resulting path.
    """
    # Using constrained layout to fit suptitle + rotated x-tick labels without tight_layout warnings;
    # passing layout at creation time avoids the "axes sizes collapsed to zero" warning that
    # set_layout_engine triggers when applied after the axes geometry has already been computed.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, layout="constrained")
    fig.suptitle("FER-2013 — Emotion Class Distribution", fontsize=14, fontweight="bold")
    for ax, split in zip(axes, SPLITS):
        if split not in counts:
            continue
        emotions = EMOTION_CLASSES
        values = [counts[split].get(e, 0) for e in emotions]
        bars = ax.bar(emotions, values, color=EMOTION_PALETTE, edgecolor="white", linewidth=0.5)
        ax.set_title(split.capitalize(), fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        # Adding count labels on bars (display-space padding so the offset stays sane regardless of bar magnitude)
        ax.bar_label(bars, labels=[str(v) for v in values], padding=3, fontsize=8)
    axes[0].set_ylabel("Image count")
    out = output_dir / "class_distribution.png"
    # Constrained layout already produces a tight figure; bbox_inches="tight" would re-run it and warn
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved class distribution plot to {out}")
    return out


def plot_valence_distribution(valence_counts: dict[str, dict[str, int]], output_dir: Path) -> Path:
    """
    Plots per-split valence distribution as a grouped bar chart.

    Saves the figure as ``valence_distribution.png`` under ``output_dir``
    and returns the resulting path.
    """
    # Using constrained layout to fit suptitle and per-axis labels without tight_layout warnings;
    # passing layout at creation time keeps the engine consistent with the initial axes geometry.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True, layout="constrained")
    fig.suptitle("FER-2013 — Valence Distribution (3-class)", fontsize=14, fontweight="bold")
    valence_order = ["Negative", "Neutral", "Positive"]
    for ax, split in zip(axes, SPLITS):
        if split not in valence_counts:
            continue
        values = [valence_counts[split].get(v, 0) for v in valence_order]
        colours = [VALENCE_COLOURS[v] for v in valence_order]
        bars = ax.bar(valence_order, values, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_title(split.capitalize(), fontsize=12)
        # Adding count labels on bars (display-space padding so the offset stays sane regardless of bar magnitude)
        ax.bar_label(bars, labels=[str(v) for v in values], padding=3, fontsize=9)
    axes[0].set_ylabel("Image count")
    out = output_dir / "valence_distribution.png"
    # Constrained layout already produces a tight figure; bbox_inches="tight" would re-run it and warn
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved valence distribution plot to {out}")
    return out


def plot_sample_grid(data_dir: Path, output_dir: Path, split: str = "train", n_per_class: int = 5) -> Path:
    """
    Plots a grid of sample images (n per emotion class) from a given split.

    Useful as a sanity check that the on-disk layout matches the assumed
    ``data/<split>/<emotion>/*.png`` convention.
    """
    fig, axes = plt.subplots(len(EMOTION_CLASSES), n_per_class,
                             figsize=(n_per_class * 1.5, len(EMOTION_CLASSES) * 1.8))
    fig.suptitle(f"FER-2013 — Sample Images ({split})", fontsize=14, fontweight="bold")
    for row, emotion in enumerate(EMOTION_CLASSES):
        emotion_dir = data_dir / split / emotion
        images = sorted(emotion_dir.glob("*.png"))[:n_per_class]
        for col in range(n_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(images):
                img = Image.open(images[col])
                ax.imshow(np.array(img), cmap="gray", vmin=0, vmax=255)
            if col == 0:
                ax.set_title(emotion, fontsize=9, fontweight="bold", loc="left")
    plt.tight_layout()
    out = output_dir / f"sample_grid_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved sample grid to {out}")
    return out


def print_summary(counts: dict[str, dict[str, int]], valence_counts: dict[str, dict[str, int]]) -> None:
    """
    Prints a text summary of dataset statistics to the console.

    Logs per-split totals, per-emotion counts, and the aggregated valence
    breakdown via the module logger at ``INFO`` level.
    """
    for split in SPLITS:
        if split not in counts:
            continue
        total = sum(counts[split].values())
        logger.info(f"\n{'='*50}")
        logger.info(f"Split: {split.upper()} ({total:,} images)")
        logger.info(f"{'='*50}")
        logger.info("  Emotion breakdown:")
        for emotion in EMOTION_CLASSES:
            n = counts[split].get(emotion, 0)
            pct = n / total * 100 if total else 0
            logger.info(f"    {emotion:<10s} {n:>5,}  ({pct:5.1f}%)")
        logger.info("  Valence breakdown:")
        for valence in ["Negative", "Neutral", "Positive"]:
            n = valence_counts[split].get(valence, 0)
            pct = n / total * 100 if total else 0
            logger.info(f"    {valence:<10s} {n:>5,}  ({pct:5.1f}%)")


def run_exploration(data_dir: Path | None = None, output_dir: Path | None = None) -> None:
    """
    Runs the full data exploration pipeline.

    Produces distribution plots, sample grids, and an integrity report.
    """
    root = _find_project_root()
    if data_dir is None:
        data_dir = root / "data"
    if output_dir is None:
        output_dir = root / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        return
    # Counting images per class
    logger.info("Counting images per class...")
    counts = count_images(data_dir)
    valence_counts = valence_distribution(counts)
    print_summary(counts, valence_counts)
    # Checking image integrity
    logger.info("\nChecking image integrity...")
    issues = check_integrity(data_dir)
    total_issues = sum(len(v) for v in issues.values())
    if total_issues == 0:
        logger.info("All images passed integrity checks.")
    else:
        for category, items in issues.items():
            if items:
                logger.warning(f"  {category}: {len(items)} issue(s)")
                for item in items[:5]:
                    logger.warning(f"    {item}")
                if len(items) > 5:
                    logger.warning(f"    ... and {len(items) - 5} more")
    # Generating plots
    logger.info("\nGenerating plots...")
    plot_class_distribution(counts, output_dir)
    plot_valence_distribution(valence_counts, output_dir)
    plot_sample_grid(data_dir, output_dir, split="train")
    plot_sample_grid(data_dir, output_dir, split="test")
    logger.info("\nExploration complete.")


if __name__ == "__main__":
    run_exploration()
