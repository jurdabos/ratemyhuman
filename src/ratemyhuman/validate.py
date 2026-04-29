"""
Validates the valence detection pipeline on the FER2013 test set.

Implements the validation process described in concept note §4.4:
load dataset → map ground truth → batch inference → compute metrics → plot.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from ratemyhuman.model import (
    VALENCE_COLOURS,
    VALENCE_ORDER,
    ValenceDetector,
    map_label_to_valence,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """
    Holds the output of a validation run.

    Aggregates the headline metrics, the full confusion matrix, per-class
    breakdowns, baselines, and a sample of misclassified items so the
    report can be both summarised and visualised downstream.
    """
    accuracy: float
    f1_macro: float
    f1_weighted: float
    confusion_matrix: np.ndarray
    per_class_metrics: dict[str, dict[str, float]]
    mcc: float = 0.0
    total_images: int = 0
    skipped_images: int = 0
    baseline_random: float = 0.0
    baseline_majority: float = 0.0
    misclassified: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """
        Returns a multi-line text summary of the validation results.

        Mirrors the layout used in the CLI/log output: headline metrics,
        per-class precision/recall/F1, and the raw confusion matrix.
        """
        lines = [
            f"{'='*60}",
            f"Validation Report  ({self.total_images - self.skipped_images} evaluated, "
            f"{self.skipped_images} skipped / {self.total_images} total)",
            f"{'='*60}",
            f"  Accuracy:          {self.accuracy:.4f}",
            f"  F1 (macro):        {self.f1_macro:.4f}",
            f"  F1 (weighted):     {self.f1_weighted:.4f}",
            f"  MCC:               {self.mcc:.4f}",
            f"  Random baseline:   {self.baseline_random:.4f}",
            f"  Majority baseline: {self.baseline_majority:.4f}",
            "",
            "  Per-class metrics:",
        ]
        for cls in VALENCE_ORDER:
            m = self.per_class_metrics.get(cls, {})
            lines.append(
                f"    {cls:<10s}  P={m.get('precision', 0):.3f}  "
                f"R={m.get('recall', 0):.3f}  "
                f"F1={m.get('f1', 0):.3f}  "
                f"N={m.get('support', 0):.0f}"
            )
        lines.append(f"\n  Confusion matrix (rows=true, cols=pred):")
        lines.append(f"  {'':>10s}  " + "  ".join(f"{c:>8s}" for c in VALENCE_ORDER))
        for i, cls in enumerate(VALENCE_ORDER):
            row = "  ".join(f"{self.confusion_matrix[i, j]:>8d}" for j in range(len(VALENCE_ORDER)))
            lines.append(f"  {cls:>10s}  {row}")
        return "\n".join(lines)


class ValidationRunner:
    """
    Runs batch validation of the ValenceDetector on a labelled image dataset.

    Follows the validation process from concept note §4.2/§4.4:
    load dataset → map ground truth → batch inference → compare → metrics → plots.
    """

    def __init__(self, detector: ValenceDetector, dataset_path: str | Path) -> None:
        """
        Initialises the runner.

        Args:
            detector: An initialised ValenceDetector instance.
            dataset_path: Root of the dataset split (e.g. data/test) containing
                          emotion-named subdirectories.
        """
        self.detector = detector
        self.dataset_path = Path(dataset_path)

    def load_dataset(self) -> list[tuple[Path, str]]:
        """
        Scans the dataset directory and returns (image_path, valence_label) pairs.

        Expects subdirectories named by emotion class (angry, happy, etc.).
        Each emotion folder name is mapped to a valence label via map_label_to_valence().
        """
        samples: list[tuple[Path, str]] = []
        for emotion_dir in sorted(self.dataset_path.iterdir()):
            if not emotion_dir.is_dir():
                continue
            try:
                valence = map_label_to_valence(emotion_dir.name)
            except KeyError:
                logger.warning(f"Skipping unrecognised directory: {emotion_dir.name}")
                continue
            for img_path in sorted(emotion_dir.glob("*.png")):
                samples.append((img_path, valence))
        logger.info(f"Loaded {len(samples)} samples from {self.dataset_path}")
        return samples

    def run_validation(self, max_misclassified: int = 20) -> ValidationReport:
        """
        Runs inference on all images and computes the full metrics suite.

        Args:
            max_misclassified: Maximum number of misclassified examples to store.

        Returns:
            A ValidationReport with all metrics, confusion matrix, and samples.
        """
        samples = self.load_dataset()
        total = len(samples)
        y_true: list[str] = []
        y_pred: list[str] = []
        skipped = 0
        misclassified: list[dict[str, Any]] = []
        for i, (img_path, true_valence) in enumerate(samples):
            if (i + 1) % 500 == 0 or i == 0:
                logger.info(f"  Processing {i + 1}/{total}...")
            try:
                result = self.detector.classify(img_path)
                y_true.append(true_valence)
                y_pred.append(result.label)
                if result.label != true_valence and len(misclassified) < max_misclassified:
                    misclassified.append({
                        "path": str(img_path),
                        "true": true_valence,
                        "pred": result.label,
                        "confidence": result.confidence,
                        "emotion_scores": result.emotion_scores,
                    })
            except (ValueError, Exception) as exc:
                skipped += 1
                logger.debug(f"Skipped {img_path.name}: {exc}")
        return self.compute_metrics(y_true, y_pred, total, skipped, misclassified)

    @staticmethod
    def compute_metrics(
        y_true: list[str],
        y_pred: list[str],
        total: int,
        skipped: int,
        misclassified: list[dict[str, Any]],
    ) -> ValidationReport:
        """
        Computes all validation metrics from true/predicted label lists.

        Metrics follow concept note §4.4 and Banerjee et al.:
        accuracy, precision, recall, F1 (macro/weighted), MCC, confusion matrix,
        plus random and majority-class baselines.
        """
        labels = VALENCE_ORDER
        acc = accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1_wt = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0,
        )
        per_class: dict[str, dict[str, float]] = {}
        for i, cls in enumerate(labels):
            per_class[cls] = {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": float(sup[i]),
            }
        # Computing baselines per concept note §4.4
        n_classes = len(labels)
        baseline_random = 1.0 / n_classes
        from collections import Counter
        counts = Counter(y_true)
        majority_count = max(counts.values()) if counts else 0
        baseline_majority = majority_count / len(y_true) if y_true else 0.0
        return ValidationReport(
            accuracy=acc,
            f1_macro=f1_mac,
            f1_weighted=f1_wt,
            mcc=mcc,
            confusion_matrix=cm,
            per_class_metrics=per_class,
            total_images=total,
            skipped_images=skipped,
            baseline_random=baseline_random,
            baseline_majority=baseline_majority,
            misclassified=misclassified,
        )

    @staticmethod
    def plot_confusion_matrix(report: ValidationReport, output_dir: Path) -> Path:
        """
        Plots and saves a confusion matrix heatmap.

        Normalises rows by true-class support to display percentages and
        overlays raw counts in a smaller font for context.
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        cm = report.confusion_matrix
        # Normalising by row (true class) for percentage display
        # Guarding against empty true-class rows (e.g. tiny test fixtures)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros(cm.shape, dtype=float),
            where=row_sums != 0,
        ) * 100
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=VALENCE_ORDER,
            yticklabels=VALENCE_ORDER,
            cbar_kws={"label": "% of true class"},
            ax=ax,
        )
        # Overlaying raw counts in smaller font
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.72,
                    f"(n={cm[i, j]})",
                    ha="center", va="center", fontsize=8, color="gray",
                )
        ax.set_xlabel("Predicted valence", fontsize=12)
        ax.set_ylabel("True valence", fontsize=12)
        ax.set_title(
            f"Confusion Matrix — Accuracy: {report.accuracy:.1%}, "
            f"F1 (macro): {report.f1_macro:.3f}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        out = output_dir / "confusion_matrix.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confusion matrix plot to {out}")
        return out

    @staticmethod
    def plot_misclassified(report: ValidationReport, output_dir: Path, n: int = 12) -> Path | None:
        """
        Plots a grid of misclassified sample images with annotations.

        Returns ``None`` when the report has no misclassified samples,
        otherwise saves the grid as ``misclassified_samples.png``.
        """
        samples = report.misclassified[:n]
        if not samples:
            logger.info("No misclassified samples to plot.")
            return None
        cols = min(4, len(samples))
        rows = (len(samples) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
        fig.suptitle("Misclassified Samples", fontsize=14, fontweight="bold")
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]
        for idx in range(rows * cols):
            row, col = divmod(idx, cols)
            ax = axes[row][col]
            ax.axis("off")
            if idx < len(samples):
                s = samples[idx]
                img = Image.open(s["path"]).convert("RGB")
                ax.imshow(np.array(img))
                colour = "#A24936" if s["pred"] != s["true"] else "#16697A"
                ax.set_title(
                    f"T: {s['true']}\nP: {s['pred']} ({s['confidence']:.0%})",
                    fontsize=8, color=colour,
                )
        plt.tight_layout()
        out = output_dir / "misclassified_samples.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved misclassified grid to {out}")
        return out


def run_validation(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    split: str = "test",
) -> ValidationReport:
    """
    Convenience entry point to run the full validation pipeline.

    Initialises the detector, runs validation on the specified split,
    prints the report, and saves plots.
    """
    from ratemyhuman.model import ValenceDetector
    root = Path(__file__).resolve().parent
    for parent in [root, *root.parents]:
        if (parent / "pyproject.toml").exists():
            root = parent
            break
    if data_dir is None:
        data_dir = root / "data" / split
    if output_dir is None:
        output_dir = root / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running validation on {data_dir}")
    detector = ValenceDetector()
    runner = ValidationRunner(detector, data_dir)
    report = runner.run_validation()
    print(report.summary())
    ValidationRunner.plot_confusion_matrix(report, output_dir)
    ValidationRunner.plot_misclassified(report, output_dir)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_validation()
