"""Tests for the validation module."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from ratemyhuman.model import EMOTION_LABELS, ValenceResult, map_label_to_valence
from ratemyhuman.validate import ValidationReport, ValidationRunner, run_validation


class TestComputeMetrics:
    """Tests the static compute_metrics method with known inputs."""

    def _make_report(self, y_true, y_pred):
        """Helper to build a report from label lists."""
        return ValidationRunner.compute_metrics(
            y_true, y_pred, total=len(y_true), skipped=0, misclassified=[],
        )

    def test_perfect_predictions(self):
        """Verifies perfect accuracy and F1 when all predictions match."""
        y = ["Negative"] * 5 + ["Neutral"] * 3 + ["Positive"] * 4
        report = self._make_report(y, y)
        assert report.accuracy == pytest.approx(1.0)
        assert report.f1_macro == pytest.approx(1.0)
        assert report.f1_weighted == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        """Verifies zero accuracy when every prediction is wrong."""
        y_true = ["Negative"] * 4 + ["Positive"] * 4
        y_pred = ["Positive"] * 4 + ["Negative"] * 4
        report = self._make_report(y_true, y_pred)
        assert report.accuracy == pytest.approx(0.0)

    def test_confusion_matrix_shape(self):
        """Verifies the confusion matrix is 3×3."""
        y = ["Negative", "Neutral", "Positive"]
        report = self._make_report(y, y)
        assert report.confusion_matrix.shape == (3, 3)

    def test_confusion_matrix_diagonal(self):
        """Verifies diagonal entries match correct predictions."""
        y_true = ["Negative"] * 3 + ["Neutral"] * 2 + ["Positive"] * 5
        report = self._make_report(y_true, y_true)
        assert report.confusion_matrix[0, 0] == 3
        assert report.confusion_matrix[1, 1] == 2
        assert report.confusion_matrix[2, 2] == 5

    def test_baselines(self):
        """Verifies random and majority baselines are computed correctly."""
        y_true = ["Negative"] * 10 + ["Neutral"] * 3 + ["Positive"] * 7
        report = self._make_report(y_true, y_true)
        assert report.baseline_random == pytest.approx(1 / 3)
        assert report.baseline_majority == pytest.approx(10 / 20)

    def test_skipped_count(self):
        """Verifies skipped image count is passed through."""
        report = ValidationRunner.compute_metrics(
            ["Negative", "Positive"], ["Negative", "Positive"],
            total=5, skipped=3, misclassified=[],
        )
        assert report.total_images == 5
        assert report.skipped_images == 3

    def test_per_class_metrics_keys(self):
        """Verifies per-class metrics contain all three valence classes."""
        y = ["Negative", "Neutral", "Positive"]
        report = self._make_report(y, y)
        assert set(report.per_class_metrics.keys()) == {"Negative", "Neutral", "Positive"}
        for cls in report.per_class_metrics:
            assert "precision" in report.per_class_metrics[cls]
            assert "recall" in report.per_class_metrics[cls]
            assert "f1" in report.per_class_metrics[cls]
            assert "support" in report.per_class_metrics[cls]

    def test_mcc_perfect(self):
        """Verifies MCC is 1.0 for perfect predictions."""
        y = ["Negative"] * 5 + ["Neutral"] * 5 + ["Positive"] * 5
        report = self._make_report(y, y)
        assert report.mcc == pytest.approx(1.0)

    def test_misclassified_passthrough(self):
        """Verifies misclassified samples are stored in the report."""
        sample = [{"path": "test.png", "true": "Negative", "pred": "Positive",
                    "confidence": 0.9, "emotion_scores": {}}]
        report = ValidationRunner.compute_metrics(
            ["Negative"], ["Positive"], total=1, skipped=0, misclassified=sample,
        )
        assert len(report.misclassified) == 1
        assert report.misclassified[0]["true"] == "Negative"


class TestValidationReportSummary:
    """Tests the report summary formatting."""

    def test_summary_contains_accuracy(self):
        """Verifies the summary includes the accuracy value."""
        report = ValidationRunner.compute_metrics(
            ["Negative", "Positive", "Positive"],
            ["Negative", "Positive", "Negative"],
            total=3, skipped=0, misclassified=[],
        )
        text = report.summary()
        assert "Accuracy" in text
        assert "F1 (macro)" in text
        assert "MCC" in text
        assert "Random baseline" in text
        assert "Majority baseline" in text
        assert "Confusion matrix" in text


# ---------------------------------------------------------------------------
# ValidationRunner tests — mocked detector (no GPU required)
# ---------------------------------------------------------------------------
class TestValidationRunner:
    """Tests the ValidationRunner class with mocked dependencies."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Creates a minimal labelled dataset for validation testing."""
        for emotion in ["happy", "angry", "neutral"]:
            d = tmp_path / emotion
            d.mkdir()
            for i in range(3):
                Image.new("L", (48, 48)).save(d / f"{i:05d}.png")
        return tmp_path

    def test_load_dataset(self, sample_dataset):
        """Verifies that load_dataset returns correct (path, valence) pairs."""
        runner = ValidationRunner(MagicMock(), sample_dataset)
        samples = runner.load_dataset()
        assert len(samples) == 9
        valences = {s[1] for s in samples}
        assert valences == {"Positive", "Negative", "Neutral"}

    def test_load_dataset_skips_unknown(self, tmp_path):
        """Verifies that unrecognised directories are skipped."""
        (tmp_path / "happy").mkdir()
        Image.new("L", (48, 48)).save(tmp_path / "happy" / "0.png")
        (tmp_path / "contempt").mkdir()
        Image.new("L", (48, 48)).save(tmp_path / "contempt" / "0.png")
        runner = ValidationRunner(MagicMock(), tmp_path)
        samples = runner.load_dataset()
        assert len(samples) == 1

    def test_run_validation_all_correct(self, sample_dataset):
        """Verifies perfect metrics when the detector is always correct."""
        mock_detector = MagicMock()
        def classify_side_effect(path):
            """Returns the correct valence for each image."""
            valence = map_label_to_valence(path.parent.name)
            r = MagicMock()
            r.label = valence
            r.confidence = 0.99
            r.emotion_scores = {}
            return r
        mock_detector.classify.side_effect = classify_side_effect
        runner = ValidationRunner(mock_detector, sample_dataset)
        report = runner.run_validation()
        assert report.accuracy == pytest.approx(1.0)
        assert report.total_images == 9
        assert report.skipped_images == 0

    def test_run_validation_with_skips(self, sample_dataset):
        """Verifies that ValueError exceptions are counted as skipped."""
        mock_detector = MagicMock()
        call_count = [0]
        def classify_effect(path):
            """Alternates between success and failure to simulate partial skips."""
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise ValueError("No face")
            r = MagicMock()
            r.label = "Positive"
            r.confidence = 0.9
            r.emotion_scores = {}
            return r
        mock_detector.classify.side_effect = classify_effect
        runner = ValidationRunner(mock_detector, sample_dataset)
        report = runner.run_validation()
        assert report.skipped_images > 0
        assert report.skipped_images < report.total_images
        assert report.total_images == 9


class TestPlotMethods:
    """Tests the ValidationRunner plot static methods."""

    def _make_report(self):
        """Creates a synthetic ValidationReport for plot testing."""
        return ValidationReport(
            accuracy=0.80, f1_macro=0.75, f1_weighted=0.78, mcc=0.65,
            confusion_matrix=np.array([[10, 1, 0], [2, 8, 1], [0, 1, 9]]),
            per_class_metrics={
                "Negative": {"precision": 0.83, "recall": 0.91, "f1": 0.87, "support": 11},
                "Neutral": {"precision": 0.80, "recall": 0.73, "f1": 0.76, "support": 11},
                "Positive": {"precision": 0.90, "recall": 0.90, "f1": 0.90, "support": 10},
            },
            total_images=32, skipped_images=0,
            misclassified=[],
        )

    def test_plot_confusion_matrix(self, tmp_path):
        """Verifies confusion matrix plot is saved."""
        report = self._make_report()
        path = ValidationRunner.plot_confusion_matrix(report, tmp_path)
        assert path.exists()
        assert path.name == "confusion_matrix.png"

    def test_plot_misclassified_with_samples(self, tmp_path):
        """Verifies misclassified grid is saved when samples exist."""
        report = self._make_report()
        img_path = tmp_path / "sample.png"
        Image.new("L", (48, 48)).save(img_path)
        report.misclassified = [
            {"path": str(img_path), "true": "Negative", "pred": "Neutral",
             "confidence": 0.6, "emotion_scores": {}},
        ]
        path = ValidationRunner.plot_misclassified(report, tmp_path)
        assert path is not None
        assert path.exists()

    def test_plot_misclassified_empty(self, tmp_path):
        """Verifies None is returned when no misclassified samples exist."""
        report = self._make_report()
        result = ValidationRunner.plot_misclassified(report, tmp_path)
        assert result is None


class TestRunValidationConvenience:
    """Tests the run_validation convenience function."""

    @patch("ratemyhuman.model.ValenceDetector")
    def test_runs_pipeline(self, MockDetector, tmp_path):
        """Verifies the convenience function runs the full pipeline."""
        for emotion in ["happy", "angry"]:
            d = tmp_path / "data" / "test" / emotion
            d.mkdir(parents=True)
            Image.new("L", (48, 48)).save(d / "0.png")
        mock_result = MagicMock()
        mock_result.label = "Positive"
        mock_result.confidence = 0.9
        mock_result.emotion_scores = {}
        MockDetector.return_value.classify.return_value = mock_result
        out = tmp_path / "docs"
        report = run_validation(data_dir=tmp_path / "data" / "test", output_dir=out)
        assert report.total_images == 2
