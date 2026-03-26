"""Tests for the validation module."""
import numpy as np
import pytest

from ratemyhuman.validate import ValidationReport, ValidationRunner


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
