"""Tests for the exploration module."""
import logging
from pathlib import Path

import pytest
from PIL import Image

from ratemyhuman.explore import (
    check_integrity,
    count_images,
    plot_class_distribution,
    plot_sample_grid,
    plot_valence_distribution,
    print_summary,
    run_exploration,
    valence_distribution,
)
from ratemyhuman.model import EMOTION_LABELS


@pytest.fixture
def sample_data(tmp_path):
    """
    Creates a minimal FER2013-like directory structure with sample images.
    """
    for split in ["train", "test"]:
        for emotion in EMOTION_LABELS:
            emotion_dir = tmp_path / split / emotion
            emotion_dir.mkdir(parents=True)
            for i in range(2):
                Image.new("L", (48, 48), color=128).save(emotion_dir / f"{i:05d}.png")
    return tmp_path


class TestCountImages:
    """
    Tests the count_images function.
    """

    def test_counts_all_splits_and_emotions(self, sample_data):
        """
        Verifies correct counts for each split and emotion class.
        """
        counts = count_images(sample_data)
        assert "train" in counts
        assert "test" in counts
        for split in ["train", "test"]:
            for emotion in EMOTION_LABELS:
                assert counts[split][emotion] == 2

    def test_missing_split_skipped(self, tmp_path):
        """
        Verifies that a missing split directory is handled gracefully.
        """
        for emotion in EMOTION_LABELS:
            (tmp_path / "train" / emotion).mkdir(parents=True)
            Image.new("L", (48, 48)).save(tmp_path / "train" / emotion / "0.png")
        counts = count_images(tmp_path)
        assert "train" in counts
        assert "val" not in counts


class TestValenceDistribution:
    """
    Tests the valence_distribution function.
    """

    def test_aggregation(self, sample_data):
        """
        Verifies correct aggregation from 7 emotions to 3 valence classes.
        """
        counts = count_images(sample_data)
        valence = valence_distribution(counts)
        for split in ["train", "test"]:
            assert valence[split]["Negative"] == 8   # angry+disgust+fear+sad = 4*2
            assert valence[split]["Neutral"] == 2    # neutral = 1*2
            assert valence[split]["Positive"] == 4   # happy+surprise = 2*2


class TestCheckIntegrity:
    """
    Tests the check_integrity function.
    """

    def test_all_valid(self, sample_data):
        """
        Verifies no issues for a clean dataset.
        """
        issues = check_integrity(sample_data)
        assert len(issues["corrupt"]) == 0
        assert len(issues["wrong_size"]) == 0
        assert len(issues["wrong_mode"]) == 0

    def test_wrong_size_detected(self, tmp_path):
        """
        Verifies detection of images with incorrect dimensions.
        """
        d = tmp_path / "train" / "happy"
        d.mkdir(parents=True)
        Image.new("L", (64, 64)).save(d / "wrong.png")
        issues = check_integrity(tmp_path)
        assert len(issues["wrong_size"]) == 1

    def test_wrong_mode_detected(self, tmp_path):
        """
        Verifies detection of images with wrong colour mode.
        """
        d = tmp_path / "train" / "angry"
        d.mkdir(parents=True)
        Image.new("RGB", (48, 48)).save(d / "rgb.png")
        issues = check_integrity(tmp_path)
        assert len(issues["wrong_mode"]) == 1


class TestPlotFunctions:
    """
    Tests that plotting functions produce output files.
    """

    def test_plot_class_distribution(self, sample_data, tmp_path):
        """
        Verifies class distribution plot is saved.
        """
        counts = count_images(sample_data)
        out = tmp_path / "out"
        out.mkdir()
        path = plot_class_distribution(counts, out)
        assert path.exists()
        assert path.name == "class_distribution.png"

    def test_plot_valence_distribution(self, sample_data, tmp_path):
        """
        Verifies valence distribution plot is saved.
        """
        counts = count_images(sample_data)
        val = valence_distribution(counts)
        out = tmp_path / "out"
        out.mkdir()
        path = plot_valence_distribution(val, out)
        assert path.exists()
        assert path.name == "valence_distribution.png"

    def test_plot_sample_grid(self, sample_data, tmp_path):
        """
        Verifies sample grid plot is saved.
        """
        out = tmp_path / "out"
        out.mkdir()
        path = plot_sample_grid(sample_data, out, split="train", n_per_class=2)
        assert path.exists()
        assert "train" in path.name


class TestPrintSummary:
    """
    Tests the print_summary function.
    """

    def test_logs_split_info(self, sample_data, caplog):
        """
        Verifies that summary information is logged for each split.
        """
        counts = count_images(sample_data)
        val = valence_distribution(counts)
        with caplog.at_level(logging.INFO, logger="ratemyhuman.explore"):
            print_summary(counts, val)
        assert "TRAIN" in caplog.text
        assert "TEST" in caplog.text


class TestRunExploration:
    """
    Tests the run_exploration integration function.
    """

    def test_full_run(self, sample_data, tmp_path):
        """
        Verifies the full exploration pipeline produces all expected outputs.
        """
        out = tmp_path / "out"
        run_exploration(data_dir=sample_data, output_dir=out)
        assert (out / "class_distribution.png").exists()
        assert (out / "valence_distribution.png").exists()
        assert (out / "sample_grid_train.png").exists()
        assert (out / "sample_grid_test.png").exists()

    def test_missing_data_dir(self, tmp_path, caplog):
        """
        Verifies graceful handling of a nonexistent data directory.
        """
        with caplog.at_level(logging.ERROR, logger="ratemyhuman.explore"):
            run_exploration(data_dir=tmp_path / "nope", output_dir=tmp_path)
        assert "not found" in caplog.text.lower()
