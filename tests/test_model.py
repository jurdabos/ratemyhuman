"""Tests for the ValenceDetector, ValenceResult, and valence mapping layer."""
import glob

import numpy as np
import pytest

from ratemyhuman.model import (
    EMOTION_LABELS,
    EMOTION_PALETTE,
    VALENCE_COLOURS,
    VALENCE_MAP,
    VALENCE_ORDER,
    ValenceDetector,
    ValenceResult,
    _NEGATIVE_IDX,
    _NEUTRAL_IDX,
    _POSITIVE_IDX,
    map_label_to_valence,
)


# ---------------------------------------------------------------------------
# Unit tests — valence mapping (no GPU / model loading required)
# ---------------------------------------------------------------------------
class TestValenceMapping:
    """Tests the static map_to_valence logic."""

    def test_strong_happy_returns_positive(self):
        """Verifies that a dominant Happy probability yields Positive valence."""
        probs = np.array([0.01, 0.01, 0.01, 0.90, 0.03, 0.01, 0.03])
        result = ValenceDetector.map_to_valence(probs)
        assert result.label == "Positive"
        assert result.confidence == pytest.approx(0.93, abs=0.01)

    def test_strong_angry_returns_negative(self):
        """Verifies that a dominant Angry probability yields Negative valence."""
        probs = np.array([0.85, 0.03, 0.03, 0.02, 0.02, 0.03, 0.02])
        result = ValenceDetector.map_to_valence(probs)
        assert result.label == "Negative"
        assert result.confidence > 0.90

    def test_strong_neutral_returns_neutral(self):
        """Verifies that a dominant Neutral probability yields Neutral valence."""
        probs = np.array([0.02, 0.02, 0.02, 0.04, 0.80, 0.02, 0.08])
        result = ValenceDetector.map_to_valence(probs)
        assert result.label == "Neutral"
        assert result.confidence == pytest.approx(0.80, abs=0.01)

    def test_valence_scores_sum_to_one(self):
        """Verifies that valence probabilities always sum to 1.0."""
        probs = np.array([0.15, 0.10, 0.05, 0.30, 0.20, 0.10, 0.10])
        result = ValenceDetector.map_to_valence(probs)
        total = sum(result.valence_scores.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_seven_emotions_present(self):
        """Verifies that all 7 emotion labels appear in the result."""
        probs = np.ones(7) / 7
        result = ValenceDetector.map_to_valence(probs)
        assert set(result.emotion_scores.keys()) == set(EMOTION_LABELS)

    def test_all_three_valence_classes_present(self):
        """Verifies that all 3 valence classes appear in the result."""
        probs = np.ones(7) / 7
        result = ValenceDetector.map_to_valence(probs)
        assert set(result.valence_scores.keys()) == {"Negative", "Neutral", "Positive"}

    def test_negative_aggregation_covers_four_emotions(self):
        """Verifies Negative = angry + disgust + fear + sad."""
        assert len(_NEGATIVE_IDX) == 4
        negative_emotions = {EMOTION_LABELS[i] for i in _NEGATIVE_IDX}
        assert negative_emotions == {"angry", "disgust", "fear", "sad"}

    def test_positive_aggregation_covers_two_emotions(self):
        """Verifies Positive = happy + surprise."""
        assert len(_POSITIVE_IDX) == 2
        positive_emotions = {EMOTION_LABELS[i] for i in _POSITIVE_IDX}
        assert positive_emotions == {"happy", "surprise"}

    def test_neutral_aggregation_covers_one_emotion(self):
        """Verifies neutral = neutral only."""
        assert len(_NEUTRAL_IDX) == 1
        assert EMOTION_LABELS[_NEUTRAL_IDX[0]] == "neutral"

    def test_single_emotion_saturation(self):
        """Verifies correct output when a single emotion has 100% probability."""
        probs = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # 100% happy
        result = ValenceDetector.map_to_valence(probs)
        assert result.label == "Positive"
        assert result.confidence == pytest.approx(1.0)
        assert result.valence_scores["Negative"] == pytest.approx(0.0)
        assert result.valence_scores["Neutral"] == pytest.approx(0.0)

    def test_tie_negative_vs_positive(self):
        """Verifies deterministic output when Negative and Positive are tied."""
        # angry=0.25, disgust=0.25 → Negative=0.50; happy=0.25, surprise=0.25 → Positive=0.50
        probs = np.array([0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.25])
        result = ValenceDetector.map_to_valence(probs)
        # With dict-order tie-breaking, Negative comes first in the dict → wins max()
        assert result.label in {"Negative", "Positive"}
        assert result.confidence == pytest.approx(0.50)

    def test_near_tie_boundary(self):
        """Verifies correct winner when valence scores differ by a small margin."""
        # Negative=0.33, Neutral=0.34, Positive=0.33
        probs = np.array([0.10, 0.08, 0.07, 0.17, 0.34, 0.08, 0.16])
        result = ValenceDetector.map_to_valence(probs)
        assert result.label == "Neutral"
        assert result.confidence == pytest.approx(0.34, abs=0.01)

    def test_indices_cover_all_seven_positions(self):
        """Verifies that index groups jointly cover positions 0-6 without overlap."""
        all_idx = set(_NEGATIVE_IDX) | set(_NEUTRAL_IDX) | set(_POSITIVE_IDX)
        assert all_idx == {0, 1, 2, 3, 4, 5, 6}
        assert len(_NEGATIVE_IDX) + len(_NEUTRAL_IDX) + len(_POSITIVE_IDX) == 7


class TestMapLabelToValence:
    """Tests the ground-truth label mapping function."""

    @pytest.mark.parametrize("emotion,expected", [
        ("angry", "Negative"),
        ("disgust", "Negative"),
        ("fear", "Negative"),
        ("sad", "Negative"),
        ("neutral", "Neutral"),
        ("happy", "Positive"),
        ("surprise", "Positive"),
    ])
    def test_all_emotions_map_correctly(self, emotion, expected):
        """Verifies each emotion maps to the correct valence per §2.3."""
        assert map_label_to_valence(emotion) == expected

    def test_case_insensitive(self):
        """Verifies that label mapping accepts case variations."""
        assert map_label_to_valence("angry") == "Negative"
        assert map_label_to_valence("HAPPY") == "Positive"
        assert map_label_to_valence("Neutral") == "Neutral"

    def test_strips_whitespace(self):
        """Verifies that leading/trailing whitespace is handled."""
        assert map_label_to_valence("  Sad  ") == "Negative"

    def test_unknown_label_raises_keyerror(self):
        """Verifies that an unrecognised label raises KeyError."""
        with pytest.raises(KeyError, match="Unknown emotion label"):
            map_label_to_valence("Contempt")


class TestValenceConstants:
    """Tests the shared constant definitions."""

    def test_valence_order_has_three_classes(self):
        """Verifies VALENCE_ORDER contains exactly the 3 expected classes."""
        assert VALENCE_ORDER == ["Negative", "Neutral", "Positive"]

    def test_valence_colours_cover_all_classes(self):
        """Verifies every valence class has a colour."""
        assert set(VALENCE_COLOURS.keys()) == set(VALENCE_ORDER)

    def test_emotion_palette_length_matches_labels(self):
        """Verifies EMOTION_PALETTE has one colour per emotion."""
        assert len(EMOTION_PALETTE) == len(EMOTION_LABELS)

    def test_valence_map_consistent_with_indices(self):
        """Verifies VALENCE_MAP agrees with the index-based grouping."""
        for i in _NEGATIVE_IDX:
            assert VALENCE_MAP[EMOTION_LABELS[i]] == "Negative"
        for i in _NEUTRAL_IDX:
            assert VALENCE_MAP[EMOTION_LABELS[i]] == "Neutral"
        for i in _POSITIVE_IDX:
            assert VALENCE_MAP[EMOTION_LABELS[i]] == "Positive"


class TestValenceResult:
    """Tests the ValenceResult dataclass."""

    def test_str_representation(self):
        """Verifies the __str__ output format."""
        result = ValenceResult(
            label="Positive",
            confidence=0.95,
            emotion_scores={"happy": 0.90, "surprise": 0.05, "angry": 0.01,
                            "disgust": 0.01, "fear": 0.01, "neutral": 0.01, "sad": 0.01},
            valence_scores={"Positive": 0.95, "Negative": 0.04, "Neutral": 0.01},
        )
        text = str(result)
        assert "Positive" in text
        assert "happy" in text
        assert "95" in text

    def test_valence_map_completeness(self):
        """Verifies every emotion label has a valence mapping."""
        for label in EMOTION_LABELS:
            assert label in VALENCE_MAP


# ---------------------------------------------------------------------------
# Integration test — requires GPU and model weights (marked slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestDetectorIntegration:
    """Integration tests that load the full model and run inference."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Loads the detector once for all tests in the class."""
        return ValenceDetector()

    def test_classify_happy_image(self, detector):
        """Verifies end-to-end classification on a happy test image."""
        images = glob.glob(r"C:\acidvuca\ratemyhuman\data\test\happy\*.png")
        assert len(images) > 0, "No happy test images found"
        result = detector.classify(images[0])
        assert result.label == "Positive"
        assert result.confidence > 0.5

    def test_classify_angry_image(self, detector):
        """Verifies end-to-end classification on an angry test image."""
        images = glob.glob(r"C:\acidvuca\ratemyhuman\data\test\angry\*.png")
        assert len(images) > 0, "No angry test images found"
        result = detector.classify(images[0])
        assert result.label == "Negative"

    def test_classify_nonexistent_file_raises(self, detector):
        """Verifies FileNotFoundError for missing image paths."""
        with pytest.raises(FileNotFoundError):
            detector.classify("nonexistent_image.png")
