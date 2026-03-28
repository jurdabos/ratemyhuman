"""Tests for the Gradio web UI (app.py)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from ratemyhuman.app import build_app, predict
from ratemyhuman.model import ValenceResult


# ---------------------------------------------------------------------------
# Unit tests — predict function with mocked detector (no GPU required)
# ---------------------------------------------------------------------------
class TestPredict:
    """Tests the predict() wrapper that feeds Gradio outputs."""

    @pytest.fixture(autouse=True)
    def _reset_detector(self):
        """Resets the module-level detector singleton between tests."""
        import ratemyhuman.app as app_mod
        app_mod._detector = None
        yield
        app_mod._detector = None

    def _make_result(self, label: str = "Positive", confidence: float = 0.93) -> ValenceResult:
        """Creates a dummy ValenceResult for testing."""
        return ValenceResult(
            label=label,
            confidence=confidence,
            emotion_scores={
                "angry": 0.01, "disgust": 0.01, "fear": 0.01,
                "happy": 0.90, "neutral": 0.03, "sad": 0.01, "surprise": 0.03,
            },
            valence_scores={"Negative": 0.04, "Neutral": 0.03, "Positive": 0.93},
        )

    def test_none_image_returns_empty(self):
        """Verifies that a None input returns empty dicts and an upload prompt."""
        valence, emotion, html = predict(None)
        assert valence == {}
        assert emotion == {}
        assert "upload" in html.lower()

    @patch("ratemyhuman.app._get_detector")
    def test_valid_image_returns_scores(self, mock_get):
        """Verifies that a valid image returns valence and emotion score dicts."""
        mock_detector = MagicMock()
        mock_detector.classify_array.return_value = self._make_result()
        mock_get.return_value = mock_detector
        # Creating a minimal RGB PIL image
        img = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
        valence, emotion, html = predict(img)
        assert "Positive" in valence
        assert valence["Positive"] == pytest.approx(0.93, abs=0.01)
        assert "happy" in emotion
        assert "Positive" in html

    @patch("ratemyhuman.app._get_detector")
    def test_no_face_returns_warning(self, mock_get):
        """Verifies graceful handling when no face is detected."""
        mock_detector = MagicMock()
        mock_detector.classify_array.side_effect = ValueError("No face detected")
        mock_get.return_value = mock_detector
        img = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
        valence, emotion, html = predict(img)
        assert valence == {}
        assert emotion == {}
        assert "No face detected" in html

    @patch("ratemyhuman.app._get_detector")
    def test_numpy_input_accepted(self, mock_get):
        """Verifies that numpy array input is accepted."""
        mock_detector = MagicMock()
        mock_detector.classify_array.return_value = self._make_result("Negative", 0.80)
        mock_get.return_value = mock_detector
        arr = np.zeros((48, 48, 3), dtype=np.uint8)
        valence, emotion, html = predict(arr)
        assert "Negative" in valence
        assert "Negative" in html

    @patch("ratemyhuman.app._get_detector")
    def test_all_seven_emotions_in_output(self, mock_get):
        """Verifies that all 7 emotion labels appear in the emotion scores output."""
        mock_detector = MagicMock()
        mock_detector.classify_array.return_value = self._make_result()
        mock_get.return_value = mock_detector
        img = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
        _, emotion, _ = predict(img)
        expected = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}
        assert set(emotion.keys()) == expected

    @patch("ratemyhuman.app._get_detector")
    def test_all_three_valence_classes_in_output(self, mock_get):
        """Verifies that all 3 valence classes appear in the valence scores output."""
        mock_detector = MagicMock()
        mock_detector.classify_array.return_value = self._make_result()
        mock_get.return_value = mock_detector
        img = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
        valence, _, _ = predict(img)
        assert set(valence.keys()) == {"Negative", "Neutral", "Positive"}


# ---------------------------------------------------------------------------
# Build test — verifies Gradio app construction (no GPU required)
# ---------------------------------------------------------------------------
class TestBuildApp:
    """Tests that the Gradio Blocks app can be constructed."""

    def test_build_app_returns_blocks(self):
        """Verifies that build_app() returns a Gradio Blocks instance."""
        import gradio as gr
        app = build_app()
        assert isinstance(app, gr.Blocks)
