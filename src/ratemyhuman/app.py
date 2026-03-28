"""
Gradio web UI for the facial valence detection demo.

Provides an interactive interface where users upload a face image
and receive the 3-class valence prediction with emotion breakdowns.
"""
import logging

import gradio as gr
import numpy as np
from PIL import Image

from ratemyhuman.model import (
    EMOTION_LABELS,
    VALENCE_COLOURS,
    VALENCE_ORDER,
    ValenceDetector,
)

logger = logging.getLogger(__name__)

# Lazy-loaded singleton so the model is only loaded once
_detector: ValenceDetector | None = None


def _get_detector() -> ValenceDetector:
    """Returns the shared ValenceDetector instance, initialising on first call."""
    global _detector
    if _detector is None:
        logger.info("Loading ValenceDetector (first request)…")
        _detector = ValenceDetector()
    return _detector


def predict(image: Image.Image | np.ndarray | None) -> tuple[dict[str, float], dict[str, float], str]:
    """
    Runs the valence detection pipeline on an uploaded image.

    Returns:
        A tuple of (valence_scores, emotion_scores, summary_html).
    """
    if image is None:
        return {}, {}, "<p style='color:#A24936;'>Please upload an image.</p>"
    detector = _get_detector()
    try:
        result = detector.classify_array(image)
    except ValueError as exc:
        return {}, {}, f"<p style='color:#A24936;'>⚠ {exc}</p>"
    # Formatting scores for gr.Label (highest-first confidences)
    valence_scores = {k: round(v, 4) for k, v in result.valence_scores.items()}
    emotion_scores = {k: round(v, 4) for k, v in result.emotion_scores.items()}
    # Building a small HTML summary with project palette colours
    colour = VALENCE_COLOURS.get(result.label, "#000000")
    top_emotion = max(result.emotion_scores, key=result.emotion_scores.get)
    summary = (
        f"<div style='text-align:center; padding:12px;'>"
        f"<h2 style='color:{colour}; margin:0;'>{result.label}</h2>"
        f"<p style='font-size:1.3em; margin:4px 0;'>{result.confidence:.1%} confidence</p>"
        f"<p style='color:#666;'>Top emotion: <b>{top_emotion}</b> "
        f"({result.emotion_scores[top_emotion]:.1%})</p>"
        f"</div>"
    )
    return valence_scores, emotion_scores, summary


# Gradio 6.x: theme and css are passed to launch(), not Blocks()
THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
CSS = ".gradio-container { max-width: 960px !important; } h1 { color: #16697A !important; }"


def build_app() -> gr.Blocks:
    """Constructs and returns the Gradio Blocks application."""
    with gr.Blocks(title="RateMyHuman — Facial Valence Detection") as app:
        gr.Markdown(
            "# RateMyHuman\n"
            "Upload a face image to detect its **emotional valence** "
            "(Positive / Neutral / Negative).\n\n"
            "The pipeline uses MTCNN for face detection and a ViT model "
            "fine-tuned on FER-2013 for 7-class emotion inference, "
            "then maps emotions to 3-class valence."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload a face image",
                    sources=["upload", "webcam"],
                )
                submit_btn = gr.Button("Classify", variant="primary")
            with gr.Column(scale=1):
                summary_output = gr.HTML(label="Result")
                valence_output = gr.Label(label="Valence scores", num_top_classes=3)
                emotion_output = gr.Label(label="Emotion scores", num_top_classes=7)
        submit_btn.click(
            fn=predict,
            inputs=[image_input],
            outputs=[valence_output, emotion_output, summary_output],
        )
        image_input.change(
            fn=predict,
            inputs=[image_input],
            outputs=[valence_output, emotion_output, summary_output],
        )
    return app


def launch(share: bool = False, server_port: int = 7860) -> None:
    """Builds and launches the Gradio demo server."""
    app = build_app()
    app.launch(share=share, server_port=server_port, theme=THEME, css=CSS)
