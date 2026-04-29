"""
Gradio web UI for the facial valence detection demo.

Provides an interactive showcase where users upload a face image
and receive the 3-class valence prediction with emotion breakdowns.
Styling follows the project palette (palettes.json) and the layout is
designed to be screenshot-friendly for the showcase.
"""
import logging

import gradio as gr
import numpy as np
from PIL import Image

from ratemyhuman.model import (
    MODEL_ID,
    VALENCE_COLOURS,
    ValenceDetector,
)

logger = logging.getLogger(__name__)
# Lazy-loaded singleton so the model is only loaded once per process
_detector: ValenceDetector | None = None

# Project palette (palettes.json) — exposed as constants for readability
_PALETTE_TEAL = "#16697A"
_PALETTE_RED = "#A24936"
_PALETTE_BLUE = "#7EBCE6"
_PALETTE_PEACH = "#E6BEAE"
_PALETTE_MUTED = "#5A6470"

# Initial summary shown before any classification has been requested
_PLACEHOLDER_SUMMARY = (
    "<div class='rmh-card rmh-card--idle'>"
    "<h2>Ready</h2>"
    "<p>Upload a face image (or use your webcam) to begin.</p>"
    "</div>"
)


def _format_status_card(message: str, *, kind: str) -> str:
    """
    Builds a styled HTML status card for info, warning, or error states.

    Used in place of the result summary when classification cannot run
    or returns no usable prediction.
    """
    icon = {"info": "ℹ", "warning": "⚠", "error": "✕"}.get(kind, "ℹ")
    return (
        f"<div class='rmh-card rmh-card--{kind}'>"
        f"<h2>{icon} {message}</h2>"
        f"</div>"
    )


def _format_result_card(label: str, confidence: float, top_emotion: str, top_emotion_p: float) -> str:
    """
    Builds the styled HTML summary card for a successful prediction.

    The accent colour is bound to the predicted valence class so the card
    visually echoes the result.
    """
    accent = VALENCE_COLOURS.get(label, _PALETTE_TEAL)
    return (
        f"<div class='rmh-card rmh-card--ok' style='--rmh-accent:{accent};'>"
        f"<h2>{label}</h2>"
        f"<p class='rmh-confidence'>{confidence:.1%} confidence</p>"
        f"<p class='rmh-top-emotion'>Top emotion: <b>{top_emotion}</b> ({top_emotion_p:.1%})</p>"
        f"</div>"
    )


def _get_detector() -> ValenceDetector:
    """
    Returns the shared ValenceDetector instance, initialising on first call.

    The detector is cached at module level so the heavy ViT and MTCNN models
    are only loaded once per process.
    """
    global _detector
    if _detector is None:
        logger.info("Loading ValenceDetector (first request)...")
        _detector = ValenceDetector()
    return _detector


def predict(image: Image.Image | np.ndarray | None) -> tuple[dict[str, float], dict[str, float], str]:
    """
    Runs the valence detection pipeline on an uploaded image.

    Returns:
        A tuple of (valence_scores, emotion_scores, summary_html). The score
        dicts are empty whenever the pipeline could not produce a prediction;
        the HTML summary always contains a styled status card explaining why.
    """
    if image is None:
        return {}, {}, _format_status_card("Please upload an image to begin.", kind="info")
    # Loading the model lazily so import-time stays cheap and tests stay fast
    try:
        detector = _get_detector()
    except Exception as exc:  # noqa: BLE001 - to surface any load-time failure to the user
        logger.exception("Failed to load ValenceDetector")
        return {}, {}, _format_status_card(f"Could not load the model: {exc}", kind="error")
    # Running the three-stage pipeline (MTCNN -> ViT -> valence aggregation)
    try:
        result = detector.classify_array(image)
    except ValueError as exc:
        # Expected when MTCNN finds no face in the supplied image
        logger.info(f"Classification skipped: {exc}")
        return {}, {}, _format_status_card(str(exc), kind="warning")
    except Exception as exc:  # noqa: BLE001 - to keep the demo alive on unexpected failures
        logger.exception("Unexpected error during classification")
        return {}, {}, _format_status_card(f"Unexpected error: {exc}", kind="error")
    # Formatting scores for gr.Label (highest-first confidences)
    valence_scores = {k: round(v, 4) for k, v in result.valence_scores.items()}
    emotion_scores = {k: round(v, 4) for k, v in result.emotion_scores.items()}
    top_emotion = max(result.emotion_scores, key=result.emotion_scores.get)
    summary = _format_result_card(
        result.label,
        result.confidence,
        top_emotion,
        result.emotion_scores[top_emotion],
    )
    return valence_scores, emotion_scores, summary


def _reset() -> tuple[None, dict, dict, str]:
    """
    Resets the input image and all output panels back to the idle state.
    """
    return None, {}, {}, _PLACEHOLDER_SUMMARY


# Gradio 6.x: theme and css are passed to launch(), not Blocks().
# The font list must follow the same [Font, str, str, str] shape as Gradio's stock
# themes (Default/Soft/Monochrome/etc., all length 4). The Glass theme uses an
# all-strings list of length 5, so any custom list of length 5 collides on length
# during launch()'s `is_custom_theme` equality check and triggers a buggy
# Font.__eq__ comparison against a string (AttributeError on `.name`).
THEME = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Inter"),
        "Lucida Sans Unicode",
        "system-ui",
        "sans-serif",
    ],
)
CSS = f"""
.gradio-container {{ max-width: 1080px !important; margin: 0 auto !important; }}
#rmh-header {{ text-align: center; padding: 4px 0 0 0; }}
#rmh-header h1 {{ color: {_PALETTE_TEAL}; margin: 0 0 4px 0; font-weight: 700; letter-spacing: -0.01em; }}
#rmh-header .rmh-subtitle {{ color: {_PALETTE_MUTED}; margin: 0; font-size: 1.0em; }}
.rmh-card {{
    border-radius: 16px;
    padding: 20px 18px;
    text-align: center;
    background: #FFFFFF;
    border: 1px solid {_PALETTE_PEACH};
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.rmh-card h2 {{ margin: 0; font-weight: 700; }}
.rmh-card--ok {{ border-color: var(--rmh-accent, {_PALETTE_TEAL}); border-width: 2px; }}
.rmh-card--ok h2 {{ color: var(--rmh-accent, {_PALETTE_TEAL}); font-size: 1.6em; }}
.rmh-card .rmh-confidence {{ font-size: 1.2em; margin: 6px 0 2px 0; color: {_PALETTE_MUTED}; }}
.rmh-card .rmh-top-emotion {{ color: {_PALETTE_MUTED}; margin: 0; font-size: 0.95em; }}
.rmh-card--idle {{ border-color: {_PALETTE_BLUE}; border-style: dashed; }}
.rmh-card--idle h2 {{ color: {_PALETTE_TEAL}; font-size: 1.2em; }}
.rmh-card--idle p {{ color: {_PALETTE_MUTED}; margin: 6px 0 0 0; }}
.rmh-card--info {{ border-color: {_PALETTE_BLUE}; }}
.rmh-card--info h2 {{ color: {_PALETTE_TEAL}; font-size: 1.1em; }}
.rmh-card--warning {{ border-color: {_PALETTE_PEACH}; background: #FFF8F2; }}
.rmh-card--warning h2 {{ color: {_PALETTE_RED}; font-size: 1.1em; }}
.rmh-card--error {{ border-color: {_PALETTE_RED}; background: #FFF1ED; }}
.rmh-card--error h2 {{ color: {_PALETTE_RED}; font-size: 1.1em; }}
#rmh-footer {{
    text-align: center;
    color: {_PALETTE_MUTED};
    font-size: 0.85em;
    padding: 12px 0 4px 0;
}}
#rmh-footer a {{ color: {_PALETTE_TEAL}; text-decoration: none; }}
#rmh-footer a:hover {{ text-decoration: underline; }}
"""


def build_app() -> gr.Blocks:
    """
    Constructs and returns the Gradio Blocks application.

    Wires the upload widget, action buttons, and result panels together
    without launching the server (kept separate to ease testing).
    """
    with gr.Blocks(title="RateMyHuman — Facial Valence Detection") as app:
        gr.HTML(
            "<div id='rmh-header'>"
            "<h1>RateMyHuman</h1>"
            "<p class='rmh-subtitle'>Facial valence detection · Negative · Neutral · Positive</p>"
            "</div>"
        )
        gr.Markdown(
            "Built for the **marketing department** to gauge viewer reactions to ad content.  "
            "Upload a face image (or use your webcam) — typically a still of a viewer reacting "
            "to an advertisement — and the demo returns its emotional **valence**:  "
            "`Positive` (the viewer reacted favourably), `Neutral` (no clear response), or "
            "`Negative` (the viewer reacted unfavourably). The 7-class emotion breakdown "
            "underneath shows *why* the valence came out the way it did.  "
            "_The first request loads the ViT weights and may take a few seconds._"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Face image",
                    sources=["upload", "webcam", "clipboard"],
                    height=380,
                    placeholder="Drop an image here, paste from clipboard, or use your webcam.",
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                    submit_btn = gr.Button("Detect valence", variant="primary", scale=2)
            with gr.Column(scale=1):
                summary_output = gr.HTML(
                    value=_PLACEHOLDER_SUMMARY,
                    label="Result",
                    show_label=False,
                )
                valence_output = gr.Label(
                    label="Valence (3 classes)",
                    num_top_classes=3,
                )
                emotion_output = gr.Label(
                    label="Emotion (7 classes)",
                    num_top_classes=7,
                )
        with gr.Accordion("Reading the result (for marketing analysts)", open=False):
            gr.Markdown(
                "**What the valence label means**  \n"
                "- **Positive** — the viewer reacted favourably to the ad (smile, surprise, delight).  \n"
                "- **Neutral** — no clear affective response; the ad neither delighted nor put them off.  \n"
                "- **Negative** — the viewer reacted unfavourably (frown, disgust, fear, sadness).  \n\n"
                "**Confidence** is the probability mass on the winning valence class. Treat results "
                "below ~60 % confidence, or two classes within ~10 pp of each other, as **ambiguous** "
                "and re-check with another frame from the same session.  \n\n"
                "**Why look at the 7 emotions?** Two `Negative` readings can mean very different things "
                "— fear and disgust at a creative are diagnostic problems, while sadness can be the "
                "intended reaction to a charity-style ad. The emotion breakdown is the layer that "
                "explains *why* the polarity came out the way it did.  \n\n"
                "**Suggested workflow**: capture frames during a viewing → classify each → aggregate "
                "counts per ad → compare valence profiles across creative variants (A/B test)."
            )
        with gr.Accordion("How it works (technical)", open=False):
            gr.Markdown(
                "1. **Face detection** — MTCNN (`facenet-pytorch`) locates and crops the largest face.\n"
                "2. **Emotion inference** — `trpakov/vit-face-expression` (ViT, ~85.8M params, "
                "fine-tuned on FER-2013) returns 7-class softmax probabilities.\n"
                "3. **Valence mapping** — probabilities are summed into 3 classes:\n"
                "   - **Negative** = angry + disgust + fear + sad\n"
                "   - **Neutral** = neutral\n"
                "   - **Positive** = happy + surprise\n\n"
                "On the FER-2013 test split the pipeline reaches **82.86%** accuracy and "
                "**0.803** macro-F1 (vs. 33.3% random / 44.2% majority baseline)."
            )
        gr.HTML(
            "<div id='rmh-footer'>"
            f"Model: <a href='https://huggingface.co/{MODEL_ID}' target='_blank' rel='noopener'>{MODEL_ID}</a>"
            " · Face detection: "
            "<a href='https://github.com/timesler/facenet-pytorch' target='_blank' rel='noopener'>facenet-pytorch (MTCNN)</a>"
            " · <a href='https://github.com/acidvuca/ratemyhuman' target='_blank' rel='noopener'>Source on GitHub</a>"
            "</div>"
        )
        # Wiring the events: explicit submit, auto-classify on image change, full reset on clear
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
        clear_btn.click(
            fn=_reset,
            inputs=[],
            outputs=[image_input, valence_output, emotion_output, summary_output],
        )
    return app


def launch(share: bool = False, server_port: int = 7860) -> None:
    """
    Builds and launches the Gradio demo server.

    Args:
        share: If True, requests a public Gradio share link.
        server_port: Local port to bind the demo to.
    """
    app = build_app()
    app.launch(share=share, server_port=server_port, theme=THEME, css=CSS)
