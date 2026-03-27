"""
Facial valence detection using a pretrained ViT emotion model.

Implements the three-stage pipeline described in concept note §4.3:
MTCNN face detection → ViT 7-class emotion inference → 3-class valence mapping.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

logger = logging.getLogger(__name__)

# HuggingFace model identifier for the pretrained ViT fine-tuned on FER2013
MODEL_ID = "trpakov/vit-face-expression"

# Label order as defined in the model's config.json (id2label)
EMOTION_LABELS: list[str] = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
]

# Valence mapping per concept note §2.3
VALENCE_MAP: dict[str, str] = {
    "angry": "Negative",
    "disgust": "Negative",
    "fear": "Negative",
    "sad": "Negative",
    "neutral": "Neutral",
    "happy": "Positive",
    "surprise": "Positive",
}

# Canonical valence class order (used in validation, plots, etc.)
VALENCE_ORDER: list[str] = ["Negative", "Neutral", "Positive"]

# Colours per valence class — project palette (see palettes.json)
VALENCE_COLOURS: dict[str, str] = {
    "Negative": "#A24936",
    "Neutral": "#7EBCE6",
    "Positive": "#DBF4A7",
}

# Per-emotion colours aligned with EMOTION_LABELS order
EMOTION_PALETTE: list[str] = [
    "#A24936", "#A24936", "#A24936", "#DBF4A7",
    "#7EBCE6", "#A24936", "#DBF4A7",
]

# Indices grouped by valence class for probability aggregation
_NEGATIVE_IDX: list[int] = [0, 1, 2, 5]  # Angry, Disgust, Fear, Sad
_NEUTRAL_IDX: list[int] = [4]             # Neutral
_POSITIVE_IDX: list[int] = [3, 6]         # Happy, Surprise


def map_label_to_valence(emotion_label: str) -> str:
    """
    Maps a single emotion folder name to its valence class.

    Used for converting FER2013 ground-truth labels to 3-class valence.
    Accepts case-insensitive input (e.g. "angry", "Angry", "ANGRY").

    Raises:
        KeyError: If the emotion label is not recognised.
    """
    # Normalising to lowercase to match VALENCE_MAP keys
    normalised = emotion_label.strip().lower()
    if normalised not in VALENCE_MAP:
        raise KeyError(
            f"Unknown emotion label: {emotion_label!r}. "
            f"Expected one of: {', '.join(EMOTION_LABELS)}"
        )
    return VALENCE_MAP[normalised]


@dataclass
class ValenceResult:
    """Holds the output of a single valence classification."""
    label: str
    confidence: float
    emotion_scores: dict[str, float] = field(default_factory=dict)
    valence_scores: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        """Returns a human-readable summary."""
        top_emotion = max(self.emotion_scores, key=self.emotion_scores.get)
        return (
            f"Valence: {self.label} ({self.confidence:.1%}) | "
            f"Top emotion: {top_emotion} ({self.emotion_scores[top_emotion]:.1%})"
        )


class ValenceDetector:
    """
    Detects faces and classifies facial valence from images.

    Uses MTCNN for face detection and a pretrained ViT (trpakov/vit-face-expression)
    for 7-class emotion inference, then maps emotions to 3-class valence.
    """

    def __init__(self, device: str | None = None, model_id: str = MODEL_ID) -> None:
        """
        Initialises the detector, loading MTCNN and the ViT emotion model.

        Args:
            device: torch device string ("cuda", "cpu", etc.). Auto-detected if None.
            model_id: HuggingFace model identifier for the emotion model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info(f"Initialising ValenceDetector on {self.device}")
        # Initialising MTCNN for face detection
        self.face_detector = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            select_largest=True,
            post_process=False,  # to return 0-255 uint8 tensor
            device=self.device,
        )
        # Loading pretrained ViT emotion model and its processor
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.emotion_model = ViTForImageClassification.from_pretrained(model_id).to(self.device)
        self.emotion_model.eval()
        logger.info(f"Loaded emotion model: {model_id} ({sum(p.numel() for p in self.emotion_model.parameters()) / 1e6:.1f}M params)")

    def detect_face(self, image: Image.Image) -> np.ndarray | None:
        """
        Detects the largest face in an image and returns it as a cropped PIL Image.

        Returns None if no face is found.
        """
        # MTCNN expects RGB PIL Image; returning a cropped face tensor (0-255)
        face_tensor = self.face_detector(image)
        if face_tensor is None:
            return None
        # Converting CHW float tensor (0-255) back to HWC uint8 numpy for the ViT processor
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return face_np

    def predict_emotion(self, face_image: np.ndarray) -> np.ndarray:
        """
        Predicts 7-class emotion probabilities from a cropped face image.

        Args:
            face_image: HWC uint8 numpy array (RGB, typically 224×224).

        Returns:
            1-D numpy array of 7 softmax probabilities in EMOTION_LABELS order.
        """
        pil_face = Image.fromarray(face_image)
        inputs = self.processor(images=pil_face, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.emotion_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs

    @staticmethod
    def map_to_valence(emotion_probs: np.ndarray) -> ValenceResult:
        """
        Aggregates 7-class emotion probabilities into 3-class valence.

        Implements the aggregation from concept note §4.3:
            P(negative) = P(angry) + P(disgust) + P(fear) + P(sad)
            P(neutral)  = P(neutral)
            P(positive) = P(happy) + P(surprise)
        """
        emotion_scores = {label: float(p) for label, p in zip(EMOTION_LABELS, emotion_probs)}
        valence_scores = {
            "Negative": float(sum(emotion_probs[i] for i in _NEGATIVE_IDX)),
            "Neutral": float(sum(emotion_probs[i] for i in _NEUTRAL_IDX)),
            "Positive": float(sum(emotion_probs[i] for i in _POSITIVE_IDX)),
        }
        label = max(valence_scores, key=valence_scores.get)
        confidence = valence_scores[label]
        return ValenceResult(
            label=label,
            confidence=confidence,
            emotion_scores=emotion_scores,
            valence_scores=valence_scores,
        )

    def classify(self, image_path: str | Path) -> ValenceResult:
        """
        Runs the full pipeline on an image file: detect → infer → map.

        Args:
            image_path: Path to the input image.

        Returns:
            ValenceResult with label, confidence, and full score breakdowns.

        Raises:
            FileNotFoundError: If the image path does not exist.
            ValueError: If no face is detected in the image.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path).convert("RGB")
        face = self.detect_face(image)
        if face is None:
            raise ValueError(f"No face detected in {path.name}")
        emotion_probs = self.predict_emotion(face)
        return self.map_to_valence(emotion_probs)

    def classify_array(self, image: np.ndarray | Image.Image) -> ValenceResult:
        """
        Runs the full pipeline on an in-memory image (numpy array or PIL Image).

        Args:
            image: RGB image as numpy HWC array or PIL Image.

        Returns:
            ValenceResult, or raises ValueError if no face detected.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        face = self.detect_face(image)
        if face is None:
            raise ValueError("No face detected in the provided image")
        emotion_probs = self.predict_emotion(face)
        return self.map_to_valence(emotion_probs)
