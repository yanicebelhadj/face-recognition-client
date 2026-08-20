# face_core.py
"""
Cœur « reconnaissance faciale » : détection, encodage (embedding 128-D), matching.

Ce module est volontairement **sans état** et **sans I/O disque** : il ne connaît
ni les utilisateurs, ni les sessions, ni les fichiers photo. Il transforme des
pixels en vecteurs. C'est `sessions.py` qui décide à qui appartient quoi.

Les modèles dlib sont chargés paresseusement et une seule fois : sur une
instance gratuite (512 Mo de RAM) on ne charge que le strict nécessaire.
"""
from __future__ import annotations

import io
import os
import threading

import dlib
import numpy as np
from PIL import Image, ImageOps

from model_store import ensure_model

# Le prédicteur 5 points est celui utilisé par l'exemple officiel dlib pour la
# reconnaissance faciale : ~10 Mo au lieu de ~100 Mo, et plus rapide, pour une
# qualité d'embedding équivalente.
SHAPE_PREDICTOR = "shape_predictor_5_face_landmarks.dat"

EMBEDDING_DIM = 128

# Taille max de l'image envoyée au détecteur. Au-delà, dlib devient très lent
# sans gagner en précision sur des visages proches de la caméra.
MAX_DETECT_SIDE = int(os.getenv("MAX_DETECT_SIDE", "640"))

# Distance euclidienne au-delà de laquelle deux visages sont considérés
# différents. 0.6 est le seuil de référence de dlib.
DEFAULT_TOLERANCE = float(os.getenv("FACE_TOLERANCE", "0.6"))

_load_lock = threading.Lock()

# Les objets modèles dlib ne sont PAS réentrants : deux appels simultanés sur la
# même instance corrompent son état interne et font tomber le process sur un
# segfault (pas une exception — tout le serveur meurt, d'où les coupures
# intermittentes). On sérialise donc chaque appel.
#
# Ce n'est pas une perte : l'inférence dlib est mono-thread et sature déjà un
# cœur, et l'alternative (une instance par thread) coûterait ~32 Mo de RAM
# supplémentaires par thread.
_inference_lock = threading.RLock()

_detector = None
_shape_predictor = None
_encoder = None


# --------------------------------------------------------------------------- #
# Chargement des modèles
# --------------------------------------------------------------------------- #
def _load_models() -> None:
    global _detector, _shape_predictor, _encoder
    if _encoder is not None:
        return
    with _load_lock:
        if _encoder is not None:
            return
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(str(ensure_model(SHAPE_PREDICTOR)))
        encoder = dlib.face_recognition_model_v1(
            str(ensure_model("dlib_face_recognition_resnet_model_v1.dat"))
        )
        _detector, _shape_predictor, _encoder = detector, shape_predictor, encoder


def warmup() -> None:
    """Charge les modèles et fait tourner une inférence à blanc."""
    _load_models()
    detect(np.zeros((120, 120, 3), dtype=np.uint8))


def models_ready() -> bool:
    return _encoder is not None


# --------------------------------------------------------------------------- #
# Décodage d'image
# --------------------------------------------------------------------------- #
def decode_image(data: bytes, max_side: int = MAX_DETECT_SIDE) -> np.ndarray:
    """
    bytes (jpeg/png/…) -> ndarray RGB uint8 contigu, redimensionné pour que le
    plus grand côté ne dépasse pas `max_side`.

    `exif_transpose` évite les photos de téléphone détectées « couchées ».
    """
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    width, height = image.size
    longest = max(width, height)
    if max_side and longest > max_side:
        ratio = max_side / longest
        image = image.resize((max(1, round(width * ratio)), max(1, round(height * ratio))), Image.LANCZOS)

    return np.ascontiguousarray(np.array(image), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Détection / encodage
# --------------------------------------------------------------------------- #
def _resize(rgb: np.ndarray, factor: float) -> np.ndarray:
    """Agrandit/réduit une image RGB (via PIL — évite d'embarquer OpenCV)."""
    height, width = rgb.shape[:2]
    resized = Image.fromarray(rgb).resize(
        (max(1, round(width * factor)), max(1, round(height * factor))), Image.BICUBIC
    )
    return np.ascontiguousarray(np.array(resized), dtype=np.uint8)


def _rect_area(rect) -> int:
    return (rect.right() - rect.left()) * (rect.bottom() - rect.top())


def _to_box(rect, shape) -> list[int]:
    """dlib.rectangle -> [top, right, bottom, left] borné à l'image."""
    height, width = shape[:2]
    return [
        max(rect.top(), 0),
        min(rect.right(), width),
        min(rect.bottom(), height),
        max(rect.left(), 0),
    ]


def detect(rgb: np.ndarray, upsample: int = 0) -> list:
    _load_models()
    with _inference_lock:
        return list(_detector(rgb, upsample))


def encode(rgb: np.ndarray, rect, jitters: int = 0) -> np.ndarray:
    _load_models()
    with _inference_lock:
        shape = _shape_predictor(rgb, rect)
        descriptor = _encoder.compute_face_descriptor(rgb, shape, jitters)
    return np.asarray(descriptor, dtype=np.float32)


def detect_and_encode(rgb: np.ndarray, upsample: int = 0, jitters: int = 0):
    """
    Chemin « temps réel » : tous les visages de la frame.

    Retourne (boxes, embeddings) où boxes est une liste [top, right, bottom, left]
    exprimée dans le repère de `rgb`.
    """
    rects = detect(rgb, upsample)
    boxes, embeddings = [], []
    for rect in rects:
        boxes.append(_to_box(rect, rgb.shape))
        embeddings.append(encode(rgb, rect, jitters))
    return boxes, embeddings


def embed_largest_face(rgb: np.ndarray) -> tuple[np.ndarray | None, list[int] | None]:
    """
    Chemin « enrôlement » : on ne garde que le plus grand visage, et on insiste
    (upsample puis agrandissement ×2) car une photo de profil ratée à
    l'inscription rend tout le reste inutile.

    Retourne (embedding, box) ou (None, None) si aucun visage n'est trouvé.
    """
    rects = detect(rgb, upsample=1)

    if not rects:
        # Dernier recours : on agrandit l'image, utile pour les petits visages.
        upscaled = _resize(rgb, 2.0)
        upscaled_rects = detect(upscaled, upsample=1)
        if not upscaled_rects:
            return None, None
        best = max(upscaled_rects, key=_rect_area)
        # jitters=1 : une passe de ré-échantillonnage, un peu plus robuste que 0
        # pour une image de référence qu'on n'encode qu'une fois.
        embedding = encode(upscaled, best, jitters=1)
        box = _to_box(
            dlib.rectangle(best.left() // 2, best.top() // 2, best.right() // 2, best.bottom() // 2),
            rgb.shape,
        )
        return embedding, box

    best = max(rects, key=_rect_area)
    return encode(rgb, best, jitters=1), _to_box(best, rgb.shape)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def match(
    embeddings: list[np.ndarray],
    known_matrix: np.ndarray | None,
    known_names: list[str],
    tolerance: float = DEFAULT_TOLERANCE,
):
    """
    Associe chaque embedding au nom connu le plus proche.

    Retourne (names, distances) — "Unknown" et None quand rien ne correspond.
    """
    if known_matrix is None or known_matrix.size == 0 or not known_names:
        return ["Unknown"] * len(embeddings), [None] * len(embeddings)

    names: list[str] = []
    distances: list[float | None] = []
    for embedding in embeddings:
        deltas = np.linalg.norm(known_matrix - embedding, axis=1)
        index = int(np.argmin(deltas))
        best = float(deltas[index])
        if best <= tolerance:
            names.append(known_names[index])
            distances.append(round(best, 4))
        else:
            names.append("Unknown")
            distances.append(round(best, 4))
    return names, distances
