# face_core.py  (version légère: OpenCV YuNet + SFace)
import cv2
import numpy as np
import PIL.Image
from pathlib import Path
import os, ntpath

MODELS_DIR = Path(__file__).parent / "pretrained_model"
YUNET   = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
SFACE   = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")

_detector = None
_recognizer = None

def _get_models():
    global _detector, _recognizer
    if _detector is None:
        # YuNet detector
        _detector = cv2.FaceDetectorYN.create(
            model=YUNET,
            config="",
            input_size=(320, 320),   # petite taille -> rapide / faible RAM
            score_threshold=0.85,
            nms_threshold=0.3,
            top_k=5000
        )
    if _recognizer is None:
        # SFace embedder
        _recognizer = cv2.FaceRecognizerSF.create(SFACE, "")
    return _detector, _recognizer

def _img_to_bgr(np_or_pil):
    if isinstance(np_or_pil, np.ndarray):
        arr = np_or_pil
        if arr.ndim == 2: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4: arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr
    else:
        return cv2.cvtColor(np.array(np_or_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def _detect(bgr):
    det, _ = _get_models()
    h, w = bgr.shape[:2]
    # l’API YuNet veut qu’on fixe l’input_size = taille image
    det.setInputSize((w, h))
    faces = det.detect(bgr)[1]  # shape: [N,15] (x,y,w,h,score, 5*2 landmarks)
    if faces is None: 
        return []
    # boîtes au format (top, right, bottom, left)
    out = []
    for f in faces:
        x, y, w, h = f[:4]
        t, l = int(max(0, y)), int(max(0, x))
        b, r = int(min(h+y, h if False else y+h)), int(min(w+x, w if False else x+w))
        out.append((t, r, b, l))
    return out

def _crop_align(bgr, box):
    t, r, b, l = box
    face = bgr[t:b, l:r]
    return face

def _feature(bgr, box):
    det, rec = _get_models()
    # SFace attend une face alignée; on peut utiliser alignCrop si landmarks, 
    # mais YuNet renvoie landmarks: on les ignore pour rester simple -> crop direct
    face = _crop_align(bgr, box)
    if face.size == 0: 
        return None
    # compute features (L2-normalized 128D)
    feat = rec.feature(face)
    return feat

def encode_face(image_np):
    """Compat: renvoie (encodings, boxes, landmarks)"""
    bgr = _img_to_bgr(image_np)
    boxes = _detect(bgr)
    encs = []
    for box in boxes:
        f = _feature(bgr, box)
        if f is not None:
            encs.append(f.astype(np.float32))
    landmarks = [[] for _ in boxes]  # on n’utilise pas ici
    return encs, boxes, landmarks

def load_known_faces(directory: str):
    base = Path(directory)
    files = list(base.rglob('*.jpg')) + list(base.rglob('*.png')) + list(base.rglob('*.jpeg'))
    if not files:
        raise ValueError(f'No faces detect in the directory: {base}')
    names, encodings = [], []
    for f in files:
        try:
            img = PIL.Image.open(f)
        except Exception:
            continue
        bgr = _img_to_bgr(img)
        encs, _, _ = encode_face(bgr)
        if encs:
            encodings.append(encs[0])
            names.append(os.path.splitext(ntpath.basename(f))[0])
    return np.array(encodings, dtype=np.float32), names

def recognize_on_frame(frame_bgr, known_encodings, known_names, threshold=0.363):
    """
    SFace renvoie des embeddings; on utilise la similarité cosinus.
    Un seuil ~0.363 (distance: 1 - cos_sim) marche bien; ajuste si besoin.
    """
    bgr = _img_to_bgr(frame_bgr)
    boxes = _detect(bgr)
    names_out = []
    for box in boxes:
        feat = _feature(bgr, box)
        if feat is None or known_encodings is None or len(known_encodings) == 0:
            names_out.append("Unknown"); continue
        # cos distance = 1 - cos_sim
        sims = (known_encodings @ feat) / (np.linalg.norm(known_encodings, axis=1) * np.linalg.norm(feat) + 1e-8)
        cos_dist = 1.0 - sims
        idx = int(np.argmin(cos_dist))
        names_out.append(known_names[idx] if cos_dist[idx] <= threshold else "Unknown")
    landmarks = [[] for _ in boxes]
    return boxes, names_out, landmarks
