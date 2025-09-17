# face_core.py
import cv2
import dlib
import numpy as np
from PIL import Image
from imutils import face_utils
from pathlib import Path
import os, ntpath

# ---------- Load pretrained models ----------
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point  = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
face_encoder            = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector           = dlib.get_frontal_face_detector()

def _transform(image, face_locations):
    coords = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord = max(rect[0],0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3],0)
        coords.append(coord)
    return coords

def encode_face(image_rgb_np: np.ndarray):
    """
    image_rgb_np: ndarray RGB (uint8)
    returns: enc_list, boxes, landmarks_list
    """
    img = np.ascontiguousarray(image_rgb_np, dtype=np.uint8)
    face_locations = face_detector(img, 1)
    enc_list, landmarks_list = [], []
    for face_location in face_locations:
        shape = pose_predictor_68_point(img, face_location)
        face_encoding = face_encoder.compute_face_descriptor(img, shape, 1)
        enc_list.append(np.array(face_encoding))
        landmarks_list.append(face_utils.shape_to_np(shape))
    boxes = _transform(img, face_locations)
    return enc_list, boxes, landmarks_list

def load_known_faces(directory: str):
    """
    Charge toutes les images .jpg/.png d'un dossier,
    renvoie (encodings: np.ndarray[N,128], names: List[str])
    """
    base = Path(directory)
    files = list(base.rglob("*.jpg")) + list(base.rglob("*.png"))
    if not files:
        raise ValueError(f"No faces detect in the directory: {base}")
    names, encodings = [], []
    for f in files:
        names.append(os.path.splitext(ntpath.basename(f))[0])
        img_rgb = np.array(Image.open(f).convert("RGB"))
        encs, _, _ = encode_face(img_rgb)
        if encs:
            encodings.append(encs[0])
    if len(encodings) == 0:
        return np.empty((0,128)), names
    return np.array(encodings), names

def recognize_on_frame(frame_bgr: np.ndarray, known_encodings: np.ndarray, known_names, tolerance: float = 0.6):
    """
    frame_bgr: image OpenCV (BGR)
    returns: (boxes, names_out, landmarks_list)
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    encs, boxes, landmarks_list = encode_face(rgb)

    if known_encodings is None or (hasattr(known_encodings, "size") and known_encodings.size == 0):
        return boxes, ["Unknown"] * len(encs), landmarks_list

    names_out = []
    for enc in encs:
        if enc is None or len(enc) == 0:
            names_out.append("Unknown"); continue
        dists = np.linalg.norm(known_encodings - enc, axis=1)
        idx = int(np.argmin(dists))
        names_out.append(known_names[idx] if dists[idx] <= tolerance else "Unknown")
    return boxes, names_out, landmarks_list
