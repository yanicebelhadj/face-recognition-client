# model_store.py
"""
Résolution des modèles pré-entraînés dlib.

Les fichiers .dat pèsent de 10 à 100 Mo : ils ne sont pas versionnés
(cf. .gitignore). Ce module les localise, dans cet ordre :

1. un fichier déposé à la main dans PRETRAINED_DIR (surcharge locale) ;
2. le paquet PyPI `face-recognition-models`, qui embarque exactement les mêmes
   modèles — c'est la source par défaut, car un `pip install` épinglé est bien
   plus fiable qu'un téléchargement HTTP pendant le build (c'était une des
   causes de builds qui échouent une fois sur deux) ;
3. en dernier recours, téléchargement depuis dlib.net ou son miroir GitHub.

`python model_store.py` vérifie au build que tout est résoluble, pour échouer
tôt plutôt qu'au premier appel utilisateur.
"""
from __future__ import annotations

import bz2
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

PRETRAINED_DIR = Path(os.getenv("PRETRAINED_DIR", "pretrained_model"))

# Modèles nécessaires au service. Le prédicteur 68 points n'y figure pas : il
# n'est utile qu'aux landmarks détaillés et coûterait ~100 Mo de RAM.
REQUIRED = [
    "shape_predictor_5_face_landmarks.dat",
    "dlib_face_recognition_resnet_model_v1.dat",
]

# Nom de fichier -> accesseur du paquet `face_recognition_models`.
_PACKAGE_ACCESSORS = {
    "shape_predictor_5_face_landmarks.dat": "pose_predictor_five_point_model_location",
    "shape_predictor_68_face_landmarks.dat": "pose_predictor_model_location",
    "dlib_face_recognition_resnet_model_v1.dat": "face_recognition_model_location",
}

# Nom de fichier -> URLs candidates (essayées dans l'ordre).
_DOWNLOAD_SOURCES = {
    name: (
        f"http://dlib.net/files/{name}.bz2",
        f"https://github.com/davisking/dlib-models/raw/master/{name}.bz2",
    )
    for name in _PACKAGE_ACCESSORS
}


def _local_copy(filename: str) -> Path | None:
    candidate = PRETRAINED_DIR / filename
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    return None


def _from_package(filename: str) -> Path | None:
    accessor = _PACKAGE_ACCESSORS.get(filename)
    if not accessor:
        return None
    try:
        import face_recognition_models  # noqa: PLC0415 — dépendance optionnelle
    except ImportError:
        return None
    path = Path(getattr(face_recognition_models, accessor)())
    return path if path.exists() else None


def _download_bz2(url: str, dest: Path) -> None:
    """Télécharge et décompresse en flux, sans jamais tout charger en RAM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 — URLs constantes
            decompressor = bz2.BZ2Decompressor()
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                while chunk := response.read(1 << 20):
                    tmp.write(decompressor.decompress(chunk))
        shutil.move(str(tmp_path), str(dest))
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _from_network(filename: str) -> Path:
    dest = PRETRAINED_DIR / filename
    last_error: Exception | None = None
    for url in _DOWNLOAD_SOURCES.get(filename, ()):
        try:
            print(f"[models] téléchargement de {filename} depuis {url}")
            _download_bz2(url, dest)
            print(f"[models] écrit : {dest} ({dest.stat().st_size / 1e6:.1f} Mo)")
            return dest
        except Exception as exc:  # noqa: BLE001 — on tente la source suivante
            last_error = exc
            print(f"[models] échec ({exc}), source suivante…")
    raise RuntimeError(
        f"Impossible de récupérer {filename}. "
        f"Installez `face-recognition-models` ou déposez le fichier dans {PRETRAINED_DIR}/. "
        f"Dernière erreur : {last_error}"
    )


def ensure_model(filename: str) -> Path:
    """Retourne le chemin d'un modèle, en le récupérant si nécessaire."""
    return _local_copy(filename) or _from_package(filename) or _from_network(filename)


def ensure_models(filenames: list[str] | None = None) -> list[Path]:
    return [ensure_model(name) for name in (filenames or REQUIRED)]


if __name__ == "__main__":
    for path in ensure_models(sys.argv[1:] or None):
        print(f"[models] OK {path} ({path.stat().st_size / 1e6:.1f} Mo)")
