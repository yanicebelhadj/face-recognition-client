# app.py
"""
API de reconnaissance faciale — cloisonnée par session.

Chaque navigateur génère un identifiant de session aléatoire et l'envoie dans
l'en-tête `X-Session-Id`. Toutes les routes de profils et de reconnaissance sont
résolues dans le périmètre de cette session uniquement : il n'existe aucune
galerie partagée, et aucune route ne permet de lister les visages d'autrui.

Voir `sessions.py` pour le modèle de données et la stratégie de persistance.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import face_core
from sessions import (
    MAX_PROFILES_PER_SESSION,
    Session,
    SessionError,
    as_embedding,
    clean_name,
    new_session_id,
    store,
)

# Une image de webcam compressée pèse quelques dizaines de Ko ; 8 Mo laissent de
# la marge pour une photo d'enrôlement tout en bornant la RAM par requête.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

# dlib sature un cœur CPU par inférence. Sur une instance gratuite (0,1–1 vCPU),
# accepter plus de deux inférences simultanées ne fait qu'empiler de la RAM et
# faire expirer les requêtes : mieux vaut répondre 503 et laisser le client
# ralentir.
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "2"))
INFERENCE_QUEUE_TIMEOUT = float(os.getenv("INFERENCE_QUEUE_TIMEOUT", "8"))

_inference_slots = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
_started_at = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Le chargement des modèles dlib prend plusieurs secondes. On le fait dans
    # un thread pour que /ping réponde immédiatement : le front peut ainsi
    # afficher « démarrage en cours » au lieu de croire le service mort.
    threading.Thread(target=_safe_warmup, name="model-warmup", daemon=True).start()
    yield


def _safe_warmup() -> None:
    try:
        face_core.warmup()
        print("[startup] modèles dlib chargés")
    except Exception as exc:  # noqa: BLE001 — le warmup est un confort, pas un prérequis
        print(f"[startup] échec du préchargement des modèles : {exc}")


app = FastAPI(title="Face API", version="2.0.0", lifespan=lifespan)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://face-recognition-client-iota.vercel.app",  # front Vercel
    "https://yanicebelhadj.fr",  # portfolio
    "https://www.yanicebelhadj.fr",
]
extra_origins = [o.strip() for o in os.getenv("EXTRA_CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + extra_origins,
    # Déploiements de préversion Vercel (une URL différente par commit).
    allow_origin_regex=r"https://face-recognition-client-[a-z0-9-]+\.vercel\.app",
    # Pas de cookie : l'identité tient dans l'en-tête X-Session-Id. Garder
    # allow_credentials à False permet d'énumérer explicitement les en-têtes,
    # ce que la spec CORS interdit de combiner avec les identifiants.
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-Id"],
    max_age=86400,
)


@app.exception_handler(SessionError)
async def _session_error_handler(_, exc: SessionError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


# --------------------------------------------------------------------------- #
# Résolution de session
# --------------------------------------------------------------------------- #
def require_session(x_session_id: str | None = Header(default=None)) -> Session:
    """Session obligatoire : toute route qui touche à des visages passe par là."""
    if not x_session_id:
        raise SessionError(
            "En-tête X-Session-Id manquant. Rechargez la page pour créer une session.",
            status_code=401,
        )
    return store.get_or_create(x_session_id)


def optional_session(x_session_id: str | None = Header(default=None)) -> Session | None:
    """
    Session facultative, pour /recognize : un client sans session reste
    fonctionnel (tout est « Unknown »), mais ne voit jamais les profils d'autrui.
    """
    if not x_session_id:
        return None
    try:
        return store.get_or_create(x_session_id)
    except SessionError:
        return None


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise SessionError("Fichier vide.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise SessionError(
            f"Image trop lourde ({len(data) // 1024} Ko, maximum {MAX_UPLOAD_BYTES // 1024} Ko).",
            status_code=413,
        )
    return data


# --------------------------------------------------------------------------- #
# Santé
# --------------------------------------------------------------------------- #
@app.get("/ping")
def ping():
    """Sonde légère : ne charge aucun modèle, sert à réveiller l'instance."""
    return {
        "status": "ok",
        "models_ready": face_core.models_ready(),
        "uptime_seconds": round(time.time() - _started_at, 1),
    }


@app.get("/health")
def health(session: Session | None = Depends(optional_session)):
    payload = {
        "status": "ok",
        "models_ready": face_core.models_ready(),
        "uptime_seconds": round(time.time() - _started_at, 1),
        "max_profiles_per_session": MAX_PROFILES_PER_SESSION,
        **store.stats(),
    }
    # Volontairement limité à la session appelante : pas de fuite de noms.
    payload["known_faces_count"] = len(session.profiles) if session else 0
    return payload


@app.post("/api/session")
def create_session():
    """Fournit un identifiant si le navigateur n'a pas `crypto.randomUUID`."""
    return {"session_id": new_session_id()}


# --------------------------------------------------------------------------- #
# Profils (périmètre : la session appelante, exclusivement)
# --------------------------------------------------------------------------- #
@app.get("/api/profiles")
def list_profiles(session: Session = Depends(require_session)):
    return {"count": len(session.profiles), "profiles": session.public_profiles()}


@app.post("/api/faces")
async def add_face(
    name: str = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(require_session),
):
    """
    Enrôle un visage. La photo est décodée en mémoire, encodée en vecteur, puis
    **jetée** : seul l'embedding est conservé, et uniquement pour cette session.
    L'embedding est renvoyé au client, qui le stocke localement afin de pouvoir
    restaurer la session sans réenvoyer la photo.
    """
    label = clean_name(name)
    data = await read_upload(file)

    async with _inference_guard():
        embedding, box = await run_in_threadpool(_embed_reference, data)

    if embedding is None:
        raise SessionError(
            "Aucun visage détecté sur cette photo. Essayez une image plus nette, de face et bien éclairée.",
            status_code=422,
        )

    session.remove_by_name(label)  # un nom = un profil
    profile = session.add(label, embedding)
    return {
        "status": "added",
        "profile": {**profile.public(), "embedding": embedding.tolist()},
        "box": box,
    }


def _embed_reference(data: bytes):
    # Une photo d'enrôlement mérite une résolution supérieure à une frame vidéo.
    rgb = face_core.decode_image(data, max_side=1024)
    return face_core.embed_largest_face(rgb)


@app.post("/api/faces/restore")
def restore_faces(
    payload: dict = Body(...),
    session: Session = Depends(require_session),
):
    """
    Réinjecte les profils conservés par le navigateur.

    C'est le mécanisme qui rend la persistance gratuite : après une mise en
    veille ou un redéploiement, le cache serveur est vide ; le client rejoue ses
    embeddings (quelques Ko de JSON) au lieu de réencoder toutes les photos.
    """
    incoming = payload.get("profiles")
    if not isinstance(incoming, list):
        raise SessionError("Charge utile invalide : `profiles` doit être une liste.")
    if len(incoming) > MAX_PROFILES_PER_SESSION:
        raise SessionError(
            f"Trop de profils : maximum {MAX_PROFILES_PER_SESSION}.",
            status_code=409,
        )

    # On valide *tout* avant de toucher à la session : un profil mal formé ne
    # doit pas laisser l'utilisateur avec une session à moitié effacée.
    validated = []
    for item in incoming:
        if not isinstance(item, dict):
            raise SessionError("Charge utile invalide : profil non conforme.")
        validated.append(
            (clean_name(item.get("name", "")), as_embedding(item.get("embedding")), item.get("id"))
        )

    session.replace_all(validated)
    return {"status": "restored", "count": len(session.profiles), "profiles": session.public_profiles()}


@app.delete("/api/faces/{profile_id}")
def delete_face(profile_id: str, session: Session = Depends(require_session)):
    if not session.remove(profile_id):
        raise SessionError("Profil introuvable dans cette session.", status_code=404)
    return {"status": "deleted", "id": profile_id, "count": len(session.profiles)}


@app.delete("/api/faces")
def clear_faces(session: Session = Depends(require_session)):
    return {"status": "cleared", "removed": session.clear()}


@app.delete("/api/session")
def destroy_session(x_session_id: str | None = Header(default=None)):
    """Efface toute trace serveur de la session (bouton « tout effacer »)."""
    dropped = bool(x_session_id) and store.drop(x_session_id)
    return {"status": "destroyed", "existed": dropped}


# --------------------------------------------------------------------------- #
# Reconnaissance
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def _inference_guard():
    """
    Limite les inférences simultanées. Au-delà, on renvoie 503 + Retry-After
    plutôt que d'empiler des requêtes qui finiront en timeout ou en OOM.
    """
    try:
        await asyncio.wait_for(_inference_slots.acquire(), timeout=INFERENCE_QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Serveur momentanément saturé, réessayez dans un instant.",
            headers={"Retry-After": "2"},
        ) from None
    try:
        yield
    finally:
        _inference_slots.release()


@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    session: Session | None = Depends(optional_session),
):
    """
    Reconnaît les visages d'une frame **dans le périmètre de la session**.

    `width`/`height` décrivent l'image effectivement analysée (le serveur
    redimensionne) : le client s'en sert pour replacer les cadres à l'écran.
    """
    data = await read_upload(file)
    known_matrix, known_names = session.known() if session else (None, [])

    async with _inference_guard():
        result = await run_in_threadpool(_recognize_sync, data, known_matrix, known_names)

    result["session_faces"] = len(session.profiles) if session else 0
    return result


def _recognize_sync(data: bytes, known_matrix, known_names) -> dict:
    rgb = face_core.decode_image(data)
    boxes, embeddings = face_core.detect_and_encode(rgb)
    names, distances = face_core.match(embeddings, known_matrix, known_names)
    height, width = rgb.shape[:2]
    return {
        "boxes": boxes,
        "names": names,
        "distances": distances,
        "width": int(width),
        "height": int(height),
    }


# --------------------------------------------------------------------------- #
# Compatibilité avec l'ancien client (le front et l'API ne se déploient pas
# forcément à la même seconde). À supprimer une fois le front à jour partout.
# --------------------------------------------------------------------------- #
@app.get("/profiles", deprecated=True)
def legacy_profiles(session: Session | None = Depends(optional_session)):
    profiles = session.public_profiles() if session else []
    return {"count": len(profiles), "profiles": profiles}


@app.post("/recognize", deprecated=True)
async def legacy_recognize(
    file: UploadFile = File(...),
    session: Session | None = Depends(optional_session),
):
    return await recognize(file=file, session=session)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
