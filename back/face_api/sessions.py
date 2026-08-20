# sessions.py
"""
Cloisonnement des visages par session.

Modèle de données : **le navigateur est la source de vérité**, le serveur n'est
qu'un cache de calcul.

- Les *photos* ne quittent jamais le poste de l'utilisateur autrement que le
  temps d'un calcul d'embedding : elles ne sont jamais écrites sur le disque du
  serveur. C'est ce qui garantit qu'une photo ajoutée sur un poste ne peut pas
  apparaître chez quelqu'un d'autre.
- Le serveur ne conserve, en RAM, que le vecteur 128-D nécessaire à la
  comparaison, rangé sous l'identifiant de session du navigateur.
- Ce cache est volatile (redéploiement, mise en veille de l'offre gratuite,
  expiration). Le navigateur détecte la perte et rejoue `restore()` avec les
  embeddings qu'il a gardés en IndexedDB : la persistance « d'une session à
  l'autre » ne coûte donc aucun stockage serveur.

Un identifiant de session est un secret aléatoire de 128 bits généré par le
client : le connaître est la seule façon d'accéder aux visages associés.
"""
from __future__ import annotations

import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from face_core import EMBEDDING_DIM

SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Une session inactive est purgée : rien ne sert de garder des données
# biométriques en RAM pour un onglet fermé depuis longtemps.
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(6 * 3600)))

# Garde-fous pour une instance gratuite (512 Mo) : ~1 Ko par profil, donc le
# coût réel est négligeable, mais ces bornes empêchent un abus de faire tomber
# le service pour tout le monde.
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))
MAX_PROFILES_PER_SESSION = int(os.getenv("MAX_PROFILES_PER_SESSION", "25"))

MAX_NAME_LENGTH = 40


class SessionError(Exception):
    """Erreur métier destinée à être traduite en réponse HTTP 4xx."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def new_session_id() -> str:
    return uuid.uuid4().hex


def is_valid_session_id(session_id: str | None) -> bool:
    return bool(session_id) and bool(SESSION_ID_RE.match(session_id))


def clean_name(raw: str) -> str:
    """Normalise un nom de profil et refuse ce qui n'en est pas un."""
    name = " ".join((raw or "").split())
    if not name:
        raise SessionError("Le nom est vide.")
    if len(name) > MAX_NAME_LENGTH:
        raise SessionError(f"Le nom ne doit pas dépasser {MAX_NAME_LENGTH} caractères.")
    return name


def as_embedding(values) -> np.ndarray:
    """Valide un embedding reçu du client (chemin `restore`)."""
    array = np.asarray(values, dtype=np.float32).ravel()
    if array.shape != (EMBEDDING_DIM,):
        raise SessionError(f"Embedding invalide : {EMBEDDING_DIM} valeurs attendues.")
    if not np.isfinite(array).all():
        raise SessionError("Embedding invalide : valeurs non finies.")
    return array


@dataclass
class Profile:
    id: str
    name: str
    embedding: np.ndarray
    created_at: float

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "created_at": self.created_at}


@dataclass
class Session:
    id: str
    created_at: float
    last_seen: float
    profiles: dict[str, Profile] = field(default_factory=dict)

    # Cache du calcul matriciel, invalidé à chaque modification. Matrice et noms
    # forment un seul champ : deux requêtes concurrentes de la même session ne
    # doivent jamais lire une matrice à jour avec les noms d'avant.
    _known: tuple[np.ndarray, list[str]] | None = field(default=None, repr=False)

    def _invalidate(self) -> None:
        self._known = None

    def known(self) -> tuple[np.ndarray, list[str]]:
        """(matrice [N,128], noms) prête pour le matching."""
        cached = self._known
        if cached is None:
            profiles = list(self.profiles.values())
            cached = (
                (np.stack([p.embedding for p in profiles]).astype(np.float32), [p.name for p in profiles])
                if profiles
                else (np.empty((0, EMBEDDING_DIM), dtype=np.float32), [])
            )
            self._known = cached
        return cached

    def add(self, name: str, embedding: np.ndarray, profile_id: str | None = None) -> Profile:
        if profile_id is None or profile_id not in self.profiles:
            if len(self.profiles) >= MAX_PROFILES_PER_SESSION:
                raise SessionError(
                    f"Limite de {MAX_PROFILES_PER_SESSION} profils atteinte pour cette session.",
                    status_code=409,
                )
        profile = Profile(
            id=profile_id or uuid.uuid4().hex,
            name=name,
            embedding=embedding,
            created_at=time.time(),
        )
        self.profiles[profile.id] = profile
        self._invalidate()
        return profile

    def remove(self, profile_id: str) -> bool:
        removed = self.profiles.pop(profile_id, None) is not None
        if removed:
            self._invalidate()
        return removed

    def remove_by_name(self, name: str) -> list[str]:
        target = name.casefold()
        ids = [pid for pid, p in self.profiles.items() if p.name.casefold() == target]
        for pid in ids:
            self.profiles.pop(pid, None)
        if ids:
            self._invalidate()
        return ids

    def replace_all(self, entries: list[tuple[str, np.ndarray, str | None]]) -> None:
        """Remplace d'un bloc le contenu de la session (chemin `restore`)."""
        if len(entries) > MAX_PROFILES_PER_SESSION:
            raise SessionError(
                f"Limite de {MAX_PROFILES_PER_SESSION} profils atteinte pour cette session.",
                status_code=409,
            )
        now = time.time()
        rebuilt: dict[str, Profile] = {}
        for name, embedding, profile_id in entries:
            identifier = profile_id if isinstance(profile_id, str) and profile_id else uuid.uuid4().hex
            rebuilt[identifier] = Profile(
                id=identifier, name=name, embedding=embedding, created_at=now
            )
        self.profiles = rebuilt
        self._invalidate()

    def clear(self) -> int:
        count = len(self.profiles)
        self.profiles.clear()
        self._invalidate()
        return count

    def public_profiles(self) -> list[dict]:
        # Les dicts Python conservent l'ordre d'insertion : c'est l'ordre d'ajout
        # côté utilisateur, et l'ordre du navigateur après une restauration.
        return [p.public() for p in self.profiles.values()]


class SessionStore:
    """Registre des sessions, protégé par un verrou (uvicorn = threadpool)."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> Session:
        if not is_valid_session_id(session_id):
            raise SessionError("Identifiant de session invalide.", status_code=400)
        now = time.time()
        with self._lock:
            self._evict(now)
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(id=session_id, created_at=now, last_seen=now)
                self._sessions[session_id] = session
            session.last_seen = now
            return session

    def peek(self, session_id: str) -> Session | None:
        """Accès sans création — utilisé par /health et les stats."""
        with self._lock:
            return self._sessions.get(session_id)

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict(self, now: float) -> None:
        """Purge les sessions expirées, puis les plus anciennes si on déborde."""
        expired = [sid for sid, s in self._sessions.items() if now - s.last_seen > SESSION_TTL_SECONDS]
        for sid in expired:
            self._sessions.pop(sid, None)

        overflow = len(self._sessions) - MAX_SESSIONS
        if overflow > 0:
            oldest = sorted(self._sessions.items(), key=lambda item: item[1].last_seen)
            for sid, _ in oldest[:overflow]:
                self._sessions.pop(sid, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "profiles": sum(len(s.profiles) for s in self._sessions.values()),
            }


store = SessionStore()
