"""
Tests du cloisonnement par session.

Ils passent volontairement par `/api/faces/restore` (embeddings injectés
directement) plutôt que par l'enrôlement photo : aucune image de visage n'est
versionnée dans ce dépôt, et la propriété qu'on veut garantir ici — « une
session ne voit jamais les données d'une autre » — ne dépend pas de dlib.

    pip install -r requirements.txt pytest httpx
    pytest tests/ -v
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def session_id() -> str:
    return uuid.uuid4().hex


def fake_profile(name: str, seed: int) -> dict:
    generator = np.random.default_rng(seed)
    return {"name": name, "embedding": generator.normal(size=128).tolist()}


def enrol(client: TestClient, sid: str, *profiles: dict):
    response = client.post(
        "/api/faces/restore", headers={"X-Session-Id": sid}, json={"profiles": list(profiles)}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_profiles_are_scoped_to_their_session(client):
    alice, bob = session_id(), session_id()
    enrol(client, alice, fake_profile("Alice", 1))
    enrol(client, bob, fake_profile("Bob", 2))

    seen_by_alice = client.get("/api/profiles", headers={"X-Session-Id": alice}).json()
    seen_by_bob = client.get("/api/profiles", headers={"X-Session-Id": bob}).json()

    assert [p["name"] for p in seen_by_alice["profiles"]] == ["Alice"]
    assert [p["name"] for p in seen_by_bob["profiles"]] == ["Bob"]


def test_unknown_session_sees_nothing(client):
    enrol(client, session_id(), fake_profile("Alice", 1))
    fresh = client.get("/api/profiles", headers={"X-Session-Id": session_id()}).json()
    assert fresh == {"count": 0, "profiles": []}


def test_health_never_leaks_names(client):
    enrol(client, session_id(), fake_profile("Alice", 1))
    payload = client.get("/health").json()
    assert "names" not in payload
    assert payload["known_faces_count"] == 0


def test_writes_require_a_session(client):
    assert client.get("/api/profiles").status_code == 401
    assert client.post("/api/faces/restore", json={"profiles": []}).status_code == 401


def test_malformed_session_id_is_rejected(client):
    for bad in ["../etc", "short", "g" * 32, ""]:
        assert client.get("/api/profiles", headers={"X-Session-Id": bad}).status_code in (400, 401)


def test_cannot_delete_another_sessions_profile(client):
    alice, bob = session_id(), session_id()
    created = enrol(client, alice, fake_profile("Alice", 1))
    profile_id = created["profiles"][0]["id"]

    assert client.delete(f"/api/faces/{profile_id}", headers={"X-Session-Id": bob}).status_code == 404
    assert client.get("/api/profiles", headers={"X-Session-Id": alice}).json()["count"] == 1


def test_restore_is_atomic(client):
    """Une charge utile invalide ne doit pas laisser la session à moitié vidée."""
    alice = session_id()
    enrol(client, alice, fake_profile("Alice", 1))

    response = client.post(
        "/api/faces/restore",
        headers={"X-Session-Id": alice},
        json={"profiles": [fake_profile("Bob", 2), {"name": "Cassé", "embedding": [1, 2, 3]}]},
    )
    assert response.status_code == 400
    assert [p["name"] for p in client.get("/api/profiles", headers={"X-Session-Id": alice}).json()["profiles"]] == [
        "Alice"
    ]


def test_restore_round_trip_preserves_ids_and_order(client):
    """Le navigateur rejoue ses profils après un redémarrage de l'instance."""
    alice = session_id()
    created = enrol(client, alice, fake_profile("Alice", 1), fake_profile("Bob", 2))
    saved = [
        {"id": p["id"], "name": p["name"], "embedding": fake_profile(p["name"], i + 1)["embedding"]}
        for i, p in enumerate(created["profiles"])
    ]

    from sessions import store

    store.drop(alice)
    assert client.get("/api/profiles", headers={"X-Session-Id": alice}).json()["count"] == 0

    restored = enrol(client, alice, *saved)
    assert [p["id"] for p in restored["profiles"]] == [p["id"] for p in saved]
    assert [p["name"] for p in restored["profiles"]] == ["Alice", "Bob"]


def test_profile_limit_is_enforced(client):
    from sessions import MAX_PROFILES_PER_SESSION

    too_many = [fake_profile(f"P{i}", i) for i in range(MAX_PROFILES_PER_SESSION + 1)]
    response = client.post(
        "/api/faces/restore", headers={"X-Session-Id": session_id()}, json={"profiles": too_many}
    )
    assert response.status_code == 409


def test_recognize_without_session_returns_no_match(client):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), (12, 34, 56)).save(buffer, "JPEG")
    response = client.post("/api/recognize", files={"file": ("frame.jpg", buffer.getvalue(), "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["names"] == []
    assert response.json()["session_faces"] == 0
