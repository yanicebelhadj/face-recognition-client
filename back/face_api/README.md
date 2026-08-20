# Face API

API FastAPI de reconnaissance faciale, cloisonnée par session.

Documentation générale, architecture et déploiement : voir le [README du
dépôt](../../README.md).

## Lancer en local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py           # http://127.0.0.1:8000 — docs sur /docs
```

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Variables d'environnement

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `PORT` | `8000` | Port d'écoute |
| `EXTRA_CORS_ORIGINS` | — | Origines autorisées supplémentaires (séparées par des virgules) |
| `MAX_DETECT_SIDE` | `640` | Côté max de l'image analysée ; baisser accélère la détection |
| `FACE_TOLERANCE` | `0.6` | Distance max pour associer deux visages (seuil dlib) |
| `MAX_CONCURRENT_INFERENCES` | `2` | Requêtes d'inférence admises en parallèle ; au-delà, réponse 503 |
| `MAX_PROFILES_PER_SESSION` | `25` | Profils max par session |
| `SESSION_TTL_SECONDS` | `21600` | Expiration d'une session inactive (6 h) |
| `MAX_SESSIONS` | `500` | Sessions gardées en RAM (les plus anciennes sont purgées) |
| `MAX_UPLOAD_BYTES` | `8388608` | Taille max d'une image envoyée |
| `PRETRAINED_DIR` | `pretrained_model` | Dossier de surcharge locale des modèles `.dat` |

## Notes d'implémentation

- **Les appels dlib sont sérialisés** (`face_core._inference_lock`). Les objets
  modèles ne sont pas réentrants : deux appels simultanés font tomber le process
  sur un segfault. `MAX_CONCURRENT_INFERENCES` reste utile comme contrôle
  d'admission — mieux vaut un 503 immédiat qu'une file d'attente qui expire.
- **Un seul worker uvicorn.** Le cache de sessions vit en RAM dans le process ;
  plusieurs workers répartiraient les sessions au hasard et dupliqueraient les
  modèles.
- **Prédicteur 5 points** plutôt que 68 : ~10 Mo au lieu de ~100 Mo pour une
  qualité d'embedding équivalente. C'est le choix de l'exemple officiel dlib.
