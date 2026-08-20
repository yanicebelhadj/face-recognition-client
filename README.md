# Face Recognition Client

Reconnaissance faciale en temps réel dans le navigateur : **React + Vite** côté
client, **FastAPI + dlib** côté API.

Chaque visiteur dispose de sa propre session : les visages qu'il enregistre
n'existent que pour lui, et il retrouve les siens à chaque visite.

---

## Confidentialité : qui détient quoi

C'est le point structurant de l'architecture — et la raison pour laquelle
l'hébergement ne coûte rien.

| Donnée | Où elle vit | Durée |
| --- | --- | --- |
| Photos | Navigateur (IndexedDB) | Jusqu'à suppression par l'utilisateur |
| Empreintes 128-D | Navigateur + cache RAM de l'API | Cache purgé après 6 h d'inactivité |
| Identifiant de session | `localStorage` du navigateur | Jusqu'à « Tout effacer » |

**Le serveur n'écrit jamais de photo sur son disque.** Une image envoyée à
`/api/faces` est décodée en mémoire, convertie en vecteur de 128 nombres, puis
jetée. Le vecteur seul est conservé, en RAM, sous l'identifiant de session du
navigateur qui l'a produit.

Il n'existe aucune galerie partagée : aucune route ne permet de lister les
visages d'une autre session. Deux postes ne peuvent donc pas voir les profils
l'un de l'autre, et deux onglets d'un même navigateur partagent les leurs.

L'identifiant de session est un secret aléatoire de 128 bits tiré par le
navigateur. Il n'apparaît jamais dans une URL, uniquement dans l'en-tête
`X-Session-Id`.

### Persistance sans stockage serveur

L'API tourne sur une offre gratuite : son cache disparaît à chaque mise en
veille ou redéploiement. Le navigateur étant la source de vérité, il détecte
l'écart et rejoue ses empreintes (`POST /api/faces/restore`, quelques Ko de
JSON) — sans réenvoyer ni recalculer la moindre photo.

Conséquence à connaître : vider les données du site efface les visages, et un
autre navigateur ou un autre appareil repart d'une session vierge.

---

## Démarrage

### API

```bash
cd back/face_api
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                        # http://127.0.0.1:8000
```

Les modèles dlib (~32 Mo) proviennent du paquet `face-recognition-models`,
installé avec les dépendances : rien à télécharger à la main.

### Client

```bash
cd front
npm install
npm run dev                                          # http://localhost:5173
```

L'URL de l'API se configure via `VITE_API_URL` (voir `front/.env`). En
production, définissez cette variable dans le tableau de bord de l'hébergeur.

### Tests

```bash
cd back/face_api
pip install pytest httpx
pytest tests/ -v
```

Ils couvrent le cloisonnement des sessions, l'atomicité de la restauration, et
la non-régression du crash lié aux appels dlib concurrents.

---

## Déploiement gratuit

`render.yaml` décrit le service API sur le plan gratuit de Render ; le client
se déploie sur Vercel ou GitHub Pages.

Deux caractéristiques du plan gratuit sont assumées dans le code plutôt que
subies :

- **Mise en veille.** L'instance s'endort après ~15 min sans trafic et met 30 à
  60 s à redémarrer. Le client affiche « Démarrage du serveur… » et réessaie
  avec un délai croissant, au lieu de conclure à une panne.
- **Disque éphémère.** Rien n'y est écrit — voir la section persistance.

Ajoutez vos propres domaines aux origines CORS via la variable
`EXTRA_CORS_ORIGINS` (liste séparée par des virgules).

---

## API

Toutes les routes `/api/*` exigent l'en-tête `X-Session-Id` (32 caractères
hexadécimaux) et ne voient que les données de cette session.

| Méthode | Route | Rôle |
| --- | --- | --- |
| `GET` | `/ping` | Sonde légère, sert à réveiller l'instance |
| `GET` | `/health` | État du service et nombre de profils de l'appelant |
| `GET` | `/api/profiles` | Profils de la session |
| `POST` | `/api/faces` | Enrôle un visage (`name` + `file`), renvoie son empreinte |
| `POST` | `/api/faces/restore` | Rejoue les empreintes gardées par le navigateur |
| `DELETE` | `/api/faces/{id}` | Supprime un profil |
| `DELETE` | `/api/faces` | Vide la session |
| `DELETE` | `/api/session` | Efface toute trace serveur de la session |
| `POST` | `/api/recognize` | Analyse une image, renvoie cadres et noms |

---

## Structure

```
back/face_api/
  app.py           routes HTTP, CORS, contrôle de charge
  sessions.py      cloisonnement par session, limites, expiration
  face_core.py     détection / encodage / matching (sans état, sans I/O)
  model_store.py   résolution des modèles dlib
  tests/           cloisonnement et non-régression
front/src/
  api.js                    client HTTP (en-tête de session, réessais)
  lib/session.js            identifiant de session
  lib/faceDb.js             stockage local des photos et empreintes
  hooks/useFaceProfiles.js  profils + synchronisation avec l'API
  hooks/useLiveRecognition.js  webcam et boucle d'analyse
  hooks/useApiStatus.js     état du serveur et réveil
```
