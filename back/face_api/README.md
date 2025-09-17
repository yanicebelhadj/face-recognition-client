# Face Recognition Client

Application **full-stack** de reconnaissance faciale en **temps réel**.  
Basée sur **Python/dlib** pour l’API et **React (Vite)** pour le client web.

---

## 🚀 Installation & Lancement

### 1) Back (API Python)

cd back/face_api
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
Lancer l’API :

python app.py


Les visages connus doivent être placés dans :

back/face_api/known_faces/
├── Yanice.png
├── Zuckerberg.png
└── ...

---

### 2) Front (React + Vite)
cd front
npm install
npm run dev


Par défaut l’app sera dispo sur :
👉 http://localhost:5173

## ⚙️ Fonctionnement

Le front (React) ouvre la webcam et capture les frames.

Les images sont envoyées à l’API Python (app.py).

face_core.py génère un vecteur 128D dlib et compare aux embeddings des known_faces/.

Le nom du match (ou "Unknown") est renvoyé au front et affiché à l’écran.

## 🛠️ Dépendances principales
### Back

Python 3.9+

dlib

opencv-python

numpy

imutils

Pillow

### Front

Node.js 18+

React 18

Vite

## 👤 Auteur

Yanice Belhadj — Software Development Engineer

## 📝 Licence

MIT — voir LICENSE.md
```bash
