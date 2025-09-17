# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np, cv2, io, os
from PIL import Image
from pathlib import Path

from face_core import load_known_faces, recognize_on_frame


KNOWN_DIR = os.getenv("KNOWN_FACES_DIR", "known_faces")

app = FastAPI(title="Face API")

origins = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:3000", "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

known_encodings = None
known_names = None

@app.on_event("startup")
def _load_db():
    global known_encodings, known_names
    try:
        known_encodings, known_names = load_known_faces(KNOWN_DIR)
    except Exception as e:
        print(f"[WARN] {e}")
        known_encodings, known_names = np.empty((0,128)), []

@app.get("/ping")
def ping(): return {"status": "ok"}

@app.get("/health")
def health():
    return {"known_faces_count": int(len(known_encodings)), "names": known_names}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    boxes, names, landmarks = recognize_on_frame(frame_bgr, known_encodings, known_names)
    print(f"[DEBUG] faces={len(boxes)} size={img.shape[1]}x{img.shape[0]}")
    return {"boxes": boxes, "names": names, "landmarks_counts": [len(l) for l in landmarks]}


@app.post("/recognize/draw")
async def recognize_draw(file: UploadFile = File(...)):
    data = await file.read()
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    boxes, names, _ = recognize_on_frame(frame_bgr, known_encodings, known_names)
    for (t,r,b,l), name in zip(boxes, names):
        cv2.rectangle(frame_bgr,(l,t),(r,b),(0,255,0),2)
        cv2.rectangle(frame_bgr,(l,b-24),(r,b),(0,255,0),cv2.FILLED)
        cv2.putText(frame_bgr,name,(l+3,b-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", frame_rgb)
    return Response(content=buf.tobytes(), media_type="image/png")

@app.post("/add-face")
async def add_face(name: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    Path(KNOWN_DIR).mkdir(parents=True, exist_ok=True)
    img.save(Path(KNOWN_DIR)/f"{name}.jpg")
    global known_encodings, known_names
    known_encodings, known_names = load_known_faces(KNOWN_DIR)
    return {"status":"added","name":name}

@app.post("/reload-known-faces")
def reload_faces():
    global known_encodings, known_names
    known_encodings, known_names = load_known_faces(KNOWN_DIR)
    return {"status":"reloaded","count": int(len(known_encodings))}
