# back/face_api/Dockerfile
FROM python:3.12-slim

# OS deps (opencv, dlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# copie du code + modèles
COPY . /app

# variables
ENV PORT=8000
ENV KNOWN_FACES_DIR=known_faces

# expose
EXPOSE 8000

# launch
CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8000"]
