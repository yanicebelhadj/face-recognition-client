# back/face_api/Dockerfile
FROM python:3.11-slim

# libgomp : requis à l'exécution par dlib (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Les modèles dlib (~30 Mo) ne sont pas versionnés : on les récupère au build,
# pour que le premier démarrage ne paie pas le téléchargement.
RUN python model_store.py

ENV PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Un seul worker : les modèles dlib sont chargés par process, et le cache de
# sessions vit en RAM dans le process. Plusieurs workers doubleraient la RAM et
# répartiraient les sessions au hasard.
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65"]
