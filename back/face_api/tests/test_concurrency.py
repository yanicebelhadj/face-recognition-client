"""
Régression : les modèles dlib ne supportent pas les appels concurrents.

Deux appels simultanés sur la même instance de `shape_predictor` /
`face_recognition_model_v1` corrompent son état interne et tuent le process sur
un segfault — pas une exception, donc rien à rattraper : l'API entière tombe.
C'était la cause des indisponibilités intermittentes.

`face_core` sérialise désormais chaque appel. Ce test échoue en *crashant* si la
protection disparaît, d'où l'exécution dans un sous-process.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    """
    import sys, threading
    import numpy as np
    from PIL import Image
    import io

    sys.path.insert(0, {backend!r})
    import face_core

    face_core.warmup()

    # Un dégradé suffit : on cherche à faire tourner le détecteur en parallèle,
    # pas à reconnaître qui que ce soit.
    gradient = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))
    frame = np.ascontiguousarray(np.dstack([gradient] * 3))
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, "JPEG")
    payload = buffer.getvalue()

    def worker():
        for _ in range(6):
            rgb = face_core.decode_image(payload)
            face_core.detect_and_encode(rgb)
            face_core.embed_largest_face(rgb)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("OK")
    """
).format(backend=str(BACKEND))


def test_concurrent_inference_does_not_crash():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, (
        f"appels dlib concurrents instables (code {result.returncode}) :\n{result.stderr}"
    )
    assert "OK" in result.stdout
