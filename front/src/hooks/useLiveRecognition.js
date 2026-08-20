/**
 * Boucle de reconnaissance en direct.
 *
 * Deux choix guident cette implémentation, tous deux liés au fait que l'API
 * tourne sur une petite instance gratuite :
 *
 * 1. **Une requête à la fois, jamais de minuteur fixe.** L'ancienne version
 *    envoyait une image toutes les 50 ms, soit vingt fois plus vite que dlib ne
 *    sait les traiter : la file d'attente enflait jusqu'à la saturation
 *    mémoire. Ici, la frame suivante n'est capturée qu'une fois la précédente
 *    traitée, et la cadence s'ajuste au temps de réponse observé.
 * 2. **Des images légères.** JPEG en 480 px de large plutôt que PNG pleine
 *    résolution : quelques dizaines de Ko au lieu de plusieurs centaines.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { recognize } from "../api";

const CAPTURE_WIDTH = 480;
const JPEG_QUALITY = 0.6;

// Cadence : jamais plus vite que MIN_GAP, jamais plus lent que MAX_GAP.
const MIN_GAP_MS = 90;
const MAX_GAP_MS = 1500;

// Après une erreur, on lève le pied progressivement (réveil de l'instance).
const ERROR_BACKOFF_MS = [1000, 2000, 4000, 8000];

const BOX_COLOR = "#00e05a";
const UNKNOWN_COLOR = "#ff9f1c";

function drawOverlay(canvas, result) {
  const context = canvas.getContext("2d");
  const { clientWidth: width, clientHeight: height } = canvas;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;

  context.clearRect(0, 0, width, height);
  if (!result) return;

  // Les cadres sont exprimés dans le repère de l'image analysée par le serveur,
  // qui la redimensionne : on repasse dans le repère affiché.
  const scaleX = width / result.width;
  const scaleY = height / result.height;

  context.lineWidth = 2;
  context.font = "600 14px ui-sans-serif, system-ui, sans-serif";
  context.textBaseline = "top";

  result.boxes.forEach(([top, right, bottom, left], index) => {
    const name = result.names[index] || "Unknown";
    const known = name !== "Unknown";
    const color = known ? BOX_COLOR : UNKNOWN_COLOR;

    const x = left * scaleX;
    const y = top * scaleY;
    const boxWidth = (right - left) * scaleX;
    const boxHeight = (bottom - top) * scaleY;

    context.strokeStyle = color;
    context.strokeRect(x, y, boxWidth, boxHeight);

    const label = known ? name : "Inconnu";
    const padding = 6;
    const labelHeight = 22;
    const labelWidth = context.measureText(label).width + padding * 2;

    context.fillStyle = color;
    context.fillRect(x, y + boxHeight - labelHeight, labelWidth, labelHeight);
    context.fillStyle = "#06130b";
    context.fillText(label, x + padding, y + boxHeight - labelHeight + 4);
  });
}

function captureFrame(video) {
  const width = CAPTURE_WIDTH;
  const height = Math.round(video.videoHeight * (width / video.videoWidth)) || width;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(video, 0, 0, width, height);

  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));
}

/**
 * @param {object} options
 * @param {boolean} options.active        démarre la boucle quand vrai
 * @param {number}  options.expectedFaces nombre de profils censés être connus de l'API
 * @param {Function} options.onDrift      appelé si l'API en connaît un nombre différent
 */
export function useLiveRecognition({ active, expectedFaces, onDrift }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [cameraError, setCameraError] = useState("");
  const [cameraReady, setCameraReady] = useState(false);
  const [latencyMs, setLatencyMs] = useState(null);
  const [faceCount, setFaceCount] = useState(0);

  // Lus dans la boucle : des refs évitent de la redémarrer à chaque rendu.
  const activeRef = useRef(active);
  const expectedRef = useRef(expectedFaces);
  const driftRef = useRef(onDrift);
  activeRef.current = active;
  expectedRef.current = expectedFaces;
  driftRef.current = onDrift;

  // Caméra
  useEffect(() => {
    let stream = null;
    let cancelled = false;

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, facingMode: "user" },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        const video = videoRef.current;
        if (!video) return;
        video.srcObject = stream;
        await video.play().catch(() => {});
        setCameraReady(true);
      } catch (error) {
        setCameraError(
          error?.name === "NotAllowedError"
            ? "Accès à la caméra refusé. Autorisez-le dans votre navigateur pour lancer la détection."
            : "Aucune caméra disponible sur cet appareil.",
        );
      }
    })();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  // Boucle d'analyse
  useEffect(() => {
    if (!cameraReady) return undefined;

    let stopped = false;
    let timer = null;
    let consecutiveErrors = 0;

    const schedule = (delay) => {
      if (!stopped) timer = setTimeout(tick, delay);
    };

    async function tick() {
      if (stopped) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!activeRef.current || !video?.videoWidth || !canvas) {
        schedule(400);
        return;
      }

      const startedAt = performance.now();
      try {
        const blob = await captureFrame(video);
        if (!blob || stopped) return schedule(MIN_GAP_MS);

        const result = await recognize(new File([blob], "frame.jpg", { type: "image/jpeg" }));
        if (stopped) return undefined;

        drawOverlay(canvas, result);
        setFaceCount(result.boxes.length);

        // L'API a redémarré et perdu le cache de la session : on prévient pour
        // qu'il soit rejoué depuis le stockage local.
        if (typeof result.session_faces === "number" && result.session_faces !== expectedRef.current) {
          driftRef.current?.();
        }

        const elapsed = performance.now() - startedAt;
        setLatencyMs(Math.round(elapsed));
        consecutiveErrors = 0;

        // On laisse au serveur un temps de respiration proportionnel à sa
        // lenteur : inutile de le saturer s'il met déjà une seconde à répondre.
        return schedule(Math.min(MAX_GAP_MS, Math.max(MIN_GAP_MS, elapsed * 0.25)));
      } catch {
        consecutiveErrors += 1;
        if (consecutiveErrors >= 3) {
          drawOverlay(canvas, null);
          setFaceCount(0);
        }
        return schedule(ERROR_BACKOFF_MS[Math.min(consecutiveErrors - 1, ERROR_BACKOFF_MS.length - 1)]);
      }
    }

    tick();

    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [cameraReady]);

  const clearOverlay = useCallback(() => {
    if (canvasRef.current) drawOverlay(canvasRef.current, null);
  }, []);

  return { videoRef, canvasRef, cameraError, cameraReady, latencyMs, faceCount, clearOverlay };
}
