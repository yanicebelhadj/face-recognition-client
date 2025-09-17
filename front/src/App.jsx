import React, { useEffect, useRef, useState } from "react";
import { ping, health, recognize, recognizeDraw } from "./api";

export default function App() {
  const [status, setStatus] = useState("…");
  const [healthInfo, setHealthInfo] = useState(null);
  const [jsonResult, setJsonResult] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);

  // Webcam refs
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const loopRef = useRef(null); // id du setInterval

  useEffect(() => {
    (async () => {
      try {
        const p = await ping(); setStatus(p.status || "ok");
        const h = await health(); setHealthInfo(h);
      } catch { setStatus("down"); }
    })();
  }, []);

  // Démarrer webcam + boucle live
  useEffect(() => {
    let stopped = false;

    async function startCam() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;

      // attendre la taille vidéo
      await new Promise(res => {
        videoRef.current.onloadedmetadata = () => res();
      });

      const video = videoRef.current;
      const canvas = overlayRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Boucle live (5 FPS = toutes les 200 ms)
      loopRef.current = setInterval(async () => {
        if (stopped) return;
        try {
          // capture → blob
          const tmp = document.createElement("canvas");
          tmp.width = video.videoWidth; tmp.height = video.videoHeight;
          tmp.getContext("2d").drawImage(video, 0, 0, tmp.width, tmp.height);
          const blob = await new Promise(r => tmp.toBlob(r, "image/png"));
          const file = new File([blob], "frame.png", { type: "image/png" });

          // appel JSON (plus léger que l'image annotée)
          const form = new FormData();
          form.append("file", file);
          const r = await fetch(import.meta.env.VITE_API_URL + "/recognize", { method: "POST", body: form });
          if (!r.ok) return;
          const data = await r.json();

          drawOverlay(canvas, data);
        } catch (e) {
          // silencieux pour éviter de spam la console
        }
      }, 200);
    }

    startCam();

    return () => {
      stopped = true;
      if (loopRef.current) clearInterval(loopRef.current);
      const tracks = videoRef.current?.srcObject?.getTracks?.() || [];
      tracks.forEach(t => t.stop());
    };
  }, []);

  // Dessin overlay
  function drawOverlay(canvas, data) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  
    (data.faces || []).forEach(face => {
      const [top, right, bottom, left] = face.box;
      const a = face.attributes || {};
      const label = `${a.age ?? "?"}y ${a.hair_color || "?"} hair ${a.eye_color || "?"} eyes`;
  
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgb(0,255,0)";
      ctx.strokeRect(left, top, right - left, bottom - top);
  
      ctx.font = "16px ui-sans-serif";
      ctx.textBaseline = "top";
      const pad = 4, textH = 18;
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = "rgb(0,255,0)";
      ctx.fillRect(left, bottom - textH, textW + pad * 2, textH);
      ctx.fillStyle = "black";
      ctx.fillText(label, left + pad, bottom - textH + 2);
    });
  }
  

  // --- le reste (tests JSON / image annotée) peut rester identique ---

  return (
    <div style={{ padding: 24, fontFamily: "ui-sans-serif, system-ui" }}>
      <h1>Face Recognition — Front React</h1>
      <p>Backend status: <b>{status}</b></p>
      {healthInfo && <p>Known faces: <b>{healthInfo.known_faces_count}</b></p>}

      <hr style={{ margin: "16px 0" }} />

      <h3>Webcam — live</h3>
      <div style={{ position: "relative", width: "min(900px, 100%)" }}>
        <video ref={videoRef} autoPlay playsInline
               style={{ width: "100%", display: "block", background: "#000" }} />
        <canvas ref={overlayRef}
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
      </div>

      {/* (Sections tests JSON / draw gardées si tu veux) */}
    </div>
  );
}
