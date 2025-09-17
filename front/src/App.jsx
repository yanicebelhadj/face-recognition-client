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
  const inflightRef = useRef(false);

  const CAP_W = 640; // capture à largeur fixe pour maitriser le scale
  let capH = null;   // sera calculé dès que la vidéo est prête


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
    (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const video = videoRef.current;
      video.srcObject = stream;
      await new Promise(r => video.onloadedmetadata = r);
  
      // Dimensions de capture normales
      capH = Math.round(video.videoHeight * (CAP_W / video.videoWidth));
  
      // Ajuste l'overlay pour "coller" visuellement à la vidéo visible
      const overlay = overlayRef.current;
      const rect = video.getBoundingClientRect();
      overlay.style.position = "absolute";
      overlay.style.left = "0";
      overlay.style.top = "0";
      overlay.style.width = "100%";
      overlay.style.height = "100%";
  
      loopRef.current = setInterval(async () => {
        if (stopped || inflightRef.current) return;
        inflightRef.current = true;
        try {
          // 1) Capture à CAP_W x capH (réduit la charge et fixe l’échelle)
          const tmp = document.createElement("canvas");
          tmp.width = CAP_W; tmp.height = capH;
          const tctx = tmp.getContext("2d");
          tctx.drawImage(video, 0, 0, tmp.width, tmp.height);
          const blob = await new Promise(r => tmp.toBlob(r, "image/png"));
          const file = new File([blob], "frame.png", { type: "image/png" });
  
          // 2) Envoi au back
          const form = new FormData();
          form.append("file", file);
          const r = await fetch(import.meta.env.VITE_API_URL + "/recognize", { method: "POST", body: form });
          const data = await recognize(file);
          console.log("DATA from API:", data);
          drawOverlay(overlayRef.current, data, CAP_W, capH);
          if (r.ok) {
            const data = await r.json();
            drawOverlay(overlayRef.current, data, CAP_W, capH); // passer dims de capture
          }
        } catch (e) {
          // no-op
        } finally {
          inflightRef.current = false;
        }
      }, 50); 
    })();
  
    return () => {
      stopped = true;
      if (loopRef.current) clearInterval(loopRef.current);
      const tracks = videoRef.current?.srcObject?.getTracks?.() || [];
      tracks.forEach(t => t.stop());
    };
  }, []);

  // Dessin overlay
  function drawOverlay(canvas, data, srcW, srcH) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
  
    // Ajuste le canvas à la taille affichée de la vidéo (CSS)
    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;
    if (canvas.width !== displayW) canvas.width = displayW;
    if (canvas.height !== displayH) canvas.height = displayH;
  
    const scaleX = displayW / srcW;
    const scaleY = displayH / srcH;
  
    ctx.clearRect(0, 0, displayW, displayH);
    ctx.lineWidth = 3;
    ctx.font = "16px ui-sans-serif";
    ctx.textBaseline = "top";
  
    const faces = data.faces || null;
    const boxes = faces ? faces.map(f => f.box) : (data.boxes || []);
    const labels = faces ? faces.map(f => f.attributes ? `${f.attributes.age ?? "?"}y` : "Face") : (data.names || []);
  
    for (let i = 0; i < boxes.length; i++) {
      const [top, right, bottom, left] = boxes[i];
      const x = Math.round(left * scaleX);
      const y = Math.round(top * scaleY);
      const w = Math.round((right - left) * scaleX);
      const h = Math.round((bottom - top) * scaleY);
  
      ctx.strokeStyle = "rgb(0,255,0)";
      ctx.strokeRect(x, y, w, h);
  
      const label = labels[i] || "Unknown";
      const pad = 4, textH = 18;
      const textW = ctx.measureText(label).width;
      ctx.fillStyle = "rgb(0,255,0)";
      ctx.fillRect(x, y + h - textH, textW + pad * 2, textH);
      ctx.fillStyle = "black";
      ctx.fillText(label, x + pad, y + h - textH + 2);
    }
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
