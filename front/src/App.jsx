import React, { useEffect, useRef, useState } from "react";
import { ping, health, recognize } from "./api";
import ProfilesPanel from "./components/ProfilesPanel";
import "./styles/app.scss";

export default function App() {
  const [status, setStatus] = useState("…");
  const [healthInfo, setHealthInfo] = useState(null);

  // Webcam refs
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const loopRef = useRef(null);      // id du setInterval
  const inflightRef = useRef(false);

  const CAP_W = 640;                 // largeur de capture
  let capH = null;                   // hauteur calculée à l’initialisation

  useEffect(() => {
    (async () => {
      try {
        const p = await ping(); setStatus(p.status || "ok");
        const h = await health(); setHealthInfo(h);
      } catch {
        setStatus("down");
      }
    })();
  }, []);

  // Démarrer webcam + boucle live
  useEffect(() => {
    let stopped = false;
    (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      const video = videoRef.current;
      video.srcObject = stream;
      await new Promise(r => (video.onloadedmetadata = r));

      // Dimensions de capture
      capH = Math.round(video.videoHeight * (CAP_W / video.videoWidth));

      loopRef.current = setInterval(async () => {
        if (stopped || inflightRef.current) return;
        inflightRef.current = true;
        try {
          // 1) Capture à CAP_W x capH (réduit la charge et fixe l’échelle)
          const tmp = document.createElement("canvas");
          tmp.width = CAP_W;
          tmp.height = capH;
          const tctx = tmp.getContext("2d");
          tctx.drawImage(video, 0, 0, tmp.width, tmp.height);
          const blob = await new Promise(r => tmp.toBlob(r, "image/png"));
          const file = new File([blob], "frame.png", { type: "image/png" });

          // 2) Envoi au back
          const data = await recognize(file);
          drawOverlay(overlayRef.current, data, CAP_W, capH);
        } catch {
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

    // Ajuste le canvas à la taille affichée (gérée par CSS)
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
    const labels = faces ? faces.map(f => f.attributes ? `${f.attributes.age ?? "?"}y` : "Face")
                         : (data.names || []);

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

  return (
    <div className="app-container">

      <div className="header">
        <h1>Real-Time Face Recognition</h1>

        <div className={`status-bar ${status === "ok" ? "ok" : "down"}`}>
          {status === "ok" ? "All systems operational" : "System unavailable"}
        </div>
      </div>


      {/* Zone centrale */}
      <section className="main">

        <div className="main-container">
        {/* Sidebar gauche : compteur + vignettes */}
          <aside className="sidebar">
            <ProfilesPanel />
          </aside>

          <div className="live-container">
            <div className="live-indicator">
              <div className="dot" />
              <span>Live Detection</span>
            </div>

            <video ref={videoRef} autoPlay playsInline />
            <canvas ref={overlayRef} />
          </div>
        </div>

        <footer>Powered by a Custom AI Engine (React + Python)</footer>
      </section>
    </div>
  );
}
