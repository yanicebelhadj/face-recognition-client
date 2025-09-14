// src/components/FaceDetect.jsx
import { useEffect, useRef, useState } from "react";
import { FaceDetection } from "@mediapipe/face_detection";
import { Camera } from "@mediapipe/camera_utils";

export default function FaceDetect() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const [fd, setFd] = useState(null);
  const [usingWebcam, setUsingWebcam] = useState(false);

  // Init MediaPipe FaceDetection
  useEffect(() => {
    const face = new FaceDetection({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
    });
    face.setOptions({
      model: "short",       // rapide
      minDetectionConfidence: 0.6,
    });
    face.onResults(onResults);
    setFd(face);
  }, []);

  const onResults = (results) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const src = usingWebcam ? videoRef.current : imgRef.current;
    const w = src.videoWidth || src.naturalWidth || 0;
    const h = src.videoHeight || src.naturalHeight || 0;
    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    if (!results.detections) return;
    ctx.lineWidth = 2;
    ctx.strokeStyle = "lime";
    results.detections.forEach((d) => {
      const bb = d.locationData.relativeBoundingBox;
      const x = bb.xMin * w;
      const y = bb.yMin * h;
      const ww = bb.width * w;
      const hh = bb.height * h;
      ctx.strokeRect(x, y, ww, hh);
    });
  };

  // Webcam
  const startWebcam = async () => {
    if (!fd) return;
    const cam = new Camera(videoRef.current, {
      onFrame: async () => {
        await fd.send({ image: videoRef.current });
      },
      width: 640,
      height: 480,
    });
    await cam.start();
    setUsingWebcam(true);
  };
  const stopWebcam = () => {
    const stream = videoRef.current?.srcObject;
    stream && stream.getTracks().forEach((t) => t.stop());
    videoRef.current.srcObject = null;
    setUsingWebcam(false);
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  // Image upload
  const onPick = (e) => {
    const f = e.target.files?.[0];
    if (!f || !fd) return;
    const url = URL.createObjectURL(f);
    const img = imgRef.current;
    img.onload = async () => {
      await fd.send({ image: img });
    };
    img.src = url;
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
        <input type="file" accept="image/*" onChange={onPick} />
        <button onClick={startWebcam} disabled={!fd || usingWebcam}>Activer webcam</button>
        <button onClick={stopWebcam} disabled={!usingWebcam}>Stop</button>
      </div>

      <div style={{ position: "relative", display: "inline-block" }}>
        <img ref={imgRef} alt="" style={{ width: 640, display: usingWebcam ? "none" : "block" }} />
        <video ref={videoRef} autoPlay playsInline muted width={640}
               style={{ display: usingWebcam ? "block" : "none" }} />
        <canvas ref={canvasRef} style={{ position: "absolute", left: 0, top: 0 }} />
      </div>

      <p style={{ fontSize: 12, opacity: 0.7, marginTop: 6 }}>
        Traitement local uniquement. Active la webcam avec consentement explicite.
      </p>
    </div>
  );
}
