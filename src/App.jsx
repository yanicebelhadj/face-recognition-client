// src/App.jsx
import FaceDetect from "./components/FaceDetect";

export default function App() {
  return (
    <div style={{ padding: 16 }}>
      <h1>Face Recognition Client (React)</h1>
      <p>Démo détection locale (webcam/image). Rien n'est envoyé côté serveur.</p>
      <FaceDetect />
    </div>
  );
}
