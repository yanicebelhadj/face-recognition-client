import ProfilesPanel from "./components/ProfilesPanel";
import { useApiStatus } from "./hooks/useApiStatus";
import { useFaceProfiles } from "./hooks/useFaceProfiles";
import { useLiveRecognition } from "./hooks/useLiveRecognition";
import "./styles/app.scss";

const STATUS_LABELS = {
  ok: "Serveur opérationnel",
  waking: "Démarrage du serveur…",
  down: "Serveur injoignable",
};

export default function App() {
  const { status, retry } = useApiStatus();
  const { profiles, loading, syncState, error, add, remove, clearAll, syncToServer } = useFaceProfiles();

  const { videoRef, canvasRef, cameraError, cameraReady, latencyMs, faceCount } = useLiveRecognition({
    active: status === "ok",
    expectedFaces: profiles.length,
    onDrift: syncToServer,
  });

  return (
    <div className="app-container">
      <div className="header">
        <h1>Reconnaissance faciale en temps réel</h1>

        <div className={`status-bar ${status}`}>
          <span>{STATUS_LABELS[status]}</span>
          {status === "down" && (
            <button type="button" className="btn-retry" onClick={retry}>
              Réessayer
            </button>
          )}
        </div>
      </div>

      {status === "waking" && (
        <p className="banner info">
          L&apos;API est hébergée sur une offre gratuite : elle se met en veille après quelques minutes
          d&apos;inactivité. Le premier démarrage prend jusqu&apos;à une minute.
        </p>
      )}
      {status === "down" && (
        <p className="banner error">
          Impossible de joindre l&apos;API. Vos visages restent enregistrés sur cet appareil : la
          reconnaissance reprendra dès que le serveur répondra.
        </p>
      )}
      {error && <p className="banner error">{error}</p>}

      <section className="main">
        <div className="main-container">
          <aside className="sidebar">
            <ProfilesPanel
              profiles={profiles}
              loading={loading}
              onAdd={add}
              onRemove={remove}
              onClearAll={clearAll}
            />
          </aside>

          <div className="live-container">
            <div className="live-indicator">
              <div className={`dot ${status === "ok" && cameraReady ? "live" : "idle"}`} />
              <span>
                Détection en direct
                {faceCount > 0 && ` · ${faceCount} visage${faceCount > 1 ? "s" : ""}`}
                {latencyMs != null && ` · ${latencyMs} ms`}
                {syncState === "syncing" && " · synchronisation…"}
              </span>
            </div>

            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} />

            {cameraError && <p className="camera-error">{cameraError}</p>}
          </div>
        </div>

        <footer>
          Propulsé par dlib · React + Python — les photos ne quittent jamais votre navigateur
        </footer>
      </section>
    </div>
  );
}
