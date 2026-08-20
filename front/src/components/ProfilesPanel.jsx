/**
 * Panneau des profils de la session courante.
 *
 * Les vignettes sont rendues depuis les photos stockées *localement* : aucune
 * image n'est demandée au serveur, qui d'ailleurs n'en conserve aucune.
 */
import { useRef, useState } from "react";

export default function ProfilesPanel({ profiles, loading, onAdd, onRemove, onClearAll }) {
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState("");
  const inputRef = useRef(null);

  function selectFile(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setPreview((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(selected);
    });
    setFormError("");
  }

  function resetForm() {
    setAdding(false);
    setName("");
    setFile(null);
    setPreview((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return null;
    });
    setFormError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function submit() {
    if (!name.trim()) return setFormError("Entrez un nom.");
    if (!file) return setFormError("Choisissez une photo.");

    setBusy(true);
    setFormError("");
    try {
      await onAdd(name, file);
      resetForm();
    } catch (error) {
      setFormError(error?.message || "Impossible d'ajouter ce profil.");
    } finally {
      setBusy(false);
    }
    return undefined;
  }

  async function remove(profile) {
    if (!confirm(`Supprimer « ${profile.name} » ?`)) return;
    try {
      await onRemove(profile.id);
    } catch {
      setFormError("Suppression impossible pour le moment.");
    }
  }

  async function clearAll() {
    if (!confirm("Effacer tous vos visages ? Cette action est définitive.")) return;
    await onClearAll();
  }

  return (
    <div className="profiles-panel">
      <h2>
        Vos profils <span className="count">{profiles.length}</span>
      </h2>

      <p className="privacy-note">
        Vos photos restent sur cet appareil et ne sont visibles que dans cette session.
      </p>

      {loading ? (
        <p className="muted">Chargement…</p>
      ) : (
        <div className="profiles">
          {profiles.map((profile) => (
            <div key={profile.id} className="profile">
              <div className="profile-img-wrap">
                {profile.photoUrl ? (
                  <img src={profile.photoUrl} alt={profile.name} />
                ) : (
                  <div className="profile-img-fallback">{profile.name.slice(0, 1).toUpperCase()}</div>
                )}
                <button
                  type="button"
                  className="delete-btn"
                  onClick={() => remove(profile)}
                  title={`Supprimer ${profile.name}`}
                >
                  ×
                </button>
              </div>
              <div className="label">{profile.name}</div>
            </div>
          ))}
          {profiles.length === 0 && <p className="muted">Aucun visage enregistré pour l'instant.</p>}
        </div>
      )}

      {adding ? (
        <div className="add-form">
          <input
            type="text"
            placeholder="Nom de la personne"
            value={name}
            onChange={(event) => setName(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && submit()}
          />
          <label className="file-label">
            {preview ? <img src={preview} className="preview-img" alt="Aperçu" /> : <span>+ Photo</span>}
            <input ref={inputRef} type="file" accept="image/*" onChange={selectFile} hidden />
          </label>
          {formError && <p className="form-error">{formError}</p>}
          <div className="form-actions">
            <button type="button" className="btn-confirm" onClick={submit} disabled={busy}>
              {busy ? "Analyse…" : "Ajouter"}
            </button>
            <button type="button" className="btn-cancel" onClick={resetForm} disabled={busy}>
              Annuler
            </button>
          </div>
        </div>
      ) : (
        <>
          <button type="button" className="btn-add" onClick={() => setAdding(true)}>
            + Ajouter une personne
          </button>
          {formError && <p className="form-error">{formError}</p>}
          {profiles.length > 0 && (
            <button type="button" className="btn-clear" onClick={clearAll}>
              Tout effacer
            </button>
          )}
        </>
      )}
    </div>
  );
}
