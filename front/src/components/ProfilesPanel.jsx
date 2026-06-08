import { useEffect, useRef, useState } from "react";
import { addFace, deleteFace } from "../api";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function ProfilesPanel() {
  const [data, setData] = useState({ count: 0, profiles: [] });
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  async function load() {
    try {
      const res = await fetch(`${API}/profiles`);
      const json = await res.json();
      setData(json);
    } catch {
      console.error("Failed to load profiles");
    }
  }

  useEffect(() => { load(); }, []);

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  function resetForm() {
    setAdding(false);
    setName("");
    setFile(null);
    setPreview(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function handleAdd() {
    if (!name.trim()) { setError("Entrez un nom."); return; }
    if (!file) { setError("Choisissez une photo."); return; }
    setLoading(true);
    setError("");
    try {
      await addFace(name.trim(), file);
      await load();
      resetForm();
    } catch {
      setError("Erreur lors de l'ajout.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(profileName) {
    if (!confirm(`Supprimer "${profileName}" ?`)) return;
    try {
      await deleteFace(profileName);
      await load();
    } catch {
      console.error("Erreur suppression");
    }
  }

  return (
    <div className="profiles-panel">
      <h2>Profils reconnus <span className="count">{data.count}</span></h2>

      <div className="profiles">
        {data.profiles.map(p => (
          <div key={p.name} className="profile">
            <div className="profile-img-wrap">
              <img src={`${API}${p.url}`} alt={p.name} />
              <button
                className="delete-btn"
                onClick={() => handleDelete(p.name)}
                title={`Supprimer ${p.name}`}
              >×</button>
            </div>
            <div className="label">{p.name}</div>
          </div>
        ))}
      </div>

      {adding ? (
        <div className="add-form">
          <input
            type="text"
            placeholder="Nom de la personne"
            value={name}
            onChange={e => setName(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleAdd()}
          />
          <label className="file-label">
            {preview
              ? <img src={preview} className="preview-img" alt="preview" />
              : <span>+ Photo</span>
            }
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              hidden
            />
          </label>
          {error && <p className="form-error">{error}</p>}
          <div className="form-actions">
            <button className="btn-confirm" onClick={handleAdd} disabled={loading}>
              {loading ? "…" : "Ajouter"}
            </button>
            <button className="btn-cancel" onClick={resetForm}>Annuler</button>
          </div>
        </div>
      ) : (
        <button className="btn-add" onClick={() => setAdding(true)}>+ Ajouter une personne</button>
      )}
    </div>
  );
}
