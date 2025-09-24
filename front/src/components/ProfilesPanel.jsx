import { useEffect, useState } from "react";
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function ProfilesPanel() {
  const [data, setData] = useState({ count: 0, profiles: [] });

  async function load() {
    try {
      const res = await fetch(`${API}/profiles`);
      const json = await res.json();
      setData(json);
    } catch (e) {
      console.error("Failed to load profiles:", e);
    }
  }

  useEffect(() => { load(); }, []);

  return (
    <div style={{ padding: 16, borderRadius: 12, background: "#f7f7f8", marginBottom: 16 }}>
      <h2 style={{ margin: 0, marginBottom: 12 }}>Profiles in Memory: {data.count}</h2>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        {data.profiles.map(p => (
          <div key={p.name} title={p.name} style={{ textAlign: "center" }}>
            <img
              src={`${API}${p.url}`}
              alt={p.name}
              width={56}
              height={56}
              style={{ objectFit: "cover", borderRadius: 8, border: "1px solid #ddd" }}
            />
            <div style={{ fontSize: 12, marginTop: 4 }}>{p.name}</div>
          </div>
        ))}
      </div>
      <button onClick={load} style={{ marginTop: 12, padding: "6px 10px", borderRadius: 8, border: "1px solid #ccc", background: "white" }}>
        Refresh
      </button>
    </div>
  );
}
