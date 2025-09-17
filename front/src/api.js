const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function ping() {
  const r = await fetch(`${API_URL}/ping`);
  return r.json();
}
export async function health() {
  const r = await fetch(`${API_URL}/health`).catch(() => null);
  return r && r.ok ? r.json() : { known_faces_count: "?", names: [] };
}
export async function recognize(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${API_URL}/recognize?attributes=true`, { method: "POST", body: form });
  return r.json();
}
export async function recognizeDraw(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch(`${API_URL}/recognize/draw`, { method: "POST", body: form });
  const blob = await r.blob();
  return URL.createObjectURL(blob);
}
