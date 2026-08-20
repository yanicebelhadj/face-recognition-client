/**
 * Client HTTP de l'API de reconnaissance faciale.
 *
 * Deux responsabilités en plus du simple `fetch` :
 *
 * 1. **Cloisonnement** — toute requête porte l'en-tête `X-Session-Id`. Sans lui,
 *    l'API refuse l'accès aux profils : c'est ce qui garantit qu'un poste ne
 *    voit jamais les visages d'un autre.
 * 2. **Réveil de l'instance gratuite** — l'API s'endort après ~15 min
 *    d'inactivité et met 30 à 60 s à redémarrer. Les erreurs réseau et les 503
 *    sont donc réessayées avec un délai croissant au lieu d'être présentées
 *    comme une panne.
 */
import { getSessionId } from "./lib/session";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(message, { status = 0, retriable = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retriable = retriable;
  }
}

const RETRIABLE_STATUSES = new Set([502, 503, 504]);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function readError(response) {
  try {
    const body = await response.json();
    if (body?.detail) return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* réponse non-JSON (page d'erreur de l'hébergeur, par exemple) */
  }
  return `Erreur ${response.status}`;
}

/**
 * @param {string} path
 * @param {object} options
 * @param {number} options.retries  nombre de tentatives supplémentaires
 * @param {number} options.timeout  délai max par tentative (ms)
 */
export async function request(path, { retries = 0, timeout = 20000, ...init } = {}) {
  let lastError = new ApiError("Requête non exécutée.");

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(`${API_URL}${path}`, {
        ...init,
        signal: controller.signal,
        headers: { "X-Session-Id": getSessionId(), ...(init.headers || {}) },
      });

      if (response.ok) return response;

      const message = await readError(response);
      lastError = new ApiError(message, {
        status: response.status,
        retriable: RETRIABLE_STATUSES.has(response.status),
      });
      if (!lastError.retriable) throw lastError;
    } catch (error) {
      if (error instanceof ApiError) {
        if (!error.retriable) throw error;
      } else {
        // Réseau coupé, DNS, CORS, ou instance en cours de réveil.
        lastError = new ApiError("Serveur injoignable.", { status: 0, retriable: true });
      }
    } finally {
      clearTimeout(timer);
    }

    if (attempt < retries) {
      await sleep(Math.min(1000 * 2 ** attempt, 8000));
    }
  }

  throw lastError;
}

async function json(path, options) {
  return (await request(path, options)).json();
}

/* --------------------------------------------------------------------- */
/* Santé                                                                  */
/* --------------------------------------------------------------------- */

/** Sonde le serveur en tolérant un long réveil (offre gratuite). */
export function ping({ retries = 4 } = {}) {
  return json("/ping", { retries, timeout: 25000 });
}

export function health() {
  return json("/health", { retries: 1 });
}

/* --------------------------------------------------------------------- */
/* Profils — toujours dans le périmètre de la session courante            */
/* --------------------------------------------------------------------- */
export function fetchProfiles() {
  return json("/api/profiles", { retries: 2 });
}

/** Envoie une photo, récupère l'empreinte 128-D. La photo n'est pas stockée. */
export function enrollFace(name, file) {
  const form = new FormData();
  form.append("name", name);
  form.append("file", file);
  return json("/api/faces", { method: "POST", body: form, retries: 1, timeout: 45000 });
}

/** Rejoue les profils gardés localement (après un redémarrage de l'API). */
export function restoreFaces(profiles) {
  return json("/api/faces/restore", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profiles }),
    retries: 2,
  });
}

export function deleteFace(id) {
  return json(`/api/faces/${encodeURIComponent(id)}`, { method: "DELETE", retries: 1 });
}

export function destroySession() {
  return json("/api/session", { method: "DELETE", retries: 1 });
}

/* --------------------------------------------------------------------- */
/* Reconnaissance                                                         */
/* --------------------------------------------------------------------- */

/**
 * Analyse une frame. Pas de réessai : en direct, mieux vaut abandonner
 * l'image courante et envoyer la suivante.
 */
export function recognize(file, { timeout = 15000 } = {}) {
  const form = new FormData();
  form.append("file", file);
  return json("/api/recognize", { method: "POST", body: form, timeout });
}

export { API_URL };
