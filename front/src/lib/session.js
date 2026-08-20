/**
 * Identifiant de session — la clé du cloisonnement.
 *
 * Il est tiré au sort dans le navigateur, conservé dans localStorage, et envoyé
 * à l'API dans l'en-tête `X-Session-Id`. Le serveur ne range les visages que
 * sous cet identifiant : deux postes différents ne peuvent donc pas voir les
 * profils l'un de l'autre, et le même poste retrouve les siens à la visite
 * suivante.
 *
 * C'est un secret : quiconque le connaît accède aux visages de la session. Il
 * ne transite que vers l'API, jamais dans une URL.
 */
const STORAGE_KEY = "face-recognition.session-id";

/** 128 bits en hexadécimal — le format exigé par l'API. */
function generateId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID().replace(/-/g, "");
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function isValid(value) {
  return typeof value === "string" && /^[0-9a-f]{32}$/.test(value);
}

let cached = null;

export function getSessionId() {
  if (cached) return cached;

  let stored = null;
  try {
    stored = localStorage.getItem(STORAGE_KEY);
  } catch {
    // Navigation privée ou stockage bloqué : on repart sur une session
    // éphémère, valable le temps de l'onglet.
  }

  cached = isValid(stored) ? stored : generateId();

  if (cached !== stored) {
    try {
      localStorage.setItem(STORAGE_KEY, cached);
    } catch {
      /* stockage indisponible — la session ne survivra pas au rechargement */
    }
  }
  return cached;
}

/** Repart de zéro : nouvelle session, donc plus aucun visage associé. */
export function resetSessionId() {
  cached = generateId();
  try {
    localStorage.setItem(STORAGE_KEY, cached);
  } catch {
    /* ignore */
  }
  return cached;
}
