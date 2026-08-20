/**
 * Stockage local des visages, avec repli progressif.
 *
 * C'est ici que vivent les photos : elles restent sur l'appareil et ne sont
 * envoyées à l'API que le temps de calculer leur empreinte.
 *
 * Trois implémentations, essayées dans cet ordre :
 *
 * 1. **IndexedDB** — le cas normal : photo complète, pas de limite gênante.
 * 2. **localStorage** — repli quand IndexedDB est cassé. C'est fréquent dans les
 *    navigateurs intégrés (WhatsApp, Instagram) et en navigation privée sur
 *    iOS : WebKit ferme la connexion sous les pieds de la transaction
 *    (« The database connection is closing »). On n'y range qu'une vignette,
 *    faute de quota pour la photo d'origine.
 * 3. **Mémoire** — dernier recours : la démonstration reste utilisable le temps
 *    de la visite, mais rien n'est conservé. `isPersistent()` permet de le dire
 *    honnêtement à l'utilisateur.
 *
 * Un enregistrement : { id, name, embedding, photo?, thumbnail?, createdAt }.
 * `photo` est un Blob (IndexedDB seulement), `thumbnail` une data URL.
 */
const DB_NAME = "face-recognition";
const DB_VERSION = 1;
const STORE = "profiles";
const LOCAL_STORAGE_KEY = "face-recognition.profiles";

/* ---------------------------------------------------------------------- */
/* Backend 1 — IndexedDB                                                   */
/* ---------------------------------------------------------------------- */
let dbPromise = null;

function isConnectionClosing(error) {
  // WebKit lève InvalidStateError avec ce message quand la connexion a été
  // fermée entre l'ouverture et la transaction. Rouvrir suffit à s'en sortir.
  return (
    error?.name === "InvalidStateError" ||
    /connection is closing|database connection/i.test(error?.message || "")
  );
}

function openDb() {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) {
      reject(new Error("IndexedDB indisponible."));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    request.onblocked = () => reject(new Error("Ouverture d'IndexedDB bloquée."));
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      // Si le navigateur ferme la connexion, on jette la promesse en cache pour
      // que la prochaine opération rouvre au lieu d'échouer.
      db.onclose = () => {
        dbPromise = null;
      };
      db.onversionchange = () => {
        db.close();
        dbPromise = null;
      };
      resolve(db);
    };
  }).catch((error) => {
    dbPromise = null;
    throw error;
  });

  return dbPromise;
}

function runOnce(mode, operation) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        let request;
        try {
          const transaction = db.transaction(STORE, mode);
          transaction.onabort = () => reject(transaction.error);
          transaction.onerror = () => reject(transaction.error);
          request = operation(transaction.objectStore(STORE));
          if (!request) {
            transaction.oncomplete = () => resolve();
            return;
          }
        } catch (error) {
          reject(error);
          return;
        }
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      }),
  );
}

async function idbRun(mode, operation) {
  try {
    return await runOnce(mode, operation);
  } catch (error) {
    if (!isConnectionClosing(error)) throw error;
    dbPromise = null; // une seconde tentative, sur une connexion neuve
    return runOnce(mode, operation);
  }
}

const indexedDbBackend = {
  name: "indexeddb",
  persistent: true,
  async list() {
    return (await idbRun("readonly", (store) => store.getAll())) || [];
  },
  save(profile) {
    return idbRun("readwrite", (store) => store.put(profile));
  },
  remove(id) {
    return idbRun("readwrite", (store) => store.delete(id));
  },
  clear() {
    return idbRun("readwrite", (store) => store.clear());
  },
};

/* ---------------------------------------------------------------------- */
/* Backend 2 — localStorage (vignettes seulement)                          */
/* ---------------------------------------------------------------------- */
function readLocalStorage() {
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeLocalStorage(records) {
  // La photo d'origine ne tient pas dans le quota : seule la vignette part.
  const serialisable = records.map((record) => ({
    id: record.id,
    name: record.name,
    embedding: record.embedding,
    thumbnail: record.thumbnail || null,
    createdAt: record.createdAt,
  }));
  localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(serialisable));
}

const localStorageBackend = {
  name: "localstorage",
  persistent: true,
  async list() {
    return readLocalStorage();
  },
  async save(profile) {
    const records = readLocalStorage().filter((record) => record.id !== profile.id);
    records.push(profile);
    writeLocalStorage(records);
  },
  async remove(id) {
    writeLocalStorage(readLocalStorage().filter((record) => record.id !== id));
  },
  async clear() {
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  },
};

/* ---------------------------------------------------------------------- */
/* Backend 3 — mémoire                                                     */
/* ---------------------------------------------------------------------- */
const memory = new Map();

const memoryBackend = {
  name: "memory",
  persistent: false,
  async list() {
    return [...memory.values()];
  },
  async save(profile) {
    memory.set(profile.id, profile);
  },
  async remove(id) {
    memory.delete(id);
  },
  async clear() {
    memory.clear();
  },
};

/* ---------------------------------------------------------------------- */
/* Sélection du backend                                                    */
/* ---------------------------------------------------------------------- */
const CHAIN = [indexedDbBackend, localStorageBackend, memoryBackend];

let active = null;

/** Retient le premier backend qui survit à une lecture réelle. */
async function backend() {
  if (active) return active;

  for (const candidate of CHAIN) {
    try {
      await candidate.list();
      active = candidate;
      if (candidate !== CHAIN[0]) {
        console.warn(`[faceStore] repli sur « ${candidate.name} » : IndexedDB inutilisable.`);
      }
      return active;
    } catch {
      /* candidat suivant */
    }
  }

  active = memoryBackend;
  return active;
}

/**
 * Rétrograde vers le backend suivant après un échec en écriture.
 * Renvoie le nouveau backend, ou `null` si on était déjà au dernier.
 */
async function degrade() {
  const index = CHAIN.indexOf(active);
  if (index < 0 || index >= CHAIN.length - 1) return null;
  active = CHAIN[index + 1];
  console.warn(`[faceStore] repli sur « ${active.name} » après un échec d'écriture.`);
  return active;
}

/** Exécute une écriture, en rétrogradant plutôt qu'en échouant. */
async function write(operation) {
  let current = await backend();
  for (;;) {
    try {
      return await operation(current);
    } catch (error) {
      const next = await degrade();
      if (!next) throw error;
      current = next;
    }
  }
}

export async function listProfiles() {
  const store = await backend();
  const records = await store.list();
  return records.sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
}

export function saveProfile(profile) {
  return write((store) => store.save(profile));
}

export function deleteProfile(id) {
  return write((store) => store.remove(id));
}

export function clearProfiles() {
  return write((store) => store.clear());
}

/** `false` quand les visages ne survivront pas à la fermeture de l'onglet. */
export async function isPersistent() {
  return (await backend()).persistent;
}

/** Nom du backend retenu — utile en débogage. */
export async function storageBackendName() {
  return (await backend()).name;
}
