/**
 * Stockage local des visages (IndexedDB).
 *
 * C'est *ici* que vivent les photos : elles restent sur le poste de
 * l'utilisateur et ne sont envoyées à l'API que le temps de calculer leur
 * empreinte. Le navigateur est donc la source de vérité, et l'API un simple
 * cache de calcul.
 *
 * Chaque enregistrement contient :
 *   id        identifiant du profil, partagé avec l'API
 *   name      libellé affiché
 *   embedding vecteur 128-D renvoyé par l'API
 *   photo     Blob de la photo d'origine (affichage des vignettes)
 *   createdAt horodatage
 *
 * Conserver l'embedding permet de restaurer la session côté serveur sans
 * réenvoyer ni recalculer les photos après une mise en veille de l'API.
 */
const DB_NAME = "face-recognition";
const DB_VERSION = 1;
const STORE = "profiles";

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) {
      reject(new Error("IndexedDB indisponible dans ce navigateur."));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  }).catch((error) => {
    dbPromise = null; // laisse une chance à un prochain essai
    throw error;
  });

  return dbPromise;
}

function run(mode, operation) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const transaction = db.transaction(STORE, mode);
        const request = operation(transaction.objectStore(STORE));
        transaction.onabort = () => reject(transaction.error);
        transaction.onerror = () => reject(transaction.error);
        if (request) {
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => reject(request.error);
        } else {
          transaction.oncomplete = () => resolve();
        }
      }),
  );
}

/** Tous les profils, dans leur ordre d'ajout. */
export async function listProfiles() {
  const records = (await run("readonly", (store) => store.getAll())) || [];
  return records.sort((a, b) => a.createdAt - b.createdAt);
}

export function saveProfile(profile) {
  return run("readwrite", (store) => store.put(profile));
}

export function deleteProfile(id) {
  return run("readwrite", (store) => store.delete(id));
}

export function clearProfiles() {
  return run("readwrite", (store) => store.clear());
}
