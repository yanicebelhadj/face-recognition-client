/**
 * Gestion des profils : stockage local + synchronisation avec l'API.
 *
 * Répartition des rôles :
 *   - le navigateur détient les photos et les empreintes (cf. `lib/faceStore`) ;
 *   - l'API ne détient qu'une copie en RAM des empreintes de *cette* session,
 *     copie qu'elle perd à chaque mise en veille de l'offre gratuite.
 *
 * D'où `syncToServer()` : dès qu'un écart est constaté entre le contenu local
 * et ce que l'API dit connaître, on rejoue les empreintes. C'est ce qui donne
 * une persistance « d'une session à l'autre » sans le moindre stockage serveur.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, deleteFace, destroySession, enrollFace, fetchProfiles, restoreFaces } from "../api";
import {
  clearProfiles,
  deleteProfile,
  isPersistent,
  listProfiles,
  saveProfile,
} from "../lib/faceStore";
import { createThumbnail } from "../lib/image";
import { resetSessionId } from "../lib/session";

/** Délai minimal entre deux tentatives de synchronisation. */
const SYNC_COOLDOWN_MS = 5000;

/** Payload de restauration : ni photo, ni donnée superflue. */
function toRestorePayload(profiles) {
  return profiles.map(({ id, name, embedding }) => ({ id, name, embedding }));
}

export function useFaceProfiles() {
  const [profiles, setProfiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [syncState, setSyncState] = useState("idle"); // idle | syncing | synced | error
  const [error, setError] = useState("");
  const [storageWarning, setStorageWarning] = useState("");

  // Miroir des profils, pour que la synchro n'ait pas à figurer dans les
  // dépendances des callbacks (et ne les recrée pas à chaque rendu).
  const profilesRef = useRef([]);
  const syncingRef = useRef(null);
  const lastSyncRef = useRef(0);
  const objectUrls = useRef(new Set());

  const revokeUrl = useCallback((url) => {
    if (!url || !url.startsWith("blob:")) return;
    URL.revokeObjectURL(url);
    objectUrls.current.delete(url);
  }, []);

  const publish = useCallback(
    (next) => {
      // Toute vignette qui disparaît de la liste voit son URL libérée.
      const kept = new Set(next.map((profile) => profile.photoUrl));
      profilesRef.current.forEach((profile) => {
        if (profile.photoUrl && !kept.has(profile.photoUrl)) revokeUrl(profile.photoUrl);
      });
      profilesRef.current = next;
      setProfiles(next);
    },
    [revokeUrl],
  );

  /**
   * URL d'affichage d'une vignette : la photo complète si le stockage a pu la
   * garder, sinon la vignette compacte du repli `localStorage`.
   */
  const toDisplayable = useCallback((profile) => {
    if (profile.photo) {
      const photoUrl = URL.createObjectURL(profile.photo);
      objectUrls.current.add(photoUrl);
      return { ...profile, photoUrl };
    }
    return { ...profile, photoUrl: profile.thumbnail || null };
  }, []);

  const noteStorageLimits = useCallback(async () => {
    setStorageWarning(
      (await isPersistent())
        ? ""
        : "Ce navigateur n'autorise pas le stockage local (navigation privée ou navigateur intégré) : " +
            "vos visages seront perdus à la fermeture de l'onglet.",
    );
  }, []);

  /**
   * Aligne l'API sur le contenu local. Idempotent : les appels concurrents
   * partagent la même promesse.
   *
   * @param {object} options
   * @param {boolean} options.force  ignore le palier anti-rafale
   */
  const syncToServer = useCallback(async ({ force = false } = {}) => {
    if (syncingRef.current) return syncingRef.current;
    // La boucle vidéo signale l'écart à chaque frame : sans ce palier, un
    // serveur qui refuse la restauration serait sollicité dix fois par seconde.
    if (!force && Date.now() - lastSyncRef.current < SYNC_COOLDOWN_MS) return false;
    lastSyncRef.current = Date.now();

    const run = (async () => {
      const local = profilesRef.current;
      setSyncState("syncing");
      try {
        const remote = await fetchProfiles();
        const remoteIds = new Set((remote.profiles || []).map((p) => p.id));
        const aligned = remoteIds.size === local.length && local.every((p) => remoteIds.has(p.id));

        if (!aligned) {
          await restoreFaces(toRestorePayload(local));
        }
        setSyncState("synced");
        setError("");
        return true;
      } catch (cause) {
        setSyncState("error");
        // Sans profil local, il n'y a rien à resynchroniser : inutile d'alarmer
        // l'utilisateur pour une opération qui ne lui apporte rien.
        setError(
          local.length === 0
            ? ""
            : cause instanceof ApiError && cause.status
              ? `Synchronisation impossible : ${cause.message}`
              : "Serveur injoignable : la reconnaissance reprendra automatiquement.",
        );
        return false;
      } finally {
        syncingRef.current = null;
      }
    })();

    syncingRef.current = run;
    return run;
  }, []);

  // Chargement initial depuis le stockage local.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stored = await listProfiles();
        if (!cancelled) publish(stored.map(toDisplayable));
      } catch {
        if (!cancelled) {
          setStorageWarning("Stockage local indisponible : les profils ne seront pas conservés.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
      if (!cancelled) await noteStorageLimits();
    })();

    return () => {
      cancelled = true;
    };
  }, [noteStorageLimits, publish, toDisplayable]);

  // Libération des URLs d'objets au démontage.
  useEffect(
    () => () => {
      objectUrls.current.forEach(URL.revokeObjectURL);
      objectUrls.current.clear();
    },
    [],
  );

  const add = useCallback(
    async (rawName, file) => {
      const name = rawName.trim();
      const { profile } = await enrollFace(name, file);

      // La vignette sert au repli `localStorage`, qui ne peut pas garder la
      // photo d'origine. On la calcule ici, une fois pour toutes.
      const record = {
        id: profile.id,
        name: profile.name,
        embedding: profile.embedding,
        photo: file,
        thumbnail: await createThumbnail(file),
        createdAt: Date.now(),
      };

      // Un nom = un profil : l'API remplace, le stockage local doit suivre.
      const duplicates = profilesRef.current.filter(
        (p) => p.name.toLowerCase() === record.name.toLowerCase(),
      );
      const kept = profilesRef.current.filter(
        (p) => p.name.toLowerCase() !== record.name.toLowerCase(),
      );

      // Le visage est déjà enrôlé côté API : un stockage local défaillant ne
      // doit pas faire échouer l'ajout, seulement priver l'utilisateur de la
      // persistance. C'est le cas des navigateurs intégrés (WhatsApp, iOS).
      try {
        await Promise.all(duplicates.map((p) => deleteProfile(p.id)));
        await saveProfile(record);
      } catch {
        setStorageWarning(
          "Ce navigateur refuse d'enregistrer vos photos : elles resteront actives " +
            "le temps de la visite, mais seront perdues à la fermeture de l'onglet.",
        );
      }
      await noteStorageLimits();

      publish([...kept, toDisplayable(record)]);
      setSyncState("synced");
      return record;
    },
    [noteStorageLimits, publish, toDisplayable],
  );

  const remove = useCallback(
    async (id) => {
      try {
        await deleteFace(id);
      } catch (cause) {
        // 404 : l'API a redémarré et ne connaît plus ce profil. Le supprimer
        // localement reste la bonne action.
        if (!(cause instanceof ApiError && cause.status === 404)) throw cause;
      }
      try {
        await deleteProfile(id);
      } catch {
        /* le stockage local est déjà signalé comme défaillant */
      }
      publish(profilesRef.current.filter((p) => p.id !== id));
    },
    [publish],
  );

  /** Efface tout, des deux côtés, et repart sur une session neuve. */
  const clearAll = useCallback(async () => {
    try {
      await destroySession();
    } catch {
      /* l'API a peut-être déjà oublié la session : sans importance */
    }
    try {
      await clearProfiles();
    } catch {
      /* idem côté stockage local */
    }
    resetSessionId();
    publish([]);
    setSyncState("synced");
    setError("");
  }, [publish]);

  return {
    profiles,
    loading,
    syncState,
    error,
    storageWarning,
    add,
    remove,
    clearAll,
    syncToServer,
  };
}
