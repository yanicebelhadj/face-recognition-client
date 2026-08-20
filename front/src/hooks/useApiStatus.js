/**
 * État de l'API, avec prise en compte du réveil de l'instance gratuite.
 *
 * Sur le plan gratuit de Render, le serveur s'endort après ~15 min sans trafic
 * et met 30 à 60 s à redémarrer. Le premier appel est donc *lent*, pas *cassé* :
 * on affiche « démarrage du serveur » pendant que `ping()` réessaie, au lieu de
 * l'ancien « System unavailable » qui donnait l'impression d'une panne.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { ping } from "../api";

const RECHECK_INTERVAL_MS = 60000;

export function useApiStatus() {
  const [status, setStatus] = useState("waking"); // waking | ok | down
  const [modelsReady, setModelsReady] = useState(false);
  const mounted = useRef(true);

  const check = useCallback(async ({ patient = true } = {}) => {
    if (mounted.current) setStatus((current) => (current === "ok" ? current : "waking"));
    try {
      const payload = await ping({ retries: patient ? 4 : 1 });
      if (!mounted.current) return true;
      setStatus("ok");
      setModelsReady(Boolean(payload.models_ready));
      return true;
    } catch {
      if (mounted.current) setStatus("down");
      return false;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    check();
    const timer = setInterval(() => check({ patient: false }), RECHECK_INTERVAL_MS);
    return () => {
      mounted.current = false;
      clearInterval(timer);
    };
  }, [check]);

  return { status, modelsReady, retry: () => check() };
}
