import { useState, useEffect } from 'react';
import { API_BASE } from '../types/lobby';
import type { SectorLayout } from '../types/game';

export function useSectorLayouts(): Record<number, SectorLayout> {
  const [sectorLayouts, setSectorLayouts] = useState<Record<number, SectorLayout>>({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/sectors/layouts`);
        const raw: Record<string, SectorLayout> = await res.json();
        if (cancelled) return;
        setSectorLayouts(
          Object.fromEntries(Object.entries(raw).map(([k, v]) => [Number(k), v]))
        );
      } catch { /* non-fatal */ }
    })();
    return () => { cancelled = true; };
  }, []);

  return sectorLayouts;
}
