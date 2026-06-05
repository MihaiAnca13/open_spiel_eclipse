import { useState, useEffect } from 'react';
import { API_BASE, SECTOR_ASSETS_BASE } from '../types/lobby';

export function useSectorAssets(): Record<number, string> {
  const [sectorImages, setSectorImages] = useState<Record<number, string>>({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/sectors/manifest`);
        const manifest: Record<string, string> = await res.json();
        if (cancelled) return;
        const urls: Record<number, string> = {};
        for (const [id, filename] of Object.entries(manifest)) {
          urls[Number(id)] = `${SECTOR_ASSETS_BASE}/${filename}`;
        }
        setSectorImages(urls);
      } catch {
        /* keep empty map -> colored-polygon fallback */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return sectorImages;
}
