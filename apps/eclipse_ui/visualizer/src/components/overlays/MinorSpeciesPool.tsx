import { useState } from 'react';
import type { MinorSpeciesDef } from '../../types/game';
import { minorSpeciesImageUrl } from '../../types/lobby';
import { minorSpeciesEffectText } from '../../utils/minorSpecies';

interface MinorSpeciesPoolProps {
  pool: number[];
  catalog: Record<string, MinorSpeciesDef>;
}

export default function MinorSpeciesPool({ pool, catalog }: MinorSpeciesPoolProps) {
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const selected = selectedId === null ? undefined : catalog[String(selectedId)];

  if (pool.length === 0) return null;

  return (
    <>
      <section className="minor-species-pool" aria-label="Available Minor Species">
        <h2>Minor Species</h2>
        <p>Available relations</p>
        <div className="minor-species-pool-grid">
          {pool.map((id) => {
            const species = catalog[String(id)];
            if (!species) return null;
            return (
              <button
                type="button"
                className="minor-species-pool-card"
                key={id}
                onClick={() => setSelectedId(id)}
                title={`View ${species.name}`}
              >
                <img src={minorSpeciesImageUrl(id + 1)} alt="" />
                <span>{species.name}</span>
                <small>{species.cost} 💰</small>
              </button>
            );
          })}
        </div>
      </section>

      {selected && selectedId !== null && (
        <div className="minor-species-modal-overlay" onClick={() => setSelectedId(null)}>
          <section className="minor-species-modal" role="dialog" aria-modal="true" aria-label={selected.name} onClick={(event) => event.stopPropagation()}>
            <button type="button" className="minor-species-modal-close" onClick={() => setSelectedId(null)} aria-label="Close">×</button>
            <img src={minorSpeciesImageUrl(selectedId + 1)} alt="" />
            <h2>{selected.name}</h2>
            <p className="minor-species-cost">Form relations: {selected.cost} Money</p>
            <p>{minorSpeciesEffectText(selected) || 'No additional effect.'}</p>
          </section>
        </div>
      )}
    </>
  );
}
