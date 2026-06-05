import { useState } from 'react';
import { POP_TRACK_LABELS } from '../../actionTypes';

interface ColonyShipsProps {
  total: number;
  available: number;
  legalPlacements: { actionId: number; sectorId: number; slotIdx: number; track: number }[];
  onPlace: (actionId: number) => void;
}

export default function ColonyShips({
  total,
  available,
  legalPlacements,
  onPlace,
}: ColonyShipsProps) {
  const [expanded, setExpanded] = useState(false);

  if (total === 0) return null;

  return (
    <div className="colony-ships">
      <div
        className={`colony-ships-header ${legalPlacements.length > 0 ? 'clickable' : ''}`}
        onClick={() => legalPlacements.length > 0 && setExpanded(e => !e)}
        title="Colony Ships"
      >
        <span className="colony-ships-label">Colony Ships</span>
        <span className="colony-ships-icons">
          {Array.from({ length: total }, (_, i) => (
            <span
              key={i}
              className={`colony-ship-icon ${i < available ? 'available' : 'used'}`}
              title={i < available ? 'Available (faceup)' : 'Used (facedown)'}
            >
              ◎
            </span>
          ))}
        </span>
        {legalPlacements.length > 0 && (
          <span className="colony-ship-badge">{legalPlacements.length} placements</span>
        )}
      </div>
      {expanded && legalPlacements.length > 0 && (
        <div className="colony-ship-targets">
          {legalPlacements.map(({ actionId, sectorId, slotIdx, track }) => (
            <button
              key={actionId}
              className="colony-ship-target-btn"
              onClick={() => { onPlace(actionId); setExpanded(false); }}
              title={`Place a ${POP_TRACK_LABELS[track]} cube`}
            >
              Sector {sectorId} · slot {slotIdx} · {POP_TRACK_LABELS[track]}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
