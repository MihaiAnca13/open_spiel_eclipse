import { INFLUENCE_TOTAL, INFLUENCE_UPKEEP } from '../../utils/game';

interface InfluenceTrackProps {
  onSectors: number;
  onActions: number;
}

export default function InfluenceTrack({ onSectors, onActions }: InfluenceTrackProps) {
  const deployed = onSectors + onActions;
  const available = Math.max(0, INFLUENCE_TOTAL - deployed);
  const upkeep = INFLUENCE_UPKEEP[Math.min(deployed, INFLUENCE_UPKEEP.length - 1)];

  // Discs fill the track from the left; spent discs leave gaps and expose the
  // upkeep cost. Order: available (on track) → on actions → on sectors.
  const cells = Array.from({ length: INFLUENCE_TOTAL }, (_, i) => {
    if (i < available) return 'avail';
    if (i < available + onActions) return 'action';
    return 'sector';
  });

  return (
    <div className="influence-track">
      <div className="influence-track-head">
        <span>Influence Track</span>
        <span className="influence-upkeep" title="Money paid each Upkeep phase">Upkeep 💰{upkeep}</span>
      </div>
      <div className="influence-pips">
        {cells.map((kind, i) => (
          <span key={i} className={`influence-pip ${kind}`} />
        ))}
      </div>
      <div className="influence-legend">
        <span><span className="influence-pip avail" /> {available} ready</span>
        <span><span className="influence-pip action" /> {onActions} actions</span>
        <span><span className="influence-pip sector" /> {onSectors} sectors</span>
      </div>
    </div>
  );
}
