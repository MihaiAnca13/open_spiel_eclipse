import type { PlayerScoreBreakdown } from '../../types/game';

interface ScoreBreakdownProps {
  score: PlayerScoreBreakdown;
  className?: string;
}

const SCORE_ROWS: { label: string; key: keyof PlayerScoreBreakdown }[] = [
  { label: 'Reputation', key: 'reputation_vp' },
  { label: 'Ambassadors', key: 'ambassador_vp' },
  { label: 'Sectors', key: 'sector_vp' },
  { label: 'Monoliths', key: 'monolith_vp' },
  { label: 'Discoveries', key: 'discovery_vp' },
  { label: 'Tech tracks', key: 'tech_track_vp' },
  { label: 'Traitor', key: 'traitor_vp' },
  { label: 'Species', key: 'species_vp' },
  { label: 'Minor Species', key: 'minor_species_vp' },
];

function signedPoints(value: number) {
  return value > 0 ? `+${value}` : String(value);
}

export default function ScoreBreakdown({ score, className }: ScoreBreakdownProps) {
  return (
    <div className={className}>
      {SCORE_ROWS.map(({ label, key }) => (
        <div className="score-breakdown-row" key={key}>
          <span>{label}:</span>
          <span>{signedPoints(score[key])}</span>
        </div>
      ))}
      <div className="score-breakdown-row score-breakdown-total">
        <span>Total VP:</span>
        <span>{score.total_vp}</span>
      </div>
    </div>
  );
}
