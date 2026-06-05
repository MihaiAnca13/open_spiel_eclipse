import type { Resources } from '../../types/game';
import {
  POPULATION_PRODUCTION_TABLE,
  POP_TRACK_MAX,
  POP_TRACK_STEPS,
} from '../../utils/game';

interface PopulationTracksProps {
  resources: Resources;
  playerColor: string;
}

export default function PopulationTracks({ resources, playerColor }: PopulationTracksProps) {
  const tracks = [
    { label: 'Mat', color: '#f97316', cubesOnTrack: resources.materials_prod },
    { label: 'Sci', color: '#818cf8', cubesOnTrack: resources.science_prod },
    { label: '$',   color: '#fbbf24', cubesOnTrack: resources.gold_prod },
  ];

  const W = 120;
  const CX = W / 2;
  const CY = W / 2 + 4;
  const OUTER_R = 46;
  const INNER_R = 30;
  const DOT_R = 3.2;
  // Arc centers (degrees): Materials=210, Science=90, Money=330 (=−30)
  const ARC_SPAN = 100;
  const ARC_CENTERS_DEG = [210, 90, 330];

  return (
    <div className="pop-tracks">
      <svg viewBox={`0 0 ${W} ${W + 8}`} width={W} height={W + 8} style={{ display: 'block' }}>
        {tracks.map((track, ti) => {
          const centerDeg = ARC_CENTERS_DEG[ti];
          const startDeg  = centerDeg - ARC_SPAN / 2;
          const prod = POPULATION_PRODUCTION_TABLE[Math.min(track.cubesOnTrack, POP_TRACK_MAX)];

          // Place 12 dots along the arc
          const dots = Array.from({ length: POP_TRACK_MAX }, (_, i) => {
            const frac = i / POP_TRACK_STEPS;
            const deg  = startDeg + frac * ARC_SPAN;
            const rad  = (deg * Math.PI) / 180;
            const x    = CX + OUTER_R * Math.cos(rad);
            const y    = CY + OUTER_R * Math.sin(rad);
            // Cubes fill from the start of the arc; empty = cube placed on sector
            const onTrack = i < track.cubesOnTrack;
            return { x, y, onTrack };
          });

          // Label position: inside the arc, near arc center
          const labelRad = (centerDeg * Math.PI) / 180;
          const lx = CX + INNER_R * Math.cos(labelRad);
          const ly = CY + INNER_R * Math.sin(labelRad);

          return (
            <g key={ti}>
              {dots.map((d, i) => (
                <circle
                  key={i}
                  cx={d.x}
                  cy={d.y}
                  r={DOT_R}
                  fill={d.onTrack ? track.color : 'none'}
                  stroke={track.color}
                  strokeWidth={0.8}
                  opacity={d.onTrack ? 0.9 : 0.35}
                />
              ))}
              {/* Production value inside the arc */}
              <text
                x={lx}
                y={ly + 3}
                textAnchor="middle"
                fill={track.color}
                fontSize="9"
                fontWeight="bold"
              >
                {prod}
              </text>
              {/* Track label at outer edge of arc center */}
              <text
                x={CX + (OUTER_R + 10) * Math.cos(labelRad)}
                y={CY + (OUTER_R + 10) * Math.sin(labelRad) + 3}
                textAnchor="middle"
                fill={track.color}
                fontSize="7"
                opacity={0.7}
              >
                {track.label}
              </text>
            </g>
          );
        })}
        {/* Center dot for aesthetics */}
        <circle cx={CX} cy={CY} r={2.5} fill={playerColor} opacity={0.5} />
      </svg>
    </div>
  );
}
