import type { ReputationSlot } from '../../types/game';
import { ACTION } from '../../actionTypes';

interface ReputationTrackProps {
  track: ReputationSlot[];
  playerColors: string[];
  interactive?: boolean;
  side?: number;
  legalReturns?: Set<number>;
  legalSwapSources?: Set<number>;
  legalSwapDestinations?: Map<number, Set<number>>;
  selectedSwapSource?: number | null;
  onSelectSwapSource?: (slot: number | null) => void;
  onAction?: (actionId: number) => void;
  compact?: boolean;
}

const SLOT_KIND_COLORS: Record<string, string> = {
  ambassador_or_rep: '#b45309',
  ambassador_only: '#15803d',
  rep_only: '#475569',
};

const SLOT_W = 80;
const SLOT_H = 44;
const SLOT_GAP = 6;

function ambassadorLabel(ambassadorFrom: number, playerColors: string[]): { label: string; color: string } {
  if (ambassadorFrom === 255) return { label: 'MS', color: '#fbbf24' };
  return { label: `P${ambassadorFrom + 1}`, color: playerColors[ambassadorFrom] ?? '#94a3b8' };
}

export default function ReputationTrack({
  track,
  playerColors,
  interactive = false,
  side,
  legalReturns,
  legalSwapSources,
  legalSwapDestinations,
  selectedSwapSource,
  onSelectSwapSource,
  onAction,
  compact = false,
}: ReputationTrackProps) {
  const w = compact ? 24 : SLOT_W;
  const h = compact ? 32 : SLOT_H;
  const gap = compact ? 4 : SLOT_GAP;
  const totalW = track.length * w + (track.length - 1) * gap;
  const dotR = compact ? 5 : 7;

  const handleSlotClick = (i: number, isReturn: boolean | undefined, isSrc: boolean | undefined, isDest: boolean | undefined) => {
    if (!interactive || !onAction) return;
    if (isReturn) {
      if (side === undefined) return;
      onAction(ACTION.DIPLOMACY_RETURN_START + side * 5 + i);
      return;
    }
    if (isSrc) {
      onSelectSwapSource?.(i);
      return;
    }
    if (isDest && selectedSwapSource !== null && selectedSwapSource !== undefined && selectedSwapSource >= 0) {
      if (side === undefined) return;
      onAction(ACTION.DIPLOMACY_SWAP_START + side * 20 + selectedSwapSource * 5 + i);
      onSelectSwapSource?.(null);
    }
  };

  return (
    <div className="reputation-track" style={{ display: 'inline-block' }}>
      <svg width={totalW} height={h + 14} style={{ display: 'block' }}>
        {track.map((slot, i) => {
          const x = i * (w + gap);
          const color = SLOT_KIND_COLORS[slot.kind] ?? '#475569';

          const isReturnTarget = interactive && legalReturns?.has(i);
          const isSwapSource = interactive && legalSwapSources?.has(i);
          const isSelected = selectedSwapSource === i;
          const isSwapDest = interactive && selectedSwapSource !== null && selectedSwapSource !== undefined &&
            selectedSwapSource >= 0 &&
            legalSwapDestinations?.get(selectedSwapSource)?.has(i);

          const canClick = isReturnTarget || isSwapSource || isSwapDest;

          return (
            <g
              key={i}
              onClick={() => handleSlotClick(i, isReturnTarget, isSwapSource, isSwapDest)}
              style={{ cursor: (canClick ? 'pointer' : undefined) as string | undefined }}
            >
              <rect
                x={x}
                y={2}
                width={w}
                height={h}
                rx={4}
                fill={color}
                opacity={0.25}
                stroke={
                  isSelected ? '#facc15' :
                  isSwapDest ? '#22c55e' :
                  isReturnTarget ? '#ef4444' :
                  isSwapSource ? '#facc15' :
                  '#64748b'
                }
                strokeWidth={canClick ? 2.5 : 1}
                strokeDasharray={
                  !slot.holds_ambassador && slot.rep_value === 'None' ? '4 2' : 'none'
                }
              />
              {slot.holds_ambassador ? (
                <>
                  <circle
                    cx={x + w / 2}
                    cy={h / 2 + 2}
                    r={dotR}
                    fill={ambassadorLabel(slot.ambassador_from, playerColors).color}
                  />
                  {!compact && (
                    <text
                      x={x + w / 2}
                      y={h + 12}
                      textAnchor="middle"
                      fill={ambassadorLabel(slot.ambassador_from, playerColors).color}
                      fontSize="7"
                    >
                      {ambassadorLabel(slot.ambassador_from, playerColors).label}
                    </text>
                  )}
                  {slot.pending_track_choice && (
                    <text
                      x={x + w / 2}
                      y={h / 2 + 2}
                      textAnchor="middle"
                      fill="#fff"
                      fontSize="8"
                      fontWeight="bold"
                    >
                      ?
                    </text>
                  )}
                </>
              ) : slot.rep_value !== 'None' ? (
                <text
                  x={x + w / 2}
                  y={h / 2 + 6}
                  textAnchor="middle"
                  fill="#e2e8f0"
                  fontSize={compact ? '9' : '13'}
                  fontWeight="bold"
                >
                  {slot.rep_value === 'One' ? '1' :
                   slot.rep_value === 'Two' ? '2' :
                   slot.rep_value === 'Three' ? '3' :
                   slot.rep_value === 'Four' ? '4' : ''}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
