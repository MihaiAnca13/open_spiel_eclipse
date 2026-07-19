// Upkeep / Cleanup phase controls. The action phase already blocks here for a
// human seat because upkeep needs decisions (colony placement, bankruptcy
// resolution, pay confirmation). These reuse the existing colony-ship / trade /
// reclaim / return-track action ranges (see UpkeepLegalActions in eclipse.cc),
// so decoding mirrors App.tsx's action-phase decoders.

import { useCallback, useMemo } from 'react';
import { ACTION, POP_TRACK_LABELS, TRADE_LABELS } from '../../actionTypes';
import { INFLUENCE_UPKEEP, POPULATION_PRODUCTION_TABLE, POP_TRACK_MAX } from '../../utils/game';
import type { GameState, Player, UpkeepState } from '../../types/game';
import { GALAXY_MAP_SIZE } from '../../constants';

interface Props {
  upkeep: UpkeepState | undefined;
  player: Player | undefined;
  gameState: GameState;
  legalActions: number[];
  actionStrings?: Record<string, string>;
  busy: boolean;
  onAction: (actionId: number) => void;
}

export default function UpkeepPanel({
  upkeep,
  player,
  gameState,
  legalActions,
  actionStrings,
  busy,
  onAction,
}: Props) {
  const legal = useMemo(() => new Set(legalActions), [legalActions]);
  const step = upkeep?.step ?? 'inactive';

  const cellToSectorId = useCallback((cell: number): number => {
    const qIdx = Math.floor(cell / GALAXY_MAP_SIZE);
    const rIdx = cell % GALAXY_MAP_SIZE;
    return gameState.galaxy[qIdx]?.[rIdx]?.sector_id ?? 0;
  }, [gameState.galaxy]);

  const inRange = (id: number, start: number, end: number) => id >= start && id < end;

  // Decode colony-ship placements (COLONY_SHIP_START range), same as App.tsx.
  const colonyPlacements = useMemo(() => {
    return legalActions
      .filter((id) => inRange(id, ACTION.COLONY_SHIP_START, ACTION.INFLUENCE))
      .map((id) => {
        const encoded = id - ACTION.COLONY_SHIP_START;
        const cellIdx = Math.floor(encoded / ACTION.COLONY_SHIP_CODES_PER_CELL);
        const rem = encoded % ACTION.COLONY_SHIP_CODES_PER_CELL;
        const slotIdx = Math.floor(rem / ACTION.COLONY_SHIP_TRACKS);
        const track = rem % ACTION.COLONY_SHIP_TRACKS;
        return { id, sectorId: cellToSectorId(cellIdx), slotIdx, track };
      });
  }, [cellToSectorId, legalActions]);

  const tradeActions = useMemo(
    () => legalActions.filter((id) => inRange(id, ACTION.TRADE_START, ACTION.COLONY_SHIP_START)),
    [legalActions]
  );
  const reclaimActions = useMemo(
    () => legalActions.filter((id) => inRange(id, ACTION.RECLAIM_FROM_CELL_START, ACTION.CHOOSE_RETURN_TRACK_START)),
    [legalActions]
  );

  const prod = (idx: number) => POPULATION_PRODUCTION_TABLE[Math.min(idx, POP_TRACK_MAX)];
  const discsPlaced = player
    ? Math.min(INFLUENCE_UPKEEP.length - 1, (player.disks_on_sectors ?? 0) + (player.disks_on_actions ?? 0))
    : 0;

  const stepLabel: Record<string, string> = {
    inactive: 'Upkeep',
    colony_ships: 'Restore colony ships',
    bankruptcy: 'Pay upkeep',
    cleanup_graveyards: 'Cleanup',
    choose_return_track: 'Return population',
  };

  // Track which ids get an explicit button so the fallback can cover the rest.
  const handled = new Set<number>();
  const mark = (id: number) => { handled.add(id); return legal.has(id); };

  return (
    <div className="panel action-panel">
      <h3 className="panel-title">♻️ {stepLabel[step] ?? 'Round end'}</h3>

      {/* Income / upkeep summary */}
      {player && (
        <div className="text-xs text-[#cbd5e1]" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <span>💰 {player.resources.gold} <em className="text-[#94a3b8]">+{prod(player.resources.gold_prod)}</em></span>
          <span>🔬 {player.resources.science} <em className="text-[#94a3b8]">+{prod(player.resources.science_prod)}</em></span>
          <span>⚙️ {player.resources.materials} <em className="text-[#94a3b8]">+{prod(player.resources.materials_prod)}</em></span>
          <span title="Influence upkeep cost">🧾 upkeep {INFLUENCE_UPKEEP[discsPlaced]} 💰</span>
        </div>
      )}

      {step === 'colony_ships' && (
        <>
          <span className="text-xs text-[#94a3b8]">
            Restore colony ships onto your population tracks, or finish.
          </span>
          {colonyPlacements.length > 0 && (
            <div className="action-row" style={{ flexDirection: 'column', gap: 4 }}>
              {colonyPlacements.map(({ id, sectorId, slotIdx, track }) => {
                mark(id);
                return (
                  <button key={id} className="action-btn primary" disabled={busy} onClick={() => onAction(id)}>
                    Sector {sectorId} · slot {slotIdx} · {POP_TRACK_LABELS[track]}
                  </button>
                );
              })}
            </div>
          )}
          <div className="action-row">
            {mark(ACTION.UPKEEP_COLONY_DONE) && (
              <button className="action-btn secondary" disabled={busy} onClick={() => onAction(ACTION.UPKEEP_COLONY_DONE)}>
                Done
              </button>
            )}
          </div>
        </>
      )}

      {step === 'bankruptcy' && (
        <>
          {mark(ACTION.UPKEEP_PAY_DONE) && legal.has(ACTION.UPKEEP_PAY_DONE) ? (
            <>
              <span className="text-xs text-[#94a3b8]">You can cover your upkeep. Pay and continue.</span>
              <div className="action-row">
                <button className="action-btn primary" disabled={busy} onClick={() => onAction(ACTION.UPKEEP_PAY_DONE)}>
                  Pay upkeep
                </button>
              </div>
            </>
          ) : (
            <span className="text-xs text-[#fca5a5]">
              ⚠️ You cannot cover your upkeep. Trade resources or reclaim influence discs to raise the money.
            </span>
          )}
          {tradeActions.length > 0 && (
            <div className="action-row" style={{ flexDirection: 'column', gap: 4 }}>
              {tradeActions.map((id) => {
                mark(id);
                const info = TRADE_LABELS[id - ACTION.TRADE_START];
                return (
                  <button key={id} className="action-btn secondary" disabled={busy} onClick={() => onAction(id)}>
                    {info ? `${info.emoji} ${info.from} → ${info.to}` : `Trade ${id}`}
                  </button>
                );
              })}
            </div>
          )}
          {reclaimActions.length > 0 && (
            <div className="action-row" style={{ flexDirection: 'column', gap: 4 }}>
              {reclaimActions.map((id) => {
                mark(id);
                const sid = cellToSectorId(id - ACTION.RECLAIM_FROM_CELL_START);
                return (
                  <button key={id} className="action-btn secondary" disabled={busy} onClick={() => onAction(id)}>
                    Reclaim disc from Sector {sid || '?'}
                  </button>
                );
              })}
            </div>
          )}
        </>
      )}

      {step === 'choose_return_track' && (
        <>
          <span className="text-xs text-[#94a3b8]">Choose a population track for the returning cube:</span>
          <div className="action-row">
            {mark(ACTION.CHOOSE_RETURN_TRACK_START + 0) && (
              <button className="action-btn primary" disabled={busy} onClick={() => onAction(ACTION.CHOOSE_RETURN_TRACK_START + 0)}>💰 Money</button>
            )}
            {mark(ACTION.CHOOSE_RETURN_TRACK_START + 1) && (
              <button className="action-btn primary" disabled={busy} onClick={() => onAction(ACTION.CHOOSE_RETURN_TRACK_START + 1)}>🔬 Science</button>
            )}
            {mark(ACTION.CHOOSE_RETURN_TRACK_START + 2) && (
              <button className="action-btn primary" disabled={busy} onClick={() => onAction(ACTION.CHOOSE_RETURN_TRACK_START + 2)}>⚙️ Materials</button>
            )}
          </div>
        </>
      )}

      {/* Safety-net fallback for any unhandled legal action (e.g. cleanup steps). */}
      {(() => {
        const rest = legalActions.filter((id) => !handled.has(id));
        if (rest.length === 0) return null;
        return (
          <div className="action-row" style={{ flexDirection: 'column', gap: 4, marginTop: 6 }}>
            {rest.map((id) => (
              <button key={id} className="action-btn secondary" disabled={busy} onClick={() => onAction(id)}>
                {actionStrings?.[String(id)] ?? `Action ${id}`}
              </button>
            ))}
          </div>
        );
      })()}
    </div>
  );
}
