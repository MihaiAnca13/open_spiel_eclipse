import { useMemo } from 'react';
import { ACTION } from '../../actionTypes';
import { shipImageUrl } from '../../types/lobby';
import type { Unit, MoveState } from '../../types/game';

const SHIP_EMOJI: Record<string, string> = {
  Interceptor: '🚀',
  Cruiser: '🛡️',
  Dreadnought: '💥',
  Starbase: '🏰',
};

interface Props {
  legalActions: number[];
  unitRegistry: Unit[];
  busy: boolean;
  onAction: (actionId: number) => void;
  moveState: MoveState;
  selectedMoveUnitIdx: number | null;
  onSelectMoveUnit: (idx: number | null) => void;
  selectedMoveSectorId?: number | null;
}

export default function MovePanel({
  legalActions, unitRegistry, busy, onAction, moveState,
  selectedMoveUnitIdx, onSelectMoveUnit,
  selectedMoveSectorId,
}: Props) {
  const legal = new Set(legalActions);
  const phase = moveState.phase;
  const canStop = legal.has(ACTION.MOVE_STOP);

  const optionsByUnit = useMemo(() => {
    const map = new Map<number, { direction: number; isWarp: boolean }[]>();
    for (const a of legalActions) {
      if (a >= ACTION.MOVE_CHOICE_START && a < ACTION.MOVE_WARP_START) {
        const encoded = a - ACTION.MOVE_CHOICE_START;
        const unitIdx = Math.floor(encoded / ACTION.MOVE_CODES_PER_UNIT);
        const direction = encoded % ACTION.MOVE_CODES_PER_UNIT;
        const entry = map.get(unitIdx) || [];
        entry.push({ direction, isWarp: direction === 6 });
        map.set(unitIdx, entry);
      }
    }
    return map;
  }, [legalActions]);

  const warpDestinations = useMemo(() => {
    const cells: number[] = [];
    for (const a of legalActions) {
      if (a >= ACTION.MOVE_WARP_DESTINATION_START) {
        cells.push(a - ACTION.MOVE_WARP_DESTINATION_START);
      }
    }
    return cells;
  }, [legalActions]);

  const unitArray = useMemo(() => {
    const seenUnits = new Set<number>();
    for (const a of legalActions) {
      if (a >= ACTION.MOVE_CHOICE_START && a < ACTION.MOVE_WARP_START) {
        const encoded = a - ACTION.MOVE_CHOICE_START;
        const unitIdx = Math.floor(encoded / ACTION.MOVE_CODES_PER_UNIT);
        seenUnits.add(unitIdx);
      }
    }
    return [...seenUnits]
      .map(idx => ({ idx, unit: unitRegistry[idx] }))
      .filter(e => e.unit)
      .filter(({ unit }) => selectedMoveSectorId == null || unit.sector_id === selectedMoveSectorId);
  }, [legalActions, unitRegistry, selectedMoveSectorId]);

  const selectedUnit = selectedMoveUnitIdx !== null ? unitRegistry[selectedMoveUnitIdx] : null;

  const renderShipIcon = (typeName: string) => {
    if (typeName in SHIP_EMOJI) {
      return <span className="move-ship-emoji">{SHIP_EMOJI[typeName]}</span>;
    }
    return null;
  };

  return (
    <div className="move-panel">
      {phase === 'choose_move' && (
        <>
          <div className="move-header">
            <span className="text-xs text-[#94a3b8]">
              {selectedMoveUnitIdx === null
                ? 'Click a highlighted sector on the map to select a ship to move.'
                : `Selected ship: ${selectedUnit?.type ?? 'Unknown'} in sector ${selectedUnit?.sector_id}. Click a highlighted hex to move.`}
              {moveState.activations_remaining > 1
                ? ` Activations left: ${moveState.activations_remaining}.`
                : ''}
            </span>
            {canStop && (
              <button
                className="action-btn secondary"
                style={{ padding: '5px 10px', fontSize: '11px', flex: 'none' }}
                disabled={busy}
                onClick={() => {
                  onSelectMoveUnit(null);
                  onAction(ACTION.MOVE_STOP);
                }}
              >
                Stop moving
              </button>
            )}
          </div>

          {unitArray.length === 0 && (
            <span className="text-xs text-[#fca5a5]">
              No ships can move (pinned, no movement value, or no wormhole connections).
            </span>
          )}

          {selectedMoveUnitIdx !== null && selectedUnit && (
            <div className="move-unit-card">
              <div className="move-unit-header">
                <img
                  className="move-ship-img"
                  src={shipImageUrl(String(selectedUnit.type))}
                  alt={String(selectedUnit.type)}
                  onError={(e) => { e.currentTarget.style.display = 'none'; }}
                />
                {renderShipIcon(String(selectedUnit.type))}
                <span className="move-unit-name">{String(selectedUnit.type)}</span>
                <span className="move-unit-sector">Sector {selectedUnit.sector_id}</span>
              </div>
              <div className="move-dirs">
                {(() => {
                  const opts = optionsByUnit.get(selectedMoveUnitIdx) ?? [];
                  const dirs = opts.filter(o => !o.isWarp);
                  const hasWarp = opts.some(o => o.isWarp);
                  return (
                    <>
                      {dirs.map((opt) => {
                        const actionId = ACTION.MOVE_CHOICE_START + selectedMoveUnitIdx * ACTION.MOVE_CODES_PER_UNIT + opt.direction;
                        return (
                          <button
                            key={actionId}
                            className="move-dir-btn"
                            disabled={busy}
                            onClick={() => onAction(actionId)}
                          >
                            {['E', 'NE', 'NW', 'W', 'SW', 'SE'][opt.direction]}
                          </button>
                        );
                      })}
                      {hasWarp && (
                        <button
                          className="move-dir-btn warp"
                          disabled={busy}
                          onClick={() => {
                            const warpOpt = opts.find(o => o.isWarp);
                            if (warpOpt) {
                              onAction(ACTION.MOVE_CHOICE_START + selectedMoveUnitIdx * ACTION.MOVE_CODES_PER_UNIT + 6);
                            }
                          }}
                        >
                          ⚡
                        </button>
                      )}
                    </>
                  );
                })()}
              </div>
              <button
                className="action-btn secondary"
                style={{ padding: '4px 8px', fontSize: '10px', flex: 'none', marginTop: '4px' }}
                disabled={busy}
                onClick={() => onSelectMoveUnit(null)}
              >
                Back to ships
              </button>
            </div>
          )}

          {selectedMoveUnitIdx === null && unitArray.length > 0 && (
            <div className="move-unit-list">
              <span className="text-xs text-[#94a3b8]">Movable ships:</span>
              {unitArray.map(({ idx, unit }) => (
                <div
                  key={idx}
                  className="move-unit-card"
                  style={{ cursor: 'pointer' }}
                  onClick={() => onSelectMoveUnit(idx)}
                >
                  <div className="move-unit-header">
                    <img
                      className="move-ship-img"
                      src={shipImageUrl(String(unit.type))}
                      alt={String(unit.type)}
                      onError={(e) => { e.currentTarget.style.display = 'none'; }}
                    />
                    {renderShipIcon(String(unit.type))}
                    <span className="move-unit-name">{String(unit.type)}</span>
                    <span className="move-unit-sector">Sector {unit.sector_id}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {phase === 'choose_warp_destination' && (
        <>
          <div className="move-header">
            <span className="text-xs text-[#94a3b8]">
              Click a highlighted sector on the map to warp to that destination.
            </span>
          </div>
          {warpDestinations.length === 0 && (
            <span className="text-xs text-[#fca5a5]">No warp portal destinations available.</span>
          )}
          <div className="move-warp-targets">
            {warpDestinations.map((cell) => {
              const actionId = ACTION.MOVE_WARP_DESTINATION_START + cell;
              const q = Math.floor(cell / 15) - 7;
              const r = (cell % 15) - 7;
              return (
                <button
                  key={actionId}
                  className="move-warp-btn"
                  disabled={busy}
                  onClick={() => onAction(actionId)}
                >
                  <span className="move-warp-icon">⚡</span>
                  <span className="move-warp-coords">({q},{r})</span>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
