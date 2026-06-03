// Contextual gameplay controls for the Action phase. Driven entirely by the
// server-supplied `legal_actions` list plus the explore sub-phase in
// `gameState.explore_state`, so the UI never needs to reimplement game rules.

import { ACTION } from './actionTypes';

const RING_NAME: Record<number, string> = { 0: 'Inner (I)', 1: 'Middle (II)', 2: 'Outer (III)' };

export interface ExploreState {
  phase: string;
  player_id: number;
  activations_remaining: number;
  zone_q: number;
  zone_r: number;
  ring: number;
  drawn_sector_ids: number[];
  drawn_count: number;
  selected_sector_id: number;
  chosen_rotation: number;
}

interface Props {
  explore: ExploreState | undefined;
  legalActions: number[];
  isMyTurn: boolean;
  isTerminal: boolean;
  busy: boolean;
  currentPlayerLabel: string;
  ancientsOnSelected: boolean;
  previewRotation: number | null;
  onConfirmPreviewRotation: () => void;
  onAction: (actionId: number) => void;
}

export default function ActionPanel({
  explore,
  legalActions,
  isMyTurn,
  isTerminal,
  busy,
  currentPlayerLabel,
  ancientsOnSelected,
  previewRotation,
  onConfirmPreviewRotation,
  onAction,
}: Props) {
  const legal = new Set(legalActions);
  const phase = explore?.phase ?? 'inactive';

  const renderActionButton = (id: number, label: string, kind: 'primary' | 'secondary' | 'danger' = 'primary') =>
    legal.has(id) ? (
      <button className={`action-btn ${kind}`} disabled={busy} onClick={() => onAction(id)}>
        {label}
      </button>
    ) : null;

  if (isTerminal) {
    return (
      <div className="panel action-panel">
        <h3 className="panel-title">Game Over</h3>
        <span className="text-xs text-[#94a3b8]">Final scores determine the winner.</span>
      </div>
    );
  }

  if (!isMyTurn) {
    return (
      <div className="panel action-panel">
        <h3 className="panel-title">Action Phase</h3>
        <div className="action-waiting">
          <span className="action-spinner" />
          Waiting for {currentPlayerLabel}…
        </div>
      </div>
    );
  }

  const drawnId =
    explore && explore.selected_sector_id > 0
      ? explore.selected_sector_id
      : explore?.drawn_sector_ids?.find((s) => s > 0) ?? 0;

  return (
    <div className="panel action-panel">
      <h3 className="panel-title">Your Action</h3>

      {phase === 'inactive' && (
        <>
          <span className="text-xs text-[#94a3b8]">
            Take one action. Play continues around the table — you act again each turn until you <strong>Pass</strong> for the round.
          </span>
          <div className="action-row">
            {renderActionButton(ACTION.EXPLORE, '🔭 Explore')}
            {renderActionButton(ACTION.PASS, '✋ Pass', 'secondary')}
          </div>
          <div className="action-row">
            <button className="action-btn secondary" disabled title="Not implemented yet">Research (soon)</button>
            <button className="action-btn secondary" disabled title="Not implemented yet">Build (soon)</button>
          </div>
        </>
      )}

      {phase === 'choose_zone' && (
        <>
          <span className="text-xs text-[#94a3b8]">
            Select a highlighted zone on the map to explore.
            {explore ? ` Activations left: ${explore.activations_remaining}.` : ''}
          </span>
          <div className="action-row">
            {renderActionButton(ACTION.EXPLORE_STOP, 'Stop exploring', 'secondary')}
          </div>
        </>
      )}

      {phase === 'place_or_discard' && (
        <>
          <div className="drawn-tile">
            <span className="drawn-tile-label">Drawn sector</span>
            <span className="drawn-tile-id">#{drawnId}</span>
            <span className="text-xs text-[#94a3b8]">Ring {RING_NAME[explore?.ring ?? -1] ?? explore?.ring}</span>
          </div>
          {legal.has(ACTION.EXPLORE_PLACE) ? (
            <span className="text-xs text-[#94a3b8]">Place this tile (you choose its rotation next) or discard it.</span>
          ) : (
            <span className="text-xs text-[#fca5a5]">
              🚫 This tile has no wormhole connection to your territory in any rotation — it can't be placed, so you must discard it. (Exploring is a gamble: not every tile fits.)
            </span>
          )}
          <div className="action-row">
            {renderActionButton(ACTION.EXPLORE_PLACE, 'Place tile')}
            {renderActionButton(ACTION.EXPLORE_DISCARD, 'Discard', legal.has(ACTION.EXPLORE_PLACE) ? 'danger' : 'primary')}
          </div>
        </>
      )}

      {phase === 'choose_rotation' && (
        <>
          <span className="text-xs text-[#94a3b8]">
            Preview the sector on the map, rotate it with the arrows, then confirm.
          </span>
          <div className="drawn-tile">
            <span className="drawn-tile-label">Preview rotation</span>
            <span className="drawn-tile-id">{previewRotation ?? '-'}</span>
          </div>
          <div className="action-row">
            <button
              className="action-btn primary"
              disabled={busy || previewRotation === null}
              onClick={onConfirmPreviewRotation}
            >
              Confirm rotation
            </button>
          </div>
        </>
      )}

      {phase === 'claim_control' && (
        <>
          {legal.has(ACTION.EXPLORE_CLAIM_YES) ? (
            <span className="text-xs text-[#94a3b8]">Claim control of the new sector? (spends an Influence disc)</span>
          ) : (
            <span className="text-xs text-[#fca5a5]">
              {ancientsOnSelected
                ? '⚔️ Ancients defend this sector — you cannot take control until they are destroyed in the Combat phase.'
                : 'No Influence discs available to claim control.'}
            </span>
          )}
          <div className="action-row">
            {renderActionButton(ACTION.EXPLORE_CLAIM_YES, 'Claim control')}
            {renderActionButton(ACTION.EXPLORE_CLAIM_NO, legal.has(ACTION.EXPLORE_CLAIM_YES) ? 'Decline' : 'Continue', 'secondary')}
          </div>
        </>
      )}

      {phase === 'discovery_reward' && (
        <>
          <span className="text-xs text-[#94a3b8]">Resolve the Discovery tile in this sector.</span>
          <div className="action-row">
            {renderActionButton(ACTION.EXPLORE_DISCOVERY_REWARD, 'Take reward')}
            {renderActionButton(ACTION.EXPLORE_DISCOVERY_VP, 'Take 2 VP', 'secondary')}
          </div>
        </>
      )}

      {phase === 'select_drawn_tile' && (
        <>
          <span className="text-xs text-[#94a3b8]">Keep one of the drawn tiles.</span>
          <div className="action-row">
            {renderActionButton(ACTION.EXPLORE_SELECT_TILE_START, `Keep #${explore?.drawn_sector_ids?.[0] ?? 0}`)}
            {renderActionButton(ACTION.EXPLORE_SELECT_TILE_START + 1, `Keep #${explore?.drawn_sector_ids?.[1] ?? 0}`)}
          </div>
        </>
      )}

      {phase === 'draw_again_decision' && (
        <>
          <span className="text-xs text-[#94a3b8]">Draw a second tile or proceed with this one?</span>
          <div className="action-row">
            {renderActionButton(ACTION.EXPLORE_DRAW_AGAIN, 'Draw again')}
            {renderActionButton(ACTION.EXPLORE_SKIP_SECOND, 'Proceed', 'secondary')}
          </div>
        </>
      )}
    </div>
  );
}
