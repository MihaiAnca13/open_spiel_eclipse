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

export interface ResearchState {
  phase: string;
  player_id: number;
  activations_remaining: number;
}

interface Props {
  explore: ExploreState | undefined;
  research: ResearchState | undefined;
  selectedRareTech: { name: string; order: number } | null;
  onClearSelectedRareTech: () => void;
  legalActions: number[];
  isMyTurn: boolean;
  isTerminal: boolean;
  busy: boolean;
  currentPlayerLabel: string;
  ancientsOnSelected: boolean;
  previewRotation: number | null;
  sectorImages: Record<number, string>;
  onPreviewDrawnTile: (sectorId: number, index: number) => void;
  onClearPreviewDrawnTile: () => void;
  onConfirmPreviewRotation: () => void;
  onAction: (actionId: number) => void;
}

export default function ActionPanel({
  explore,
  research,
  selectedRareTech,
  onClearSelectedRareTech,
  legalActions,
  isMyTurn,
  isTerminal,
  busy,
  currentPlayerLabel,
  ancientsOnSelected,
  previewRotation,
  sectorImages,
  onPreviewDrawnTile,
  onClearPreviewDrawnTile,
  onConfirmPreviewRotation,
  onAction,
}: Props) {
  const legal = new Set(legalActions);
  const phase = research?.phase === 'choose_tech' ? 'choose_tech' : (explore?.phase ?? 'inactive');

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
            {renderActionButton(ACTION.RESEARCH, '🔬 Research')}
          </div>
          <div className="action-row">
            {renderActionButton(ACTION.PASS, '✋ Pass', 'secondary')}
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
          <span className="text-xs text-[#94a3b8]">Choose one of the two drawn tiles to keep.</span>
          <div className="drawn-tiles-grid">
            {explore?.drawn_sector_ids?.filter(id => id > 0).map((sectorId, idx) => {
              const imgUrl = sectorImages[sectorId];
              return (
                <div
                  key={idx}
                  className={`drawn-tile-card ${legal.has(ACTION.EXPLORE_SELECT_TILE_START + idx) ? 'clickable' : 'disabled'}`}
                  onClick={() => {
                    if (legal.has(ACTION.EXPLORE_SELECT_TILE_START + idx)) {
                      onAction(ACTION.EXPLORE_SELECT_TILE_START + idx);
                      onClearPreviewDrawnTile();
                    }
                  }}
                  onMouseEnter={() => onPreviewDrawnTile(sectorId, idx)}
                  onMouseLeave={onClearPreviewDrawnTile}
                >
                  <div className="drawn-tile-card-thumb">
                    {imgUrl ? (
                      <img src={imgUrl} alt={`Sector ${sectorId}`} className="drawn-tile-card-img" />
                    ) : (
                      <div className="drawn-tile-card-placeholder">
                        <span className="text-2xl">🌌</span>
                      </div>
                    )}
                    <div className="drawn-tile-card-badge">Tile {idx + 1}</div>
                  </div>
                  <div className="drawn-tile-card-info">
                    <span className="drawn-tile-card-id">Sector {sectorId}</span>
                    <span className="text-xs text-[#64748b]">{legal.has(ACTION.EXPLORE_SELECT_TILE_START + idx) ? 'Click to keep' : 'Unavailable'}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {phase === 'draw_again_decision' && (
        <>
          {(() => {
            const firstSectorId = explore?.drawn_sector_ids?.[0] ?? 0;
            const imgUrl = firstSectorId > 0 ? sectorImages[firstSectorId] : null;
            return (
              <div
                className="drawn-tile-card-draw-again"
                onMouseEnter={() => firstSectorId > 0 && onPreviewDrawnTile(firstSectorId, 0)}
                onMouseLeave={onClearPreviewDrawnTile}
              >
                <div className="drawn-tile-card-draw-again-thumb">
                  {imgUrl ? (
                    <img src={imgUrl} alt={`Sector ${firstSectorId}`} className="drawn-tile-card-draw-again-img" />
                  ) : (
                    <div className="drawn-tile-card-draw-again-placeholder">
                      <span className="text-4xl">🌌</span>
                    </div>
                  )}
                  <div className="drawn-tile-card-draw-again-badge">Your Tile</div>
                </div>
                <div className="drawn-tile-card-draw-again-info">
                  <span className="drawn-tile-card-draw-again-id">Sector {firstSectorId}</span>
                  <span className="text-xs text-[#64748b]">Draw again for a second tile or proceed with this one.</span>
                </div>
              </div>
            );
          })()}
          <div className="action-row">
            {renderActionButton(ACTION.EXPLORE_DRAW_AGAIN, 'Draw again')}
            {renderActionButton(ACTION.EXPLORE_SKIP_SECOND, 'Proceed', 'secondary')}
          </div>
        </>
      )}

      {phase === 'choose_tech' && (
        <>
          {selectedRareTech ? (
            <>
              <span className="text-xs text-[#94a3b8]">
                Choose which track to place the Rare Tech <strong>{selectedRareTech.name}</strong> on:
              </span>
              <div className="action-row flex-col gap-2 mt-2">
                {(() => {
                  const rareIdx = selectedRareTech.order - 24;
                  return (
                    <>
                      {renderActionButton(
                        ACTION.RESEARCH_RARE_START + rareIdx * 3 + 0,
                        '⚔️ Military Track'
                      )}
                      {renderActionButton(
                        ACTION.RESEARCH_RARE_START + rareIdx * 3 + 1,
                        '🔬 Grid Track'
                      )}
                      {renderActionButton(
                        ACTION.RESEARCH_RARE_START + rareIdx * 3 + 2,
                        '⚙️ Nano Track'
                      )}
                    </>
                  );
                })()}
                <button
                  className="action-btn secondary mt-1"
                  disabled={busy}
                  onClick={onClearSelectedRareTech}
                >
                  Back to Tech Market
                </button>
              </div>
            </>
          ) : (
            <>
              <span className="text-xs text-[#94a3b8]">
                Select a technology card in the market below to research.
                {research ? ` Activations left: ${research.activations_remaining}.` : ''}
              </span>
              <div className="action-row mt-2">
                {renderActionButton(ACTION.RESEARCH_STOP, 'Stop researching', 'secondary')}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
