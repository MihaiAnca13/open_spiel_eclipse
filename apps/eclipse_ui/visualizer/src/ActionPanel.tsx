// Contextual gameplay controls for the Action phase. Driven entirely by the
// server-supplied `legal_actions` list plus the explore sub-phase in
// `gameState.explore_state`, so the UI never needs to reimplement game rules.

import { ACTION } from './actionTypes';
import type { InfluenceState, BuildState, MoveState, Unit, BuildCosts } from './types/game';
import BuildPanel from './components/ui/BuildPanel';
import MovePanel from './components/ui/MovePanel';

const RING_NAME: Record<number, string> = { 0: 'Inner (I)', 1: 'Middle (II)', 2: 'Outer (III)' };
type MainActionPreview = 'research' | 'build' | 'influence' | 'upgrade' | 'move' | 'explore';

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
  influence: InfluenceState | undefined;
  build: BuildState | undefined;
  move: MoveState | undefined;
  unitRegistry?: Unit[];
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
  onStartPreviewAction: (actionId: number) => void;
  previewAction: MainActionPreview | null;
  onCancelPreviewAction: () => void;
  onConfirmPreviewAction: (actionId: number) => void;
  previewResearchActions: number[];
  selectedBuildType: number | null;
  onSelectBuildType: (type: number | null) => void;
  builtShipsCount: number;
  buildCosts?: BuildCosts | null;
  hasPassed: boolean;
  selectedMoveUnitIdx: number | null;
  onSelectMoveUnit: (idx: number | null) => void;
  selectedMoveSectorId?: number | null;
}

export default function ActionPanel({
  explore,
  research,
  influence,
  build,
  move,
  unitRegistry,
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
  onStartPreviewAction,
  previewAction,
  onCancelPreviewAction,
  onConfirmPreviewAction,
  previewResearchActions,
  selectedBuildType,
  onSelectBuildType,
  builtShipsCount,
  buildCosts,
  hasPassed,
  selectedMoveUnitIdx,
  onSelectMoveUnit,
  selectedMoveSectorId,
}: Props) {
  const legal = new Set(legalActions);
  const influencePhase = influence?.phase ?? 'inactive';
  const buildPhase = build?.phase ?? 'inactive';
  const movePhase = move?.phase ?? 'inactive';
  const phase = influencePhase !== 'inactive'
    ? influencePhase
    : buildPhase !== 'inactive'
      ? 'choose_build'
      : movePhase !== 'inactive'
        ? movePhase
        : research?.phase === 'choose_tech'
          ? 'choose_tech'
          : (explore?.phase ?? 'inactive');

  const renderActionButton = (id: number, label: string, kind: 'primary' | 'secondary' | 'danger' = 'primary', forceDisabled = false) =>
    legal.has(id) ? (
      <button className={`action-btn ${kind}`} disabled={busy || forceDisabled} onClick={() => onAction(id)}>
        {label}
      </button>
    ) : null;

  const renderPreviewButton = (id: number, label: string, kind: 'primary' | 'secondary' | 'danger' = 'primary', forceDisabled = false) =>
    legal.has(id) ? (
      <button className={`action-btn ${kind}`} disabled={busy || forceDisabled} onClick={() => onStartPreviewAction(id)}>
        {label}
      </button>
    ) : null;

  const renderResearchChoiceButton = (id: number, label: string) => {
    if (previewAction === 'research') {
      if (!previewResearchActions.includes(id)) return null;
      return (
        <button className="action-btn primary" disabled={busy} onClick={() => onConfirmPreviewAction(id)}>
          {label}
        </button>
      );
    }
    return renderActionButton(id, label);
  };

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
          {previewAction ? (
            <>
              <span className="text-xs text-[#94a3b8]">
                {previewAction === 'research' && 'Preview research choices. Pick a tech below to commit the action, or cancel.'}
                {previewAction === 'build' && 'Build preview. Review your current costs, then commit to enter build mode.'}
                {previewAction === 'influence' && 'Influence preview. Commit when you are ready to place or reclaim discs.'}
                {previewAction === 'upgrade' && 'Upgrade preview. Commit when you are ready to modify your blueprints.'}
                {previewAction === 'move' && 'Move preview. Commit when you are ready to choose ships and destinations.'}
                {previewAction === 'explore' && `Explore preview. Ring sizes: ${RING_NAME[0]} ${legal.has(ACTION.EXPLORE) ? '' : ''}. Commit when you are ready to pick a zone.`}
              </span>
              {previewAction === 'build' && buildCosts && (
                <div className="text-xs text-[#cbd5e1]">
                  Costs: I {buildCosts.Interceptor}, C {buildCosts.Cruiser}, D {buildCosts.Dreadnought}, S {buildCosts.Starbase}, O {buildCosts.Orbital}, M {buildCosts.Monolith}
                </div>
              )}
              <div className="action-row">
                {previewAction !== 'research' && (
                  <button className="action-btn primary" disabled={busy} onClick={() => onConfirmPreviewAction(
                    previewAction === 'build' ? ACTION.BUILD :
                    previewAction === 'influence' ? ACTION.INFLUENCE :
                    previewAction === 'upgrade' ? ACTION.UPGRADE :
                    previewAction === 'move' ? ACTION.MOVE :
                    ACTION.EXPLORE
                  )}>
                    Commit action
                  </button>
                )}
                <button className="action-btn secondary" disabled={busy} onClick={onCancelPreviewAction}>
                  Cancel
                </button>
              </div>
            </>
          ) : (
            <>
              <span className="text-xs text-[#94a3b8]">
                Take one action. Play continues around the table — you act again each turn until you <strong>Pass</strong> for the round.
              </span>
              <div className="action-row">
                {renderPreviewButton(ACTION.EXPLORE, '🔭 Explore', hasPassed ? 'secondary' : 'primary', hasPassed)}
                {renderPreviewButton(ACTION.RESEARCH, '🔬 Research', hasPassed ? 'secondary' : 'primary', hasPassed)}
              </div>
              <div className="action-row">
                {renderPreviewButton(ACTION.INFLUENCE, '🔵 Influence', hasPassed ? 'secondary' : 'primary', hasPassed)}
                {renderActionButton(ACTION.PASS, '✋ Pass', hasPassed ? 'secondary' : 'primary', hasPassed)}
              </div>
              <div className="action-row">
                {renderPreviewButton(ACTION.BUILD, '⚙️ Build', hasPassed ? 'secondary' : 'primary', hasPassed)}
                {renderPreviewButton(ACTION.MOVE, '↗️ Move', hasPassed ? 'secondary' : 'primary', hasPassed)}
              </div>
              <div className="action-row">
                {renderPreviewButton(ACTION.UPGRADE, '🛠️ Upgrade', hasPassed ? 'secondary' : 'primary', hasPassed)}
              </div>
            </>
          )}
        </>
      )}

      {phase === 'choose_influence' && influence && (
        <>
          <span className="text-xs text-[#94a3b8]">
            {influence.activations_remaining > 1
              ? `Manage your influence. Activations left: ${influence.activations_remaining}.`
              : 'Manage your influence.'}
            {' '}Click a highlighted sector on the map to place or reclaim a disc, or stop to finish.
          </span>
          <div className="action-row">
            {renderActionButton(ACTION.INFLUENCE_STOP, 'Stop influence', 'secondary')}
          </div>
        </>
      )}

      {phase === 'choose_return_track' && (
        <>
          <span className="text-xs text-[#94a3b8]">
            Choose which population track to place the returning cube on:
            {influence && influence.pending_returns.length > 1 && ` (${influence.pending_returns.length} cubes remaining)`}
          </span>
          <div className="action-row">
            {renderActionButton(ACTION.CHOOSE_RETURN_TRACK_START + 0, '💰 Money')}
            {renderActionButton(ACTION.CHOOSE_RETURN_TRACK_START + 1, '🔬 Science')}
            {renderActionButton(ACTION.CHOOSE_RETURN_TRACK_START + 2, '⚙️ Materials')}
          </div>
        </>
      )}

      {phase === 'choose_build' && (
        <BuildPanel
          legalActions={legalActions}
          busy={busy}
          onAction={onAction}
          activationsRemaining={build?.activations_remaining ?? 0}
          selectedBuildType={selectedBuildType}
          onSelectBuildType={onSelectBuildType}
          builtShipsCount={builtShipsCount}
          buildCosts={buildCosts}
        />
      )}

      {(phase === 'choose_move' || phase === 'choose_warp_destination') && move && (
        <MovePanel
          legalActions={legalActions}
          unitRegistry={unitRegistry ?? []}
          busy={busy}
          onAction={onAction}
          moveState={move}
          selectedMoveUnitIdx={selectedMoveUnitIdx}
          onSelectMoveUnit={onSelectMoveUnit}
          selectedMoveSectorId={selectedMoveSectorId}
        />
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

      {(phase === 'choose_tech' || previewAction === 'research') && (
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
                      {renderResearchChoiceButton(ACTION.RESEARCH_RARE_START + rareIdx * 3 + 0, '⚔️ Military Track')}
                      {renderResearchChoiceButton(ACTION.RESEARCH_RARE_START + rareIdx * 3 + 1, '🔬 Grid Track')}
                      {renderResearchChoiceButton(ACTION.RESEARCH_RARE_START + rareIdx * 3 + 2, '⚙️ Nano Track')}
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
                {previewAction === 'research'
                  ? 'Select a technology card in the market below to commit Research.'
                  : 'Select a technology card in the market below to research.'}
                {research ? ` Activations left: ${research.activations_remaining}.` : ''}
              </span>
              <div className="action-row mt-2">
                {phase === 'choose_tech'
                  ? renderActionButton(ACTION.RESEARCH_STOP, 'Stop researching', 'secondary')
                  : <button className="action-btn secondary" disabled={busy} onClick={onCancelPreviewAction}>Cancel</button>}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
