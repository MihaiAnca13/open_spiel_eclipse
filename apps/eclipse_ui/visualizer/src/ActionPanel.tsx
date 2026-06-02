// Contextual gameplay controls for the Action phase. Driven entirely by the
// server-supplied `legal_actions` list plus the explore sub-phase in
// `gameState.explore_state`, so the UI never needs to reimplement game rules.

// Action ids — mirror open_spiel/games/eclipse/eclipse.cc:36-52.
export const ACTION = {
  PASS: 0,
  EXPLORE: 17,
  EXPLORE_PLACE: 18,
  EXPLORE_DISCARD: 19,
  EXPLORE_ROT_START: 20, // +0..5
  EXPLORE_CLAIM_YES: 26,
  EXPLORE_CLAIM_NO: 27,
  EXPLORE_DISCOVERY_REWARD: 28,
  EXPLORE_DISCOVERY_VP: 29,
  EXPLORE_SELECT_TILE_START: 30, // +0..1
  EXPLORE_DRAW_AGAIN: 32,
  EXPLORE_SKIP_SECOND: 33,
  EXPLORE_STOP: 34,
  EXPLORE_ZONE_START: 35, // + hex_to_index(q, r)
} as const;

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
  onAction,
}: Props) {
  const legal = new Set(legalActions);
  const phase = explore?.phase ?? 'inactive';

  const Btn = ({ id, label, kind = 'primary' }: { id: number; label: string; kind?: 'primary' | 'secondary' | 'danger' }) =>
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
            <Btn id={ACTION.EXPLORE} label="🔭 Explore" />
            <Btn id={ACTION.PASS} label="✋ Pass" kind="secondary" />
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
            <Btn id={ACTION.EXPLORE_STOP} label="Stop exploring" kind="secondary" />
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
            <Btn id={ACTION.EXPLORE_PLACE} label="Place tile" />
            <Btn id={ACTION.EXPLORE_DISCARD} label="Discard" kind={legal.has(ACTION.EXPLORE_PLACE) ? 'danger' : 'primary'} />
          </div>
        </>
      )}

      {phase === 'choose_rotation' && (
        <>
          <span className="text-xs text-[#94a3b8]">Choose a rotation that forms a wormhole connection.</span>
          <div className="action-row action-wrap">
            {[0, 1, 2, 3, 4, 5].map((rot) => (
              <Btn key={rot} id={ACTION.EXPLORE_ROT_START + rot} label={`Rot ${rot}`} />
            ))}
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
            <Btn id={ACTION.EXPLORE_CLAIM_YES} label="Claim control" />
            <Btn id={ACTION.EXPLORE_CLAIM_NO} label={legal.has(ACTION.EXPLORE_CLAIM_YES) ? 'Decline' : 'Continue'} kind="secondary" />
          </div>
        </>
      )}

      {phase === 'discovery_reward' && (
        <>
          <span className="text-xs text-[#94a3b8]">Resolve the Discovery tile in this sector.</span>
          <div className="action-row">
            <Btn id={ACTION.EXPLORE_DISCOVERY_REWARD} label="Take reward" />
            <Btn id={ACTION.EXPLORE_DISCOVERY_VP} label="Take 2 VP" kind="secondary" />
          </div>
        </>
      )}

      {phase === 'select_drawn_tile' && (
        <>
          <span className="text-xs text-[#94a3b8]">Keep one of the drawn tiles.</span>
          <div className="action-row">
            <Btn id={ACTION.EXPLORE_SELECT_TILE_START} label={`Keep #${explore?.drawn_sector_ids?.[0] ?? 0}`} />
            <Btn id={ACTION.EXPLORE_SELECT_TILE_START + 1} label={`Keep #${explore?.drawn_sector_ids?.[1] ?? 0}`} />
          </div>
        </>
      )}

      {phase === 'draw_again_decision' && (
        <>
          <span className="text-xs text-[#94a3b8]">Draw a second tile or proceed with this one?</span>
          <div className="action-row">
            <Btn id={ACTION.EXPLORE_DRAW_AGAIN} label="Draw again" />
            <Btn id={ACTION.EXPLORE_SKIP_SECOND} label="Proceed" kind="secondary" />
          </div>
        </>
      )}
    </div>
  );
}
