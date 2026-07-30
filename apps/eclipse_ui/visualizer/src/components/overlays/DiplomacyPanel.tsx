import { useMemo, useState } from 'react';
import { ACTION, POP_TRACK_LABELS } from '../../actionTypes';
import type { DiplomacyState, GameState } from '../../types/game';
import ReputationTrack from '../ui/ReputationTrack';
import { getPlayerColor } from '../../theme';

interface Props {
  diplomacy: DiplomacyState;
  gameState: GameState;
  legalActions: number[];
  busy: boolean;
  onAction: (actionId: number) => void;
  playerLabel: (pid: number) => string;
}

export default function DiplomacyPanel({
  diplomacy,
  gameState,
  legalActions,
  busy,
  onAction,
  playerLabel,
}: Props) {
  const legal = useMemo(() => new Set(legalActions), [legalActions]);
  const phase = diplomacy.phase;
  const proposerId = diplomacy.player_id;
  const partnerId = diplomacy.partner_id;
  const currentPlayerId = gameState.current_player;
  const playerColors = gameState.players.map((_, i) => getPlayerColor(i));

  const [swapSource, setSwapSource] = useState<number | null>(null);

  const isCurrent = (pid: number) => pid === currentPlayerId;
  const isCurrentAccepted = phase === 'choose_accept' && isCurrent(partnerId);

  const decodePickTrackActions = useMemo(() => {
    const tracks: number[] = [];
    for (const a of legalActions) {
      if (a >= ACTION.DIPLOMACY_PICK_TRACK_START && a < ACTION.DIPLOMACY_RETURN_START) {
        tracks.push(a);
      }
    }
    return tracks;
  }, [legalActions]);

  const rearrangeSide = diplomacy.rearrange_side;
  const rearrangePlayerId = rearrangeSide === 0 ? proposerId : partnerId;

  const decodeReturnSlots = useMemo(() => {
    const slots = new Set<number>();
    for (const a of legalActions) {
      if (a >= ACTION.DIPLOMACY_RETURN_START && a < ACTION.DIPLOMACY_SWAP_START) {
        const encoded = a - ACTION.DIPLOMACY_RETURN_START;
        const side = Math.floor(encoded / 5);
        if (side === rearrangeSide) slots.add(encoded % 5);
      }
    }
    return slots;
  }, [legalActions, rearrangeSide]);

  const decodeSwapSources = useMemo(() => {
    const slots = new Set<number>();
    for (const a of legalActions) {
      if (a >= ACTION.DIPLOMACY_SWAP_START && a < ACTION.DIPLOMACY_ACCEPT) {
        const encoded = a - ACTION.DIPLOMACY_SWAP_START;
        const side = Math.floor(encoded / 20);
        if (side === rearrangeSide) {
          const src = Math.floor((encoded % 20) / 5);
          slots.add(src);
        }
      }
    }
    return slots;
  }, [legalActions, rearrangeSide]);

  const decodeSwapDestinations = useMemo(() => {
    const map = new Map<number, Set<number>>();
    for (const a of legalActions) {
      if (a >= ACTION.DIPLOMACY_SWAP_START && a < ACTION.DIPLOMACY_ACCEPT) {
        const encoded = a - ACTION.DIPLOMACY_SWAP_START;
        const side = Math.floor(encoded / 20);
        if (side === rearrangeSide) {
          const src = Math.floor((encoded % 20) / 5);
          const dst = encoded % 5;
          if (!map.has(src)) map.set(src, new Set());
          map.get(src)!.add(dst);
        }
      }
    }
    return map;
  }, [legalActions, rearrangeSide]);

  const rearrangePlayer = rearrangePlayerId < gameState.players.length
    ? gameState.players[rearrangePlayerId]
    : null;

  const decodeReturnTrackActions = useMemo(() => {
    const tracks: number[] = [];
    for (const a of legalActions) {
      if (a >= ACTION.CHOOSE_RETURN_TRACK_START && a < ACTION.CHOOSE_RETURN_TRACK_START + 3) {
        tracks.push(a);
      }
    }
    return tracks;
  }, [legalActions]);

  return (
    <div className="diplomacy-panel">
      <h3 className="panel-title">
        {phase === 'choose_accept' && 'Diplomatic Relations Proposal'}
        {phase === 'choose_pop_track' && 'Pick a Population Track'}
        {phase === 'choose_rearrange' && 'Free an Ambassador Slot'}
        {phase === 'choose_return_track' && 'Return Pop Cube to Track'}
      </h3>

      {phase === 'choose_accept' && (
        <div className="diplomacy-content">
          <p className="diplomacy-info">
            {playerLabel(proposerId)} proposes Diplomatic Relations with {playerLabel(partnerId)}.
          </p>
          <div className="diplomacy-buttons">
            <button
              className="btn btn-primary"
              disabled={busy || !legal.has(ACTION.DIPLOMACY_ACCEPT) || !isCurrentAccepted}
              onClick={() => onAction(ACTION.DIPLOMACY_ACCEPT)}
            >
              Accept
            </button>
            <button
              className="btn btn-danger"
              disabled={busy || !legal.has(ACTION.DIPLOMACY_DECLINE) || !isCurrentAccepted}
              onClick={() => onAction(ACTION.DIPLOMACY_DECLINE)}
            >
              Decline
            </button>
          </div>
        </div>
      )}

      {phase === 'choose_pop_track' && (
        <div className="diplomacy-content">
          <p className="diplomacy-info">
            {playerLabel(diplomacy.pop_track_side === 0 ? proposerId : partnerId)}:
            choose a track for your pop cube.
          </p>
          <div className="diplomacy-buttons">
            {decodePickTrackActions.map((actionId) => {
              const encoded = actionId - ACTION.DIPLOMACY_PICK_TRACK_START;
              const track = encoded % 3;
              return (
                <button
                  key={actionId}
                  className="btn btn-secondary"
                  disabled={busy}
                  onClick={() => onAction(actionId)}
                >
                  {POP_TRACK_LABELS[track]}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {phase === 'choose_rearrange' && (
        <div className="diplomacy-content">
          <p className="diplomacy-info">
            {playerLabel(rearrangePlayerId)} needs to free an Ambassador slot.
            {swapSource !== null ? ' Click a destination slot to swap.' : ' Click a rep tile to return it, or click a swap-source slot.'}
          </p>
          {rearrangePlayer && (
            <ReputationTrack
              track={rearrangePlayer.reputation_track}
              playerColors={playerColors}
              interactive
              side={rearrangeSide}
              legalReturns={decodeReturnSlots}
              legalSwapSources={decodeSwapSources}
              legalSwapDestinations={decodeSwapDestinations}
              selectedSwapSource={swapSource}
              onSelectSwapSource={setSwapSource}
              onAction={onAction}
            />
          )}
        </div>
      )}

      {phase === 'choose_return_track' && (
        <div className="diplomacy-content">
          <p className="diplomacy-info">
            {playerLabel(diplomacy.player_id)}: return the pop cube to a track.
          </p>
          <div className="diplomacy-buttons">
            {decodeReturnTrackActions.map((actionId) => {
              const track = actionId - ACTION.CHOOSE_RETURN_TRACK_START;
              return (
                <button
                  key={actionId}
                  className="btn btn-secondary"
                  disabled={busy}
                  onClick={() => onAction(actionId)}
                >
                  {POP_TRACK_LABELS[track]}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
