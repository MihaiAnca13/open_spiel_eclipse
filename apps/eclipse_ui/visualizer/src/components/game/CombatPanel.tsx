// Combat phase controls. Driven by the server-supplied combat_state + legal
// actions, mirroring EclipseState::CombatLegalActions (eclipse.cc). Every legal
// action that isn't given an explicit button still gets rendered through the
// action_strings fallback, so a combat phase can never soft-lock the UI.

import { useCallback, useMemo } from 'react';
import { ACTION } from '../../actionTypes';
import { getPlayerColor } from '../../theme';
import type { CombatState, DiscoveryCatalog, GameState } from '../../types/game';
import DiscoveryChoice from '../ui/DiscoveryChoice';

const NO_PLAYER = 254;

// DieColor: YELLOW, ORANGE, BLUE, RED, PURPLE (dice.h).
const DIE_COLORS = ['#eab308', '#f97316', '#3b82f6', '#ef4444', '#a855f7'];
const DIE_COLOR_NAMES = ['Yellow', 'Orange', 'Blue', 'Red', 'Purple'];

// ReputationTiles serialize as One..Four (types.h); the ordinal is the VP value.
const REP_VP: Record<string, number> = { One: 1, Two: 2, Three: 3, Four: 4 };

interface Props {
  combat: CombatState;
  gameState: GameState;
  discoveryCatalog: DiscoveryCatalog;
  legalActions: number[];
  actionStrings?: Record<string, string>;
  busy: boolean;
  onAction: (actionId: number) => void;
  playerLabel: (pid: number) => string;
}

interface DecisionButton {
  id: number;
  label: string;
  kind: 'primary' | 'secondary' | 'danger';
}

export default function CombatPanel({
  combat,
  gameState,
  discoveryCatalog,
  legalActions,
  actionStrings,
  busy,
  onAction,
  playerLabel,
}: Props) {
  const legal = useMemo(() => new Set(legalActions), [legalActions]);

  // Resolve a galaxy cell index to its sector id (cell = (q+7)*15 + (r+7)).
  const cellToSectorId = useCallback((cell: number): number => {
    const qIdx = Math.floor(cell / 15);
    const rIdx = cell % 15;
    return gameState.galaxy?.[qIdx]?.[rIdx]?.sector_id ?? 0;
  }, [gameState.galaxy]);

  const cs = combat;
  const discoveryTile = useMemo(() => {
    if (!cs.discovery_decision_sector) return undefined;
    for (const row of gameState.galaxy ?? []) {
      const sector = row.find((candidate) => candidate.sector_id === cs.discovery_decision_sector);
      if (!sector) continue;
      if (sector.discovery_tile === undefined || sector.discovery_tile === 0) return undefined;
      return discoveryCatalog[String(sector.discovery_tile)];
    }
    return undefined;
  }, [cs.discovery_decision_sector, discoveryCatalog, gameState.galaxy]);
  const pendingDie =
    cs.pending_target_group_player !== NO_PLAYER &&
    cs.pending_die_index < cs.pending_die_count &&
    (cs.pending_die_values?.[cs.pending_die_index] ?? 0) !== 0;
  const popAttack = cs.phase === 'attack_population' && cs.pop_attack_damage_remaining > 0;

  const inRange = (id: number, start: number, end: number) => id >= start && id < end;

  // Build the explicit decision buttons for whatever the player must decide now,
  // honoring CombatLegalActions ordering (pending die → pop attack → phase).
  const decision: DecisionButton[] = useMemo(() => {
    const out: DecisionButton[] = [];
    const push = (id: number, label: string, kind: DecisionButton['kind'] = 'primary') => {
      if (legal.has(id)) out.push({ id, label, kind });
    };

    if (pendingDie) {
      for (const id of legalActions) {
        if (!inRange(id, ACTION.COMBAT_DICE_TARGET_START, ACTION.COMBAT_REP_SELECT_START)) continue;
        const unitIdx = id - ACTION.COMBAT_DICE_TARGET_START;
        const unit = gameState.unit_registry?.[unitIdx];
        const label = unit
          ? `🎯 ${unit.type} — ${playerLabel(unit.player_id)}${unit.damage ? ` (${unit.damage} dmg)` : ''}`
          : `🎯 Unit ${unitIdx}`;
        push(id, label);
      }
      return out;
    }

    if (popAttack) {
      for (const id of legalActions) {
        if (!inRange(id, ACTION.COMBAT_POP_TARGET_START, ACTION.COMBAT_INFLUENCE_YES)) continue;
        const slot = id - ACTION.COMBAT_POP_TARGET_START;
        push(id, `💥 Destroy population (slot ${slot})`, 'danger');
      }
      return out;
    }

    switch (cs.phase) {
      case 'choose_engagement_action':
        push(ACTION.COMBAT_ATTACK, '⚔️ Attack', 'primary');
        for (const id of legalActions) {
          if (!inRange(id, ACTION.COMBAT_RETREAT_TO_CELL_START, ACTION.COMBAT_DICE_TARGET_START)) continue;
          const sid = cellToSectorId(id - ACTION.COMBAT_RETREAT_TO_CELL_START);
          push(id, `🏳️ Retreat to Sector ${sid || '?'}`, 'secondary');
        }
        push(ACTION.COMBAT_CONTINUE, 'Continue', 'secondary');
        break;
      case 'select_reputation_tile':
        for (const id of legalActions) {
          if (!inRange(id, ACTION.COMBAT_REP_SELECT_START, ACTION.COMBAT_REP_SKIP)) continue;
          const tile = cs.drawn_tiles?.[id - ACTION.COMBAT_REP_SELECT_START];
          const vp = tile != null ? REP_VP[tile] : undefined;
          push(id, `🏅 Keep tile${vp ? ` (+${vp} VP)` : ''}`, 'primary');
        }
        push(ACTION.COMBAT_REP_SKIP, 'Skip', 'secondary');
        break;
      case 'influence_sectors':
        push(ACTION.COMBAT_INFLUENCE_YES, `🔵 Place disc in Sector ${cs.influence_decision_sector || '?'}`, 'primary');
        push(ACTION.COMBAT_INFLUENCE_NO, 'Decline', 'secondary');
        push(ACTION.COMBAT_CONTINUE, 'Continue', 'secondary');
        break;
      case 'discovery_award':
        push(ACTION.COMBAT_DISCOVERY_REWARD, '🎁 Take reward', 'primary');
        push(ACTION.COMBAT_DISCOVERY_VP, 'Take 2 VP', 'secondary');
        push(ACTION.COMBAT_CONTINUE, 'Continue', 'secondary');
        break;
      default:
        push(ACTION.COMBAT_CONTINUE, 'Continue', 'primary');
        break;
    }
    return out;
  }, [cellToSectorId, cs, pendingDie, popAttack, legalActions, legal, gameState, playerLabel]);

  // Any legal action not covered by an explicit decision button — the safety net.
  const handledIds = useMemo(() => new Set(decision.map((d) => d.id)), [decision]);
  const fallback = useMemo(
    () => legalActions.filter((id) => !handledIds.has(id)),
    [legalActions, handledIds]
  );

  const phaseLabel: Record<string, string> = {
    inactive: 'Inactive',
    determine_battles: 'Determining battles',
    missile_phase: 'Missile phase',
    choose_engagement_action: 'Engagement: attack or retreat',
    engagement_firing: 'Firing',
    select_reputation_tile: 'Reputation tiles',
    attack_population: 'Attacking population',
    influence_sectors: 'Reclaiming control',
    discovery_award: 'Discovery tile',
    repair: 'Repairing',
  };

  const die =
    pendingDie
      ? {
          value: cs.pending_die_values[cs.pending_die_index],
          colorIdx: cs.pending_die_colors?.[cs.pending_die_index] ?? 0,
        }
      : null;

  const recentDestroyed = (cs.destroyed_ships ?? []).slice(0, Math.max(0, cs.destroyed_ships_size)).slice(-4);
  const isDiscoveryDecision =
    cs.phase === 'discovery_award' &&
    (legal.has(ACTION.COMBAT_DISCOVERY_REWARD) || legal.has(ACTION.COMBAT_DISCOVERY_VP));

  return (
    <div className="panel action-panel">
      <h3 className="panel-title">⚔️ Combat</h3>

      {/* Battle header */}
      <div className="text-xs text-[#cbd5e1]" style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        <span>
          <strong>Sector {cs.active_sector_id}</strong> · {phaseLabel[cs.phase] ?? cs.phase}
          {cs.engagement_round > 0 ? ` · round ${cs.engagement_round}` : ''}
        </span>
        {(cs.current_attacker_id !== NO_PLAYER || cs.current_defender_id !== NO_PLAYER) && (
          <span className="text-[#94a3b8]">
            {cs.current_attacker_id !== NO_PLAYER && (
              <span style={{ color: getPlayerColor(cs.current_attacker_id) }}>
                {playerLabel(cs.current_attacker_id)} (attacker)
              </span>
            )}
            {cs.current_attacker_id !== NO_PLAYER && cs.current_defender_id !== NO_PLAYER && ' vs '}
            {cs.current_defender_id !== NO_PLAYER && (
              <span style={{ color: getPlayerColor(cs.current_defender_id) }}>
                {playerLabel(cs.current_defender_id)} (defender)
              </span>
            )}
          </span>
        )}
      </div>

      {/* Initiative timeline */}
      {cs.initiative_size > 0 && (
        <div className="combat-initiative" style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginTop: 4 }}>
          {cs.initiative_timeline.slice(0, cs.initiative_size).map((g, i) => (
            <span
              key={i}
              title={`Initiative ${g.initiative}${g.is_npc ? ' · NPC' : ''}${g.retreating ? ' · retreating' : ''}`}
              style={{
                fontSize: 9,
                padding: '1px 4px',
                borderRadius: 4,
                border: `1px solid ${getPlayerColor(g.player_id)}`,
                color: getPlayerColor(g.player_id),
                opacity: g.destroyed || g.alive_count === 0 ? 0.35 : i === cs.initiative_idx ? 1 : 0.8,
                textDecoration: g.retreating ? 'line-through' : 'none',
                background: i === cs.initiative_idx ? 'rgba(255,255,255,0.08)' : 'transparent',
              }}
            >
              {g.type} ×{g.alive_count}
            </span>
          ))}
        </div>
      )}

      {/* Pending die readout */}
      {die && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 6 }}>
          <span
            style={{
              width: 22,
              height: 22,
              borderRadius: 4,
              background: DIE_COLORS[die.colorIdx] ?? '#64748b',
              color: '#0f172a',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 12,
            }}
          >
            {die.value}
          </span>
          <span className="text-xs text-[#94a3b8]">
            {DIE_COLOR_NAMES[die.colorIdx] ?? '?'} {cs.pending_dice_are_missiles ? 'missile' : 'cannon'} — assign to a
            target ship.
          </span>
        </div>
      )}

      {/* Decision buttons */}
      {isDiscoveryDecision ? (
        <DiscoveryChoice
          tile={discoveryTile}
          legalActions={legalActions}
          busy={busy}
          rewardActionId={ACTION.COMBAT_DISCOVERY_REWARD}
          vpActionId={ACTION.COMBAT_DISCOVERY_VP}
          onAction={onAction}
        />
      ) : decision.length > 0 && (
        <div className="action-row" style={{ flexDirection: 'column', gap: 4, marginTop: 6 }}>
          {decision.map((b) => (
            <button
              key={b.id}
              className={`action-btn ${b.kind}`}
              disabled={busy}
              onClick={() => onAction(b.id)}
            >
              {b.label}
            </button>
          ))}
        </div>
      )}

      {/* Safety-net fallback for any unhandled legal action */}
      {fallback.length > 0 && (
        <div className="action-row" style={{ flexDirection: 'column', gap: 4, marginTop: 6 }}>
          {fallback.map((id) => (
            <button key={id} className="action-btn secondary" disabled={busy} onClick={() => onAction(id)}>
              {actionStrings?.[String(id)] ?? `Action ${id}`}
            </button>
          ))}
        </div>
      )}

      {/* Recently destroyed ships */}
      {recentDestroyed.length > 0 && (
        <div className="text-xs text-[#94a3b8]" style={{ marginTop: 6 }}>
          Destroyed:{' '}
          {recentDestroyed
            .map((d) => `${d.count}× ${d.type} (${playerLabel(d.player_id)})`)
            .join(', ')}
        </div>
      )}
    </div>
  );
}
