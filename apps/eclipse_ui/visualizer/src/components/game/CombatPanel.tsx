import { useMemo, useState, useRef } from 'react';
import { ACTION } from '../../actionTypes';
import { getPlayerColor } from '../../theme';
import { NPC_PLAYER_ID, NO_PLAYER, GALAXY_MAP_SIZE, GALAXY_OFFSET } from '../../constants';
import type {
  CombatState,
  DiscoveryCatalog,
  GameState,
  SectorLayout,
  Unit,
} from '../../types/game';
import DiscoveryChoice from '../ui/DiscoveryChoice';

const DIE_COLORS = ['#eab308', '#f97316', '#3b82f6', '#ef4444', '#a855f7'];
const DIE_COLOR_NAMES = ['Yellow', 'Orange', 'Blue', 'Red', 'Purple'];
const REP_VP: Record<string, number> = { One: 1, Two: 2, Three: 3, Four: 4 };
const BLUEPRINT_INDEX: Record<string, number> = {
  interceptor: 0,
  cruiser: 1,
  dreadnought: 2,
  starbase: 3,
};

interface Props {
  combat: CombatState;
  gameState: GameState;
  sectorLayouts: Record<number, SectorLayout>;
  discoveryCatalog: DiscoveryCatalog;
  legalActions: number[];
  canAct: boolean;
  busy: boolean;
  onAction: (actionId: number) => void;
  playerLabel: (pid: number) => string;
}

interface IndexedUnit {
  index: number;
  unit: Unit;
}

interface CombatEvent {
  key: string;
  text: string;
}

function isInRange(action: number, start: number, end: number) {
  return action >= start && action < end;
}

function shipLabel(type: string) {
  return type.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function combatantLabel(playerId: number, playerLabel: (pid: number) => string) {
  return playerId === NPC_PLAYER_ID ? 'NPC' : playerLabel(playerId);
}

function combatantColor(playerId: number) {
  return playerId === NPC_PLAYER_ID ? '#f59e0b' : getPlayerColor(playerId);
}

function phaseTitle(combat: CombatState) {
  switch (combat.phase) {
    case 'missile_phase': return 'Missile volley';
    case 'choose_engagement_action': return 'Engagement decision';
    case 'engagement_firing': return 'Cannon volley';
    case 'select_reputation_tile': return 'Reputation rewards';
    case 'attack_population': return 'Population bombardment';
    case 'influence_sectors': return 'Influence resolution';
    case 'discovery_award': return 'Discovery reward';
    case 'repair': return 'Repairing ships';
    case 'determine_battles': return 'Preparing battles';
    default: return 'Combat resolution';
  }
}

export default function CombatPanel({
  combat,
  gameState,
  sectorLayouts,
  discoveryCatalog,
  legalActions,
  canAct,
  busy,
  onAction,
  playerLabel,
}: Props) {
  const legal = useMemo(() => new Set(legalActions), [legalActions]);
  const [events, setEvents] = useState<CombatEvent[]>([]);

  const battleQueue = combat.battle_queue.slice(0, combat.battle_queue_size);
  const activeSector = useMemo(() => {
    for (const row of gameState.galaxy) {
      const sector = row.find((candidate) => candidate.sector_id === combat.active_sector_id);
      if (sector) return sector;
    }
    return undefined;
  }, [combat.active_sector_id, gameState.galaxy]);
  const activeUnits = useMemo(
    () => gameState.unit_registry
      .map((unit, index) => ({ unit, index }))
      .filter((entry) => entry.unit.sector_id === combat.active_sector_id),
    [combat.active_sector_id, gameState.unit_registry]
  );

  const targetActions = useMemo(() => new Map(
    legalActions
      .filter((action) => isInRange(action, ACTION.COMBAT_DICE_TARGET_START, ACTION.COMBAT_REP_SELECT_START))
      .map((action) => [action - ACTION.COMBAT_DICE_TARGET_START, action]),
  ), [legalActions]);
  const targetUnits = activeUnits.filter(({ index }) => targetActions.has(index));
  const pendingDie =
    combat.pending_target_group_player !== NO_PLAYER &&
    combat.pending_die_index < combat.pending_die_count &&
    (combat.pending_die_values[combat.pending_die_index] ?? 0) > 0;
  const populationActions = legalActions.filter((action) =>
    isInRange(action, ACTION.COMBAT_POP_TARGET_START, ACTION.COMBAT_INFLUENCE_YES)
  );
  const retreatActions = legalActions.filter((action) =>
    isInRange(action, ACTION.COMBAT_RETREAT_TO_CELL_START, ACTION.COMBAT_DICE_TARGET_START)
  );
  const reputationActions = legalActions.filter((action) =>
    isInRange(action, ACTION.COMBAT_REP_SELECT_START, ACTION.COMBAT_REP_SKIP)
  );
  const discoveryTile = useMemo(() => {
    const sectorId = combat.discovery_decision_sector;
    if (!sectorId) return undefined;
    const sector = gameState.galaxy.flat().find((candidate) => candidate.sector_id === sectorId);
    if (!sector?.discovery_tile_present || sector.discovery_tile === undefined) return undefined;
    const raw = String(sector.discovery_tile);
    return discoveryCatalog[raw] ?? Object.values(discoveryCatalog).find(
      (tile) => String(tile.id) === raw || tile.slug === raw || tile.name === raw
    );
  }, [combat.discovery_decision_sector, discoveryCatalog, gameState.galaxy]);

  const groups = useMemo(() => {
    const grouped = new Map<number, IndexedUnit[]>();
    for (const entry of activeUnits) {
      const units = grouped.get(entry.unit.player_id) ?? [];
      units.push(entry);
      grouped.set(entry.unit.player_id, units);
    }
    return [...grouped.entries()].sort(([left], [right]) => left - right);
  }, [activeUnits]);

  const eventKeyRef = useRef(0);
  const sectorForCell = (cell: number) => {
    const q = Math.floor(cell / GALAXY_MAP_SIZE) - GALAXY_OFFSET;
    const r = (cell % GALAXY_MAP_SIZE) - GALAXY_OFFSET;
    return gameState.galaxy[q + GALAXY_OFFSET]?.[r + GALAXY_OFFSET];
  };
  const submit = (action: number, eventText?: string) => {
    if (!canAct || busy || !legal.has(action)) return;
    if (eventText) {
      setEvents((current) => [{ key: `action-${++eventKeyRef.current}-${action}`, text: eventText }, ...current].slice(0, 8));
    }
    onAction(action);
  };
  const shipHull = (unit: Unit) => {
    if (unit.player_id === NPC_PLAYER_ID) return undefined;
    const index = BLUEPRINT_INDEX[unit.type.toLowerCase()];
    return index === undefined ? undefined : gameState.players[unit.player_id]?.blueprints[index]?.total_stats.hull;
  };
  const populationLabel = (slot: number) => {
    const planet = sectorLayouts[combat.pop_attack_sector_id]?.planets.find((candidate) => candidate.slot === slot);
    return planet ? `${planet.type} population` : `Population slot ${slot + 1}`;
  };
  const waitingPlayer = combat.pending_player !== NO_PLAYER
    ? combat.pending_player
    : combat.pending_target_group_player !== NO_PLAYER && pendingDie
      ? combat.pending_target_group_player
      : combat.tile_select_player !== NO_PLAYER
        ? combat.tile_select_player
        : combat.influence_decision_player !== NO_PLAYER
          ? combat.influence_decision_player
          : combat.discovery_decision_player;

  return (
    <section className="combat-screen" aria-label="Combat">
      <header className="combat-screen-header">
        <div>
          <p className="combat-screen-eyebrow">Combat phase</p>
          <h1>Sector {combat.active_sector_id || '—'} · {phaseTitle(combat)}</h1>
          <p className="combat-screen-subtitle">
            {combat.active_sector_id > 0 && `Battle ${combat.current_battle_idx + 1} of ${combat.battle_queue_size}`}
            {combat.engagement_round > 0 && ` · Engagement round ${combat.engagement_round}`}
          </p>
        </div>
        <div className="combat-screen-status">
          {waitingPlayer !== NO_PLAYER ? (
            <span>
              {canAct ? 'Your decision' : `Waiting for ${combatantLabel(waitingPlayer, playerLabel)}`}
            </span>
          ) : <span>Resolving automatically…</span>}
        </div>
      </header>

      <div className="combat-queue" aria-label="Combat sector order">
        {battleQueue.map((battle, index) => (
          <div
            key={`${battle.sector_id}-${index}`}
            className={`combat-queue-item ${index === combat.current_battle_idx ? 'active' : ''} ${index < combat.current_battle_idx ? 'complete' : ''}`}
          >
            <span>{index + 1}</span>
            Sector {battle.sector_id}
          </div>
        ))}
        {battleQueue.length === 0 && <span className="combat-queue-empty">Post-battle resolution</span>}
      </div>

      <main className="combat-screen-content">
        <section className="combat-battlefield" aria-label="Active sector ships">
          <div className="combat-sector-details">
            <strong>Sector {activeSector?.sector_id ?? combat.active_sector_id}</strong>
            {activeSector && <span>Coordinates {activeSector.coords.q}, {activeSector.coords.r}</span>}
            {combat.current_attacker_id !== NO_PLAYER && combat.current_defender_id !== NO_PLAYER && (
              <span>
                {combatantLabel(combat.current_attacker_id, playerLabel)} attacks {combatantLabel(combat.current_defender_id, playerLabel)}
              </span>
            )}
          </div>

          <div className="combat-rosters">
            {groups.map(([playerId, units]) => (
              <article className="combat-roster" key={playerId} style={{ borderColor: combatantColor(playerId) }}>
                <header style={{ color: combatantColor(playerId) }}>
                  {combatantLabel(playerId, playerLabel)}
                  {playerId === NPC_PLAYER_ID && <small> automatic</small>}
                </header>
                <div className="combat-ship-list">
                  {units.map(({ index, unit }, position) => {
                    const action = targetActions.get(index);
                    const hull = shipHull(unit);
                    return (
                      <button
                        type="button"
                        key={index}
                        className={`combat-ship-card ${action !== undefined ? 'targetable' : ''}`}
                        disabled={action === undefined || !canAct || busy}
                        onClick={() => action !== undefined && submit(action, `Targeted ${shipLabel(unit.type)} ${position + 1}.`)}
                      >
                        <span>{shipLabel(unit.type)} {position + 1}</span>
                        <small>
                          {unit.damage} damage{hull !== undefined ? ` / ${hull} hull` : ''}
                        </small>
                      </button>
                    );
                  })}
                </div>
              </article>
            ))}
            {groups.length === 0 && <p className="combat-empty-state">No ships remain in this sector.</p>}
          </div>

          {combat.initiative_size > 0 && (
            <div className="combat-timeline" aria-label="Initiative timeline">
              {combat.initiative_timeline.slice(0, combat.initiative_size).map((group, index) => (
                <span
                  key={`${group.player_id}-${group.type}-${index}`}
                  className={`${index === combat.initiative_idx ? 'active' : ''} ${group.destroyed || group.alive_count === 0 ? 'destroyed' : ''} ${group.retreating ? 'retreating' : ''}`}
                  style={{ borderColor: combatantColor(group.player_id) }}
                >
                  {group.initiative} · {shipLabel(group.type)} ×{group.alive_count}
                </span>
              ))}
            </div>
          )}

          {combat.destroyed_ships_size > 0 && (
            <div className="combat-destroyed-summary">
              <strong>Destroyed this battle</strong>
              {combat.destroyed_ships.slice(0, combat.destroyed_ships_size).map((record, index) => (
                <span key={`${record.player_id}-${record.type}-${index}`}>
                  {record.count}× {shipLabel(record.type)} · {combatantLabel(record.player_id, playerLabel)}
                </span>
              ))}
            </div>
          )}
        </section>

        <aside className="combat-decision-panel" aria-live="polite">
          {pendingDie && (
            <div className="combat-decision-block">
              <h2>Assign this die</h2>
              <div className="combat-dice-row">
                {combat.pending_die_values.slice(0, combat.pending_die_count).map((value, index) => (
                  <span
                    key={index}
                    className={index === combat.pending_die_index ? 'active' : ''}
                    style={{ background: DIE_COLORS[combat.pending_die_colors[index] ?? 0] ?? '#64748b' }}
                    title={`${DIE_COLOR_NAMES[combat.pending_die_colors[index] ?? 0] ?? 'Unknown'} die`}
                  >
                    {value || '•'}
                  </span>
                ))}
              </div>
              <p>
                {DIE_COLOR_NAMES[combat.pending_die_colors[combat.pending_die_index] ?? 0] ?? 'Unknown'}{' '}
                {combat.pending_dice_are_missiles ? 'missile' : 'cannon'} die: select one highlighted ship.
              </p>
              {targetUnits.length === 0 && <p className="combat-note">Waiting for a legal target.</p>}
            </div>
          )}

          {combat.phase === 'choose_engagement_action' && legal.has(ACTION.COMBAT_ATTACK) && (
            <div className="combat-decision-block">
              <h2>{shipLabel(combat.active_ship_type)} action</h2>
              <button className="combat-action primary" disabled={!canAct || busy} onClick={() => submit(ACTION.COMBAT_ATTACK, 'Chose to attack.')}>Attack</button>
              {retreatActions.length > 0 && <p className="combat-choice-label">Or retreat this group</p>}
              {retreatActions.map((action) => {
                const sector = sectorForCell(action - ACTION.COMBAT_RETREAT_TO_CELL_START);
                return (
                  <button className="combat-action secondary" key={action} disabled={!canAct || busy} onClick={() => submit(action, `Retreating to Sector ${sector?.sector_id ?? '?'}.`)}>
                    Retreat to Sector {sector?.sector_id ?? '?'}{sector ? ` (${sector.coords.q}, ${sector.coords.r})` : ''}
                  </button>
                );
              })}
            </div>
          )}

          {combat.phase === 'select_reputation_tile' && reputationActions.length > 0 && (
            <div className="combat-decision-block">
              <h2>Choose one reputation tile</h2>
              {reputationActions.map((action) => {
                const tile = combat.drawn_tiles[action - ACTION.COMBAT_REP_SELECT_START];
                return <button className="combat-action primary" key={action} disabled={!canAct || busy} onClick={() => submit(action, `Kept a ${REP_VP[tile] ?? '?'} VP reputation tile.`)}>Keep {REP_VP[tile] ?? '?'} VP tile</button>;
              })}
              {legal.has(ACTION.COMBAT_REP_SKIP) && <button className="combat-action secondary" disabled={!canAct || busy} onClick={() => submit(ACTION.COMBAT_REP_SKIP, 'Skipped reputation tiles.')}>Skip</button>}
            </div>
          )}

          {combat.phase === 'attack_population' && populationActions.length > 0 && (
            <div className="combat-decision-block">
              <h2>Destroy population</h2>
              <p>{combat.pop_attack_damage_remaining} damage remaining in Sector {combat.pop_attack_sector_id}.</p>
              {populationActions.map((action) => {
                const slot = action - ACTION.COMBAT_POP_TARGET_START;
                return <button className="combat-action danger" key={action} disabled={!canAct || busy} onClick={() => submit(action, `Destroyed ${populationLabel(slot)}.`)}>{populationLabel(slot)}</button>;
              })}
            </div>
          )}

          {combat.phase === 'influence_sectors' && (legal.has(ACTION.COMBAT_INFLUENCE_YES) || legal.has(ACTION.COMBAT_INFLUENCE_NO)) && (
            <div className="combat-decision-block">
              <h2>Claim Sector {combat.influence_decision_sector}</h2>
              <p>Place an influence disc if you want to control this sector.</p>
              {legal.has(ACTION.COMBAT_INFLUENCE_YES) && <button className="combat-action primary" disabled={!canAct || busy} onClick={() => submit(ACTION.COMBAT_INFLUENCE_YES, 'Placed an influence disc.')}>Place disc</button>}
              {legal.has(ACTION.COMBAT_INFLUENCE_NO) && <button className="combat-action secondary" disabled={!canAct || busy} onClick={() => submit(ACTION.COMBAT_INFLUENCE_NO, 'Declined to place an influence disc.')}>Decline</button>}
            </div>
          )}

          {combat.phase === 'discovery_award' && (legal.has(ACTION.COMBAT_DISCOVERY_REWARD) || legal.has(ACTION.COMBAT_DISCOVERY_VP)) && (
            <div className="combat-decision-block">
              <h2>Discovery in Sector {combat.discovery_decision_sector}</h2>
              <DiscoveryChoice
                tile={discoveryTile}
                legalActions={legalActions}
                busy={!canAct || busy}
                rewardActionId={ACTION.COMBAT_DISCOVERY_REWARD}
                vpActionId={ACTION.COMBAT_DISCOVERY_VP}
                onAction={(action) => submit(action, action === ACTION.COMBAT_DISCOVERY_VP ? 'Kept 2 discovery VP.' : 'Took discovery reward.')}
              />
            </div>
          )}

          {legal.has(ACTION.COMBAT_CONTINUE) && (
            <div className="combat-decision-block">
              <h2>{canAct ? 'Ready to continue' : 'Automatic resolution'}</h2>
              <p>{canAct ? 'Advance the current combat step.' : 'The server is resolving this step.'}</p>
              <button className="combat-action primary" disabled={!canAct || busy} onClick={() => submit(ACTION.COMBAT_CONTINUE, 'Continued combat resolution.')}>Continue</button>
            </div>
          )}

          {!canAct && waitingPlayer !== NO_PLAYER && <p className="combat-note">Only the current player can submit this decision.</p>}

          {events.length > 0 && (
            <div className="combat-event-log">
              <h2>Recent events</h2>
              {events.map((event) => <p key={event.key}>{event.text}</p>)}
            </div>
          )}
        </aside>
      </main>
    </section>
  );
}
