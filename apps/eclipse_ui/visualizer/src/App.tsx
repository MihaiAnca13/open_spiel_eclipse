import { useEffect, useMemo, useState } from 'react';
import './App.css';
import {
  SPECIES_THEME,
  UNOWNED_COLOR,
  NPC_COLOR_GCDS,
  NPC_COLOR_GUARDIAN,
  getSpeciesHexColor,
} from './theme';
import ActionPanel, { ACTION, type ExploreState } from './ActionPanel';

import { API_BASE, WS_BASE } from './types/lobby';

interface HexCoord {
  q: number;
  r: number;
}

interface Resources {
  gold: number;
  science: number;
  materials: number;
  gold_prod: number;
  science_prod: number;
  materials_prod: number;
}

interface ShipStats {
  initiative: number;
  computer: number;
  shield: number;
  energy_net: number;
  hull: number;
  movement: number;
  cannons: number[];
  missiles: number[];
}

interface Blueprint {
  slots: string[];
  capacity: number;
  base_stats: ShipStats;
  total_stats: ShipStats;
}

interface Player {
  id: number;
  score: number;
  species_id: string;
  is_ai: boolean;
  has_passed: boolean;
  disks_on_sectors: number;
  disks_on_actions: number;
  resources: Resources;
  colony_ships_total: number;
  colony_ships_available: number;
  orbitals: number;
  monoliths: number;
  blueprints: Blueprint[];
  reputation_tiles: string[];
  trade_rate: number;
  researched_techs: number;
}

interface Sector {
  sector_id: number;
  owner_id: number;
  coords: HexCoord;
  rotation: number;
  points: number;
  occupied_slots_mask: number;
  discovery_tile_present: boolean;
  orbital_built: boolean;
  monolith_built: boolean;
}

interface Unit {
  player_id: number;
  type: string;
  sector_id: number;
  damage: number;
}

interface GameState {
  players: Player[];
  galaxy: Sector[][];
  reputation_tiles: string[];
  unit_registry: Unit[];
  tech_tray: Record<string, {
    count: number;
    category: string;
    base_cost: number;
    min_cost: number;
    copies: number;
  }>;
  tech_bag: string[];
  gcds_difficulty: string;
  guardian_difficulty: string;
  ancient_difficulty: string;
  current_player: number;
  current_phase: number;
  current_round: number;
  turn_order: number[];
  pass_order: number[];
  explore_state?: ExploreState;
  sector_bag_inner: number;
  sector_bag_middle: number;
  sector_bag_outer: number;
}

interface StagedPlayerConfig {
  species: string | null;
  is_ai: boolean;
  starting_sector?: number;
}

interface SetupConfig {
  players: number;
  rng_seed: number;
  npc_difficulty: string;
  staged_players: StagedPlayerConfig[];
}

export interface SetupSnapshot {
  config: SetupConfig;
  state: GameState;
  finalized: boolean;
  // Attached by the server once the game has started (see api/main.py).
  legal_actions?: number[];
  action_strings?: Record<string, string>;
  current_player?: number;
  is_terminal?: boolean;
}

// Remaining tiles in a ring bag = set bits in its bitmask.
function bagCount(mask: number | undefined): number {
  let m = (mask ?? 0) >>> 0;
  let c = 0;
  while (m) { m &= m - 1; c++; }
  return c;
}

// Influence disc model — mirrors open_spiel/games/eclipse: total discs and the
// upkeep cost exposed as discs leave the track (state.h INFLUENCE_UPKEEP_TABLE).
const INFLUENCE_TOTAL = 12;
const INFLUENCE_UPKEEP = [0, 0, 1, 2, 3, 5, 7, 10, 13, 17, 21, 25, 30];

function InfluenceTrack({ onSectors, onActions }: { onSectors: number; onActions: number }) {
  const deployed = onSectors + onActions;
  const available = Math.max(0, INFLUENCE_TOTAL - deployed);
  const upkeep = INFLUENCE_UPKEEP[Math.min(deployed, INFLUENCE_UPKEEP.length - 1)];
  // Discs fill the track from the left; spent discs leave gaps and expose the
  // upkeep cost. Order: available (on track) → on actions → on sectors.
  const cells = Array.from({ length: INFLUENCE_TOTAL }, (_, i) => {
    if (i < available) return 'avail';
    if (i < available + onActions) return 'action';
    return 'sector';
  });
  return (
    <div className="influence-track">
      <div className="influence-track-head">
        <span>Influence Track</span>
        <span className="influence-upkeep" title="Money paid each Upkeep phase">Upkeep 💰{upkeep}</span>
      </div>
      <div className="influence-pips">
        {cells.map((kind, i) => (
          <span key={i} className={`influence-pip ${kind}`} />
        ))}
      </div>
      <div className="influence-legend">
        <span><span className="influence-pip avail" /> {available} ready</span>
        <span><span className="influence-pip action" /> {onActions} actions</span>
        <span><span className="influence-pip sector" /> {onSectors} sectors</span>
      </div>
    </div>
  );
}

function App({ initialMetadata, initialSnapshot, mySeatIdx = -1, playerNames = [] }: { initialMetadata: any; initialSnapshot?: SetupSnapshot; mySeatIdx?: number; playerNames?: (string | null)[] }) {
  const playerLabel = (pid: number) => playerNames[pid] || `Player ${pid + 1}`;
  const [rngSeed, setRngSeed] = useState<number>(initialSnapshot?.config.rng_seed ?? 42);
  const [numPlayers, setNumPlayers] = useState<number>(initialSnapshot?.config.players ?? 4);
  const [gameMetadata] = useState<any>(initialMetadata);
  const [difficulty, setDifficulty] = useState<string>(initialMetadata.npc_difficulties?.[0] ?? 'Easy');
  const [snapshot, setSnapshot] = useState<SetupSnapshot | null>(initialSnapshot ?? null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [jsonExpanded, setJsonExpanded] = useState<boolean>(false);
  const [actionInProgress, setActionInProgress] = useState<boolean>(false);
  const [hoveredSector, setHoveredSector] = useState<Sector | null>(null);
  const [setupFinalized, setSetupFinalized] = useState<boolean>(initialSnapshot?.finalized ?? false);
  const [playerStartingSectors, setPlayerStartingSectors] = useState<Record<number, number>>({});

  const speciesList: string[] = gameMetadata.species ?? Object.keys(SPECIES_THEME);
  const [speciesChoices, setSpeciesChoices] = useState<Record<number, string>>(() => {
    if (initialSnapshot) {
      return Object.fromEntries(
        initialSnapshot.config.staged_players.map((p, i) => [i, p.species ?? speciesList[i % speciesList.length]])
      );
    }
    const choices: Record<number, string> = {};
    for (let i = 0; i < 6; i++) {
      choices[i] = speciesList[i % speciesList.length];
    }
    return choices;
  });
  const [aiChoices, setAiChoices] = useState<Record<number, boolean>>(() => {
    if (initialSnapshot) {
      return Object.fromEntries(
        initialSnapshot.config.staged_players.map((p, i) => [i, p.is_ai])
      );
    }
    return Object.fromEntries(Array.from({ length: 6 }, (_, i) => [i, i !== 0]));
  });

  const getAiDefault = (playerId: number) => aiChoices[playerId] ?? (playerId !== 0);

  const gameState = snapshot?.state ?? null;

  const getSpeciesOptions = (playerId: number) => {
    const selectedSpecies = Array.from({ length: numPlayers }, (_, i) => i)
      .filter(id => id !== playerId)
      .map(id => speciesChoices[id]);
    
    return speciesList.map((name: string) => {
      const isTerran = SPECIES_THEME[name]?.isTerran;
      const isAlienAndTaken = !isTerran && selectedSpecies.includes(name);
      
      return {
        value: name,
        label: SPECIES_THEME[name]?.displayLabel ?? name,
        disabled: isAlienAndTaken,
      };
    });
  };

  const stagedPlayers = useMemo(
    () =>
      Array.from({ length: numPlayers }, (_, playerId) => ({
        species: speciesChoices[playerId] || speciesList[playerId % speciesList.length],
        is_ai: getAiDefault(playerId)
      })),
    [aiChoices, numPlayers, speciesChoices, speciesList]
  );

  const handleInitializeStage1 = async () => {
    setLoading(true);
    setError(null);
    try {
      const config: SetupConfig = {
        players: numPlayers,
        rng_seed: rngSeed,
        npc_difficulty: difficulty,
        staged_players: stagedPlayers
      };
      const response = await fetch(`${API_BASE}/setup/pre-choice`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (!response.ok) {
        throw new Error('FastAPI server is not responding. Make sure apps/eclipse_ui/run.sh or the backend is running.');
      }
      const data = await response.json();
      setSnapshot(data);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleFinalizeStage2 = async () => {
    if (!snapshot) return;
    setLoading(true);
    setError(null);

    const playerChoices = Array.from({ length: numPlayers }, (_, playerId) => ({
      species: speciesChoices[playerId] || speciesList[playerId % speciesList.length],
      is_ai: getAiDefault(playerId)
    }));

    try {
      const response = await fetch(`${API_BASE}/setup/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          snapshot,
          player_choices: playerChoices
        })
      });
      if (!response.ok) {
        throw new Error('Stage 2 setup failed. Check FastAPI/native bindings.');
      }
      const data = await response.json();
      setSnapshot(data);
      setSetupFinalized(true);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSpeciesChange = (playerId: number, species: string) => {
    setSpeciesChoices((prev) => ({ ...prev, [playerId]: species }));
    
    // Assign random starting sector for terran players
    if (SPECIES_THEME[species]?.isTerran) {
      const terranSectors = [221, 223, 225, 227, 229, 231];
      const takenSectors = Object.values(playerStartingSectors);
      const availableSectors = terranSectors.filter(s => !takenSectors.includes(s));
      if (availableSectors.length > 0) {
        const randomSector = availableSectors[Math.floor(Math.random() * availableSectors.length)];
        setPlayerStartingSectors(prev => ({ ...prev, [playerId]: randomSector }));
      }
    } else {
      // Remove starting sector assignment for non-terran
      setPlayerStartingSectors(prev => {
        const newSectors = { ...prev };
        delete newSectors[playerId];
        return newSectors;
      });
    }
  };

  const handleAiChange = (playerId: number, isAi: boolean) => {
    setAiChoices((prev) => ({ ...prev, [playerId]: isAi }));
  };

  // Once the game has started, subscribe to live state broadcasts so every
  // player's action (and the server's AI auto-play) re-renders the board.
  useEffect(() => {
    if (!setupFinalized) return;
    const playerId = sessionStorage.getItem('eclipse_player_id') ?? '';
    let ws: WebSocket | null = null;
    let closed = false;
    const connect = () => {
      ws = new WebSocket(`${WS_BASE}/ws?player_id=${encodeURIComponent(playerId)}`);
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'lobby_state' && msg.lobby?.phase === 'started' && msg.lobby.snapshot) {
          setSnapshot(msg.lobby.snapshot as SetupSnapshot);
        }
      };
      ws.onclose = () => {
        if (!closed) setTimeout(connect, 1000);
      };
    };
    connect();
    return () => {
      closed = true;
      ws?.close();
    };
  }, [setupFinalized]);

  const submitAction = async (actionId: number) => {
    if (actionInProgress) return;
    setActionInProgress(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/game/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seat: mySeatIdx, action_id: actionId }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `Action failed (${response.status})`);
      }
      const updated = await response.json();
      if (updated) setSnapshot(updated as SetupSnapshot);
    } catch (err: any) {
      setError(err.message || 'Action failed');
    } finally {
      setActionInProgress(false);
    }
  };

  const hexSize = 35;
  const viewBoxWidth = 600;
  const viewBoxHeight = 520;
  const centerX = viewBoxWidth / 2;
  const centerY = viewBoxHeight / 2;

  const getHexPoints = (cx: number, cy: number, r: number) => {
    const points = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 180) * (30 + 60 * i);
      points.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
    }
    return points.join(' ');
  };

  const getPlayerHexColor = (ownerId: number, sectorId: number) => {
    if (ownerId === 255) {
      if (sectorId >= 271 && sectorId <= 274) return NPC_COLOR_GUARDIAN;
      if (sectorId === 1) return NPC_COLOR_GCDS;
      return UNOWNED_COLOR;
    }
    const player = gameState?.players[ownerId];
    return getSpeciesHexColor(player?.species_id);
  };

  const getUnitsInSector = (sectorId: number) => {
    if (!gameState) return [];
    return gameState.unit_registry.filter((unit) => unit.sector_id === sectorId);
  };

  const activeSectors: { sector: Sector; cx: number; cy: number }[] = [];
  if (gameState?.galaxy) {
    const mapSize = gameState.galaxy.length;
    const offset = 7;
    for (let qIdx = 0; qIdx < mapSize; qIdx++) {
      for (let rIdx = 0; rIdx < mapSize; rIdx++) {
        const sector = gameState.galaxy[qIdx][rIdx];
        if (sector && sector.sector_id > 0) {
          const q = qIdx - offset;
          const r = rIdx - offset;
          const cx = centerX + hexSize * (Math.sqrt(3) * q + (Math.sqrt(3) / 2) * r);
          const cy = centerY + hexSize * (1.5 * r);
          activeSectors.push({ sector, cx, cy });
        }
      }
    }
  }

  const playerIds = Array.from({ length: numPlayers }, (_, playerId) => playerId);

  // ── Gameplay (Action phase) derived state ──────────────────────────────────
  const legalActions = snapshot?.legal_actions ?? [];
  const exploreState = gameState?.explore_state;
  const explorePhase = exploreState?.phase ?? 'inactive';
  const isTerminal = snapshot?.is_terminal ?? false;
  const isStarted = setupFinalized && snapshot?.current_player !== undefined;
  const isMyTurn = isStarted && snapshot?.current_player === mySeatIdx && !isTerminal;
  const inZoneSelect = isMyTurn && explorePhase === 'choose_zone';
  const currentPlayerLabel =
    snapshot?.current_player !== undefined ? playerLabel(snapshot.current_player) : '';
  const selectedSectorId = exploreState?.selected_sector_id ?? 0;
  const ancientsOnSelected =
    !!gameState &&
    selectedSectorId > 0 &&
    gameState.unit_registry.some((u) => u.sector_id === selectedSectorId && u.player_id === 255);

  // Legal explore zones → clickable hexes (action id 35 + hex_to_index(q,r)).
  const legalZones = inZoneSelect
    ? legalActions
        .filter((a) => a >= ACTION.EXPLORE_ZONE_START)
        .map((a) => {
          const idx = a - ACTION.EXPLORE_ZONE_START;
          const q = Math.floor(idx / 15) - 7;
          const r = (idx % 15) - 7;
          return {
            action: a,
            q,
            r,
            cx: centerX + hexSize * (Math.sqrt(3) * q + (Math.sqrt(3) / 2) * r),
            cy: centerY + hexSize * (1.5 * r),
          };
        })
    : [];

  return (
    <div className="w-[95%] mx-auto app-container">
      <header className="header">
        <h1>Eclipse setup & map visualizer</h1>
        <p>Interactive staged setup backed by the shared OpenSpiel Eclipse core</p>
        {mySeatIdx >= 0 && (
          <span className="text-xs text-[#60a5fa] font-semibold">
            You are Player {mySeatIdx + 1} — {playerLabel(mySeatIdx)}
          </span>
        )}
      </header>

      {error && (
        <div className="bg-[#7f1d1d] border border-[#f87171] p-3 rounded-lg mb-4 text-sm">
          <strong>Error connecting to backend:</strong> {error}
        </div>
      )}

      <div className={`main-layout ${setupFinalized ? '!grid-cols-1' : ''}`}>
        {!setupFinalized && (
        <div className="panel">
          <div>
            <h3 className="panel-title">Setup Config</h3>
            <span className="text-xs text-[#94a3b8]">
              Stage 1 resolves initial randomness from this UI-driven configuration
            </span>
          </div>

          <div className="form-group">
            <label>RNG Seed</label>
            <input type="number" value={rngSeed} onChange={(e) => setRngSeed(Number(e.target.value))} />
          </div>

          <div className="form-group">
            <label>Number of Players ({numPlayers})</label>
            <input type="range" min="2" max="6" value={numPlayers} onChange={(e) => setNumPlayers(Number(e.target.value))} />
          </div>

          <div className="form-group">
            <label>NPC Difficulty (GCDS & Guardians)</label>
            <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
              {(gameMetadata?.npc_difficulties ?? ['Easy', 'Medium', 'Hard']).map((d: string) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-4 border-t border-[#2d313f] pt-4 mt-2">
            <div>
              <h3 className="panel-title">Player Choices</h3>
              <span className="text-xs text-[#94a3b8]">
                Species and control flags are UI-owned and sent into the setup pipeline
              </span>
            </div>

            <div className="flex flex-col gap-2.5">
              {playerIds.map((playerId) => {
                const currentTurnIdx = gameState?.turn_order.indexOf(playerId) ?? -1;
                const isFirstPicker = gameState
                  ? currentTurnIdx === gameState.players.length - 1
                  : false;
                return (
                  <div key={playerId} className={`player-choice-card ${isFirstPicker ? 'active' : ''}`}>
                    <div className="player-card-header">
                      <span className="text-[13px] font-bold">{playerLabel(playerId)}</span>
                      <span className="player-badge">
                        {gameState
                          ? isFirstPicker
                            ? '🔥 Picker 1'
                            : `Turn Position: ${currentTurnIdx + 1}`
                          : 'Pending Stage 1 turn order'}
                      </span>
                    </div>

                    <div className="form-group gap-1">
                      <select
                        value={speciesChoices[playerId] || speciesList[playerId % speciesList.length]}
                        onChange={(e) => handleSpeciesChange(playerId, e.target.value)}
                      >
                        {getSpeciesOptions(playerId).map((opt: { value: string; label: string; disabled?: boolean }) => (
                          <option key={opt.value} value={opt.value} disabled={opt.disabled}>{opt.label}</option>
                        ))}
                      </select>
                      {SPECIES_THEME[speciesChoices[playerId]]?.isTerran && playerStartingSectors[playerId] && (
                        <span className="text-xs text-[#94a3b8] mt-1">
                          Starting Sector: {playerStartingSectors[playerId]}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2 text-xs">
                      <input
                        type="checkbox"
                        id={`ai-${playerId}`}
                        checked={getAiDefault(playerId)}
                        onChange={(e) => handleAiChange(playerId, e.target.checked)}
                      />
                      <label htmlFor={`ai-${playerId}`} className="text-[#cbd5e1] cursor-pointer">
                        Control as AI Agent
                      </label>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <button className="btn-primary" onClick={handleInitializeStage1} disabled={loading}>
            {loading ? 'Processing C++...' : 'Generate Pre-Choice Snapshot'}
          </button>

          {snapshot && (
            <div className="flex flex-col gap-4 border-t border-[#2d313f] pt-4 mt-4">
              <div>
                <h3 className="panel-title">Stage 2: Finalize Setup</h3>
                <span className="text-xs text-[#94a3b8]">
                  Applies current player choices to the deterministic Stage 1 snapshot
                </span>
              </div>
              <button className="btn-secondary" onClick={handleFinalizeStage2} disabled={loading}>
                {loading ? 'Processing C++...' : 'Finalize Species & Spawn Map'}
              </button>
            </div>
          )}
        </div>
        )}

        <div className="board-container">
          {gameState && (
            <div className="turn-order-bar">
              <span className="text-xs uppercase text-[#94a3b8] font-bold mr-2">
                {isStarted ? `Round ${gameState.current_round} · Turn Order:` : 'Active Turn Order:'}
              </span>
              {gameState.turn_order.map((playerId, idx) => {
                if (playerId === 255) return null;
                const player = gameState.players[playerId];
                const isActive = gameState.current_player === playerId;
                const passed = player?.has_passed;
                return (
                  <div key={playerId} className={`turn-badge ${isActive ? 'active' : ''} ${passed ? 'passed' : ''}`}>
                    <span className="turn-num">{idx + 1}</span>
                    <span className={SPECIES_THEME[player?.species_id ?? '']?.cssClass ?? ''}>{playerLabel(playerId)} ({player?.species_id || 'Choosing...'})</span>
                    {player?.is_ai && <span className="text-[9px] bg-[#334155] text-[#93c5fd] px-1 py-0.5 rounded">AI</span>}
                    {passed && <span className="text-[9px] text-[#94a3b8]">✓ passed</span>}
                  </div>
                );
              })}
            </div>
          )}

          {isStarted && gameState && (
            <div className="stacks-bar">
              <span className="text-xs uppercase text-[#94a3b8] font-bold mr-1">Sector stacks left:</span>
              <span className="stack-pill inner" title="Ring I (Inner) tiles remaining">
                <span className="stack-ring">I</span> {bagCount(gameState.sector_bag_inner)}
              </span>
              <span className="stack-pill middle" title="Ring II (Middle) tiles remaining">
                <span className="stack-ring">II</span> {bagCount(gameState.sector_bag_middle)}
              </span>
              <span className="stack-pill outer" title="Ring III (Outer) tiles remaining">
                <span className="stack-ring">III</span> {bagCount(gameState.sector_bag_outer)}
              </span>
            </div>
          )}

          {isStarted && gameState && (
            <div className="game-hud">
              <ActionPanel
                explore={exploreState}
                legalActions={legalActions}
                isMyTurn={isMyTurn}
                isTerminal={isTerminal}
                busy={actionInProgress}
                currentPlayerLabel={currentPlayerLabel}
                ancientsOnSelected={ancientsOnSelected}
                onAction={submitAction}
              />
              <div className="panel economy-panel">
                <h3 className="panel-title">Players</h3>
                <div className="economy-grid">
                  {gameState.turn_order
                    .filter((pid) => pid !== 255)
                    .map((pid) => {
                      const p = gameState.players[pid];
                      if (!p) return null;
                      const onSectors = p.disks_on_sectors ?? 0;
                      const onActions = p.disks_on_actions ?? 0;
                      const discsLeft = Math.max(0, INFLUENCE_TOTAL - onSectors - onActions);
                      const isMine = pid === mySeatIdx;
                      return (
                        <div
                          key={pid}
                          className={`economy-card ${isMine ? 'mine' : ''} ${pid === gameState.current_player ? 'active' : ''}`}
                        >
                          <div className="economy-name">
                            <span className={SPECIES_THEME[p.species_id ?? '']?.cssClass ?? ''}>{playerLabel(pid)}</span>
                            <span className="economy-score">⭐ {p.score}</span>
                          </div>
                          <div className="economy-resources">
                            <span className="res gold" title="Money">💰 {p.resources.gold} <em>+{p.resources.gold_prod}</em></span>
                            <span className="res science" title="Science">🔬 {p.resources.science} <em>+{p.resources.science_prod}</em></span>
                            <span className="res materials" title="Materials">⚙️ {p.resources.materials} <em>+{p.resources.materials_prod}</em></span>
                          </div>
                          {isMine ? (
                            <InfluenceTrack onSectors={onSectors} onActions={onActions} />
                          ) : (
                            <div className="economy-meta">
                              <span title="Influence discs available">🔵 {discsLeft} discs</span>
                            </div>
                          )}
                          {p.has_passed && <div className="economy-meta"><span className="economy-passed">passed</span></div>}
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          )}

          <div className="map-viewport">
            {gameState ? (
              <svg width="100%" height="100%" viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}>
                {activeSectors.map(({ sector, cx, cy }) => {
                  const fillColor = getPlayerHexColor(sector.owner_id, sector.sector_id);
                  const isCenter = sector.sector_id === 1;
                  const units = getUnitsInSector(sector.sector_id);
                  // Ownership / threat cues so a wild (uncontrolled) sector is never
                  // mistaken for one you control. NPC units (Ancients/Guardians)
                  // block control until cleared in combat.
                  const owned = sector.owner_id !== 255;
                  const hasHostiles = units.some((u) => u.player_id === 255);
                  const stroke = isCenter
                    ? '#eab308'
                    : hasHostiles
                      ? '#ef4444'
                      : owned
                        ? '#f8fafc'
                        : '#475569';
                  const strokeWidth = isCenter || hasHostiles || owned ? '2.5' : '1.5';

                  return (
                    <g
                      key={`${sector.sector_id}-${sector.coords.q}-${sector.coords.r}`}
                      onMouseEnter={() => setHoveredSector(sector)}
                      onMouseLeave={() => setHoveredSector(null)}
                    >
                      <polygon
                        points={getHexPoints(cx, cy, hexSize - 1.5)}
                        fill={fillColor}
                        fillOpacity={owned || hasHostiles ? 1 : 0.65}
                        stroke={stroke}
                        strokeWidth={strokeWidth}
                        className="hex-polygon"
                      />

                      <text
                        x={cx}
                        y={cy - 6}
                        textAnchor="middle"
                        fill="#f8fafc"
                        fontSize="11px"
                        fontWeight="bold"
                        style={{ pointerEvents: 'none' }}
                      >
                        {sector.sector_id === 1 ? 'GCDS' : sector.sector_id}
                      </text>

                      <text
                        x={cx}
                        y={cy + 8}
                        textAnchor="middle"
                        fill="#94a3b8"
                        fontSize="8px"
                        style={{ pointerEvents: 'none' }}
                      >
                        ({sector.coords.q},{sector.coords.r})
                      </text>

                      {units.length > 0 && (
                        <g transform={`translate(${cx}, ${cy + 18})`}>
                          <circle r="7" fill="#dc2626" stroke="#ffffff" strokeWidth="1" />
                          <text y="2.5" textAnchor="middle" fill="#ffffff" fontSize="8px" fontWeight="bold">
                            {units.length}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}

                {legalZones.map((zone) => (
                  <g
                    key={`zone-${zone.action}`}
                    className="explore-zone"
                    onClick={() => submitAction(zone.action)}
                  >
                    <polygon
                      points={getHexPoints(zone.cx, zone.cy, hexSize - 1.5)}
                      className="explore-zone-hex"
                    />
                    <text
                      x={zone.cx}
                      y={zone.cy + 4}
                      textAnchor="middle"
                      fill="#a7f3d0"
                      fontSize="14px"
                      fontWeight="bold"
                      style={{ pointerEvents: 'none' }}
                    >
                      +
                    </text>
                  </g>
                ))}
              </svg>
            ) : (
              <div className="text-center text-[#64748b]">
                <svg className="w-16 h-16 mb-4 text-[#334155]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
                </svg>
                <h3>Galaxy Map Uninitialized</h3>
                <p className="text-sm">Send a Stage 1 setup config to spin up the shared C++ core</p>
              </div>
            )}

            {hoveredSector && gameState && (
              <div className="galaxy-info-overlay">
                <div className="galaxy-info-item">
                  <span>Sector ID</span>
                  <span>{hoveredSector.sector_id === 1 ? '001 (Center)' : hoveredSector.sector_id}</span>
                </div>
                <div className="galaxy-info-item">
                  <span>Coordinates</span>
                  <span>({hoveredSector.coords.q}, {hoveredSector.coords.r})</span>
                </div>
                <div className="galaxy-info-item">
                  <span>Owner</span>
                  <span>
                    {hoveredSector.owner_id === 255 ? 'Unowned (Neutral)' : `${playerLabel(hoveredSector.owner_id)} (${gameState.players[hoveredSector.owner_id]?.species_id})`}
                  </span>
                </div>
                <div className="galaxy-info-item">
                  <span>Sector Points</span>
                  <span>⭐️ {hoveredSector.points}</span>
                </div>
                {getUnitsInSector(hoveredSector.sector_id).length > 0 && (
                  <div className="galaxy-info-item">
                    <span>Active Units</span>
                    <span>
                      {getUnitsInSector(hoveredSector.sector_id).map((unit) => `${unit.type} (P${unit.player_id})`).join(', ')}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {gameState && (
            <div className="panel">
              <h3 className="panel-title">Round 1 Technology Market</h3>
              <div className="flex flex-col gap-4">
                {['Military', 'Grid', 'Nano', 'Rare'].map((category) => {
                  const categoryTechs = Object.entries(gameState.tech_tray).filter(([_, tech]) => tech.category === category);
                  if (categoryTechs.length === 0) return null;
                  return (
                    <div key={category}>
                      <h4 className="text-sm font-semibold text-[#94a3b8] mb-2 uppercase">{category} Track</h4>
                      <div className="tech-grid">
                        {categoryTechs.map(([techName, tech]) => (
                          <div key={techName} className={`tech-card ${tech.category}`}>
                            <div className="tech-name">{techName}</div>
                            <span className={`tech-category-badge ${tech.category}`}>{tech.category}</span>
                            <div className="tech-count mt-1.5">
                              Qty Available: <strong>{tech.count}</strong>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {snapshot && (
            <div className="json-inspector">
              <div className="json-title" onClick={() => setJsonExpanded(!jsonExpanded)}>
                <span>🔍 Inspect Raw Setup Snapshot JSON</span>
                <span>{jsonExpanded ? 'Collapse ▲' : 'Expand ▼'}</span>
              </div>
              {jsonExpanded && (
                <pre className="json-code">
                  {JSON.stringify(snapshot, null, 2)}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
