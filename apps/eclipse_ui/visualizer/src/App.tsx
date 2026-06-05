import { useEffect, useMemo, useState } from 'react';
import './App.css';
import { ImageHoverPreview, useImageHoverPreview } from './ImageHoverPreview';
import {
  SPECIES_THEME,
  UNOWNED_COLOR,
  NPC_COLOR_GCDS,
  NPC_COLOR_GUARDIAN,
  getPlayerColor,
} from './theme';
import ActionPanel, {type ExploreState, type ResearchState} from './ActionPanel';
import { ACTION, TRADE_LABELS, POP_TRACK_LABELS } from './actionTypes';

import {
  API_BASE,
  WS_BASE,
  SECTOR_ASSETS_BASE,
  buildTechMarketRows,
  techImageUrl,
  TECH_CATEGORIES,
  type TechMarketEntry,
} from './types/lobby';

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

interface PlanetLayout {
  slot: number;
  type: string;
  dx: number;
  dy: number;
}

interface SectorLayout {
  influence_space: { dx: number; dy: number };
  planets: PlanetLayout[];
  monolith_anchor: { dx: number; dy: number };
  orbital_anchor: { dx: number; dy: number };
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
  tech_tray: Record<string, TechMarketEntry>;
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
  research_state?: ResearchState;
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

const EMPTY_LEGAL_ACTIONS: number[] = [];

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

// Production table: index = cubes remaining on track (12=full=0 prod, 0=empty=28 prod)
const POPULATION_PRODUCTION_TABLE = [28, 24, 21, 18, 15, 12, 10, 8, 6, 4, 3, 2, 0] as const;
const POP_TRACK_MAX = 12;
const POP_TRACK_STEPS = POP_TRACK_MAX - 1;

// Three population tracks arranged as arcs in a circle (like the control board).
// cubesOnTrack[n] = cubes remaining on track n (12=full, 0=empty).
// Tracks: 0=Materials, 1=Science, 2=Money
function PopulationTracks({ resources, playerColor }: { resources: Resources; playerColor: string }) {
  const tracks = [
    { label: 'Mat', color: '#f97316', cubesOnTrack: resources.materials_prod },
    { label: 'Sci', color: '#818cf8', cubesOnTrack: resources.science_prod },
    { label: '$',   color: '#fbbf24', cubesOnTrack: resources.gold_prod },
  ];

  const W = 120;
  const CX = W / 2;
  const CY = W / 2 + 4;
  const OUTER_R = 46;
  const INNER_R = 30;
  const DOT_R = 3.2;
  // Arc centers (degrees): Materials=210, Science=90, Money=330 (=−30)
  const ARC_SPAN = 100;
  const ARC_CENTERS_DEG = [210, 90, 330];

  return (
    <div className="pop-tracks">
      <svg viewBox={`0 0 ${W} ${W + 8}`} width={W} height={W + 8} style={{ display: 'block' }}>
        {tracks.map((track, ti) => {
          const centerDeg = ARC_CENTERS_DEG[ti];
          const startDeg  = centerDeg - ARC_SPAN / 2;
          const prod = POPULATION_PRODUCTION_TABLE[Math.min(track.cubesOnTrack, POP_TRACK_MAX)];

          // Place 12 dots along the arc
          const dots = Array.from({ length: POP_TRACK_MAX }, (_, i) => {
            const frac = i / POP_TRACK_STEPS;
            const deg  = startDeg + frac * ARC_SPAN;
            const rad  = (deg * Math.PI) / 180;
            const x    = CX + OUTER_R * Math.cos(rad);
            const y    = CY + OUTER_R * Math.sin(rad);
            // Cubes fill from the start of the arc; empty = cube placed on sector
            const onTrack = i < track.cubesOnTrack;
            return { x, y, onTrack };
          });

          // Label position: inside the arc, near arc center
          const labelRad = (centerDeg * Math.PI) / 180;
          const lx = CX + INNER_R * Math.cos(labelRad);
          const ly = CY + INNER_R * Math.sin(labelRad);

          return (
            <g key={ti}>
              {dots.map((d, i) => (
                <circle
                  key={i}
                  cx={d.x}
                  cy={d.y}
                  r={DOT_R}
                  fill={d.onTrack ? track.color : 'none'}
                  stroke={track.color}
                  strokeWidth={0.8}
                  opacity={d.onTrack ? 0.9 : 0.35}
                />
              ))}
              {/* Production value inside the arc */}
              <text
                x={lx}
                y={ly + 3}
                textAnchor="middle"
                fill={track.color}
                fontSize="9"
                fontWeight="bold"
              >
                {prod}
              </text>
              {/* Track label at outer edge of arc center */}
              <text
                x={CX + (OUTER_R + 10) * Math.cos(labelRad)}
                y={CY + (OUTER_R + 10) * Math.sin(labelRad) + 3}
                textAnchor="middle"
                fill={track.color}
                fontSize="7"
                opacity={0.7}
              >
                {track.label}
              </text>
            </g>
          );
        })}
        {/* Center dot for aesthetics */}
        <circle cx={CX} cy={CY} r={2.5} fill={playerColor} opacity={0.5} />
      </svg>
    </div>
  );
}

function ColonyShips({
  total,
  available,
  legalPlacements,
  onPlace,
}: {
  total: number;
  available: number;
  legalPlacements: { actionId: number; sectorId: number; slotIdx: number; track: number }[];
  onPlace: (actionId: number) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  if (total === 0) return null;

  return (
    <div className="colony-ships">
      <div
        className={`colony-ships-header ${legalPlacements.length > 0 ? 'clickable' : ''}`}
        onClick={() => legalPlacements.length > 0 && setExpanded(e => !e)}
        title="Colony Ships"
      >
        <span className="colony-ships-label">Colony Ships</span>
        <span className="colony-ships-icons">
          {Array.from({ length: total }, (_, i) => (
            <span
              key={i}
              className={`colony-ship-icon ${i < available ? 'available' : 'used'}`}
              title={i < available ? 'Available (faceup)' : 'Used (facedown)'}
            >
              ◎
            </span>
          ))}
        </span>
        {legalPlacements.length > 0 && (
          <span className="colony-ship-badge">{legalPlacements.length} placements</span>
        )}
      </div>
      {expanded && legalPlacements.length > 0 && (
        <div className="colony-ship-targets">
          {legalPlacements.map(({ actionId, sectorId, slotIdx, track }) => (
            <button
              key={actionId}
              className="colony-ship-target-btn"
              onClick={() => { onPlace(actionId); setExpanded(false); }}
              title={`Place a ${POP_TRACK_LABELS[track]} cube`}
            >
              Sector {sectorId} · slot {slotIdx} · {POP_TRACK_LABELS[track]}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function TradePanel({
  tradeRate,
  legalTradeActions,
  onTrade,
  resources,
}: {
  tradeRate: number;
  legalTradeActions: number[];
  onTrade: (actionId: number) => void;
  resources: Resources;
}) {
  const [open, setOpen] = useState(false);
  const allConversions = Array.from({ length: 6 }, (_, i) => i);

  const canAfford = (conv: number): boolean => {
    switch (conv) {
      case 0: case 1: return resources.gold >= tradeRate;
      case 2: case 3: return resources.science >= tradeRate;
      case 4: case 5: return resources.materials >= tradeRate;
      default: return false;
    }
  };

  const handleTrade = (conv: number) => {
    if (!canAfford(conv)) return;
    onTrade(ACTION.TRADE_START + conv);
  };

  const handleOverlayKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setOpen(false);
    }
  };

  return (
    <>
      <div className="trade-panel">
        <span className="trade-label">Trade (×{tradeRate})</span>
        <button
          className="trade-btn"
          onClick={() => setOpen(!open)}
          title={open ? 'Close trade modal' : 'Open trade modal'}
          aria-label={open ? 'Close trade modal' : 'Open trade modal'}
        >
          ⇄
        </button>
      </div>

      {open && (
        <div
          className="trade-modal-overlay"
          onClick={() => setOpen(false)}
          onKeyDown={handleOverlayKeyDown}
          role="dialog"
          aria-label="Trade resources"
        >
          <div className="trade-modal" onClick={e => e.stopPropagation()}>
            <div className="trade-modal-header">
              <span>Trade (×{tradeRate})</span>
              <button
                className="trade-modal-close"
                onClick={() => setOpen(false)}
                aria-label="Close trade modal"
              >✕</button>
            </div>
            <div className="trade-modal-body">
              {allConversions.map(conv => {
                const info = TRADE_LABELS[conv];
                if (!info) return null;
                const affordable = canAfford(conv);
                const isLegal = legalTradeActions.includes(ACTION.TRADE_START + conv);
                const unavailable = !isLegal || !affordable;
                return (
                  <button
                    key={conv}
                    className={`trade-modal-option ${unavailable ? 'trade-modal-disabled' : ''}`}
                    onClick={() => handleTrade(conv)}
                    disabled={unavailable}
                    title={isLegal
                      ? affordable
                        ? `Pay ${tradeRate} ${info.from} → 1 ${info.to}`
                        : `Need ${tradeRate} ${info.from} (have ${
                            conv === 0 || conv === 1 ? resources.gold
                            : conv === 2 || conv === 3 ? resources.science
                            : resources.materials
                          })`
                      : 'Not available this turn'
                    }
                  >
                    <span className="trade-modal-emoji">{info.emoji}</span>
                    <div className="trade-modal-info">
                      <span className="trade-modal-from">{info.from}</span>
                      <span className="trade-modal-arrow">→</span>
                      <span className="trade-modal-to">{info.to}</span>
                    </div>
                    <span className="trade-modal-cost">
                      {isLegal
                        ? affordable ? `−${tradeRate}` : `Need ${tradeRate}`
                        : '—'}
                      {info.from}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function App({
  initialMetadata,
  initialSnapshot,
  mySeatIdx = -1,
  playerNames = [],
  isHost = false,
}: {
  initialMetadata: any;
  initialSnapshot?: SetupSnapshot;
  mySeatIdx?: number;
  playerNames?: (string | null)[];
  isHost?: boolean;
}) {
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
  const [previewRotation, setPreviewRotation] = useState<number | null>(null);
  const [previewingDrawnTile, setPreviewingDrawnTile] = useState<number | null>(null);
  const [previewingDrawnTileIndex, setPreviewingDrawnTileIndex] = useState<number | null>(null);
  const [hoveredSector, setHoveredSector] = useState<Sector | null>(null);
  const [debugStateText, setDebugStateText] = useState<string>('');
  const [debugBusy, setDebugBusy] = useState<boolean>(false);
  // sector_id -> image URL (real tile art painted inside each hex).
  const [sectorImages, setSectorImages] = useState<Record<number, string>>({});
  const [brokenImages, setBrokenImages] = useState<Set<number>>(new Set());
  // sector_id -> pixel layout (planet positions, anchors) from sector_layouts.json
  const [sectorLayouts, setSectorLayouts] = useState<Record<number, SectorLayout>>({});
  const { preview, beginPreview, clearPreview } = useImageHoverPreview();
  const [setupFinalized, setSetupFinalized] = useState<boolean>(initialSnapshot?.finalized ?? false);
  const [playerStartingSectors, setPlayerStartingSectors] = useState<Record<number, number>>({});
  const [selectedRareTech, setSelectedRareTech] = useState<{ name: string; order: number } | null>(null);

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
  const techRows = gameState
    ? buildTechMarketRows(gameMetadata.tech_catalog ?? {}, gameState.tech_tray)
    : {};

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

  // Load the sector_id -> tile-art manifest once. Failure is non-fatal: hexes
  // just fall back to their colored polygons.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/sectors/manifest`);
        const manifest: Record<string, string> = await res.json();
        if (cancelled) return;
        const urls: Record<number, string> = {};
        for (const [id, filename] of Object.entries(manifest)) {
          urls[Number(id)] = `${SECTOR_ASSETS_BASE}/${filename}`;
        }
        setSectorImages(urls);
      } catch {
        /* keep empty map -> colored-polygon fallback */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Load pixel layout for each sector tile (planet positions, anchors).
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/sectors/layouts`);
        const raw: Record<string, SectorLayout> = await res.json();
        if (cancelled) return;
        setSectorLayouts(
          Object.fromEntries(Object.entries(raw).map(([k, v]) => [Number(k), v]))
        );
      } catch { /* non-fatal */ }
    })();
    return () => { cancelled = true; };
  }, []);

  const sectorImageUrl = (id: number): string | null =>
    (!brokenImages.has(id) && sectorImages[id]) || null;

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

  const getPlayerId = () => sessionStorage.getItem('eclipse_player_id') ?? '';

  const dumpDebugState = async () => {
    if (debugBusy) return;
    setDebugBusy(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/debug/state/dump`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_id: getPlayerId() }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `Dump failed (${response.status})`);
      }
      const data = await response.json();
      setDebugStateText(JSON.stringify(data.game_blob, null, 2));
    } catch (err: any) {
      setError(err.message || 'Dump failed');
    } finally {
      setDebugBusy(false);
    }
  };

  const loadDebugState = async () => {
    if (debugBusy) return;
    setDebugBusy(true);
    setError(null);
    try {
      if (!debugStateText.trim()) {
        throw new Error('Paste a dumped game state first');
      }
      const gameBlob = JSON.parse(debugStateText);
      const response = await fetch(`${API_BASE}/debug/state/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_id: getPlayerId(), game_blob: gameBlob }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `Load failed (${response.status})`);
      }
      const updated = await response.json();
      if (updated) setSnapshot(updated as SetupSnapshot);
      setDebugStateText(JSON.stringify(gameBlob, null, 2));
    } catch (err: any) {
      setError(err.message || 'Load failed');
    } finally {
      setDebugBusy(false);
    }
  };

  const hexSize = 35;
  // The PNG tile art is 180 degrees from the backend's rotation=0 wormhole
  // masks. Apply that fixed offset, then apply the game's rotation.
  const IMAGE_ROTATION_OFFSET = 180;
  const SQRT3 = Math.sqrt(3);

  // Axial (q,r) → pixel for a flat-top hex layout.
  const axialToPixel = (q: number, r: number) => ({
    cx: hexSize * (1.5 * q),
    cy: hexSize * (SQRT3 / 2) * q + hexSize * SQRT3 * r,
  });

  const getHexPoints = (cx: number, cy: number, r: number) => {
    const points = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 180) * (60 * i); // flat-top: vertices left/right
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
    return getPlayerColor(ownerId);
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
          const { cx, cy } = axialToPixel(q, r);
          activeSectors.push({ sector, cx, cy });
        }
      }
    }
  }

  const playerIds = Array.from({ length: numPlayers }, (_, playerId) => playerId);

  // ── Gameplay (Action phase) derived state ──────────────────────────────────
  const legalActions = snapshot?.legal_actions ?? EMPTY_LEGAL_ACTIONS;
  const exploreState = gameState?.explore_state;
  const explorePhase = exploreState?.phase ?? 'inactive';
  const researchState = gameState?.research_state;
  const isTerminal = snapshot?.is_terminal ?? false;
  const isStarted = setupFinalized && snapshot?.current_player !== undefined;
  const isMyTurn = isStarted && snapshot?.current_player === mySeatIdx && !isTerminal;
  const isResearchPhase = isMyTurn && researchState?.phase === 'choose_tech';
  const inZoneSelect = isMyTurn && explorePhase === 'choose_zone';
  const currentPlayerLabel =
    snapshot?.current_player !== undefined ? playerLabel(snapshot.current_player) : '';
  const selectedSectorId = exploreState?.selected_sector_id ?? 0;
  const legalRotations = useMemo(
    () =>
      legalActions
        .filter((action) => action >= ACTION.EXPLORE_ROT_START && action < ACTION.EXPLORE_ROT_START + 6)
        .map((action) => action - ACTION.EXPLORE_ROT_START)
        .sort((a, b) => a - b),
    [legalActions]
  );
  const inRotationPreview =
    isMyTurn &&
    explorePhase === 'choose_rotation' &&
    selectedSectorId > 0 &&
    legalRotations.length > 0;
  const currentPreviewRotation =
    inRotationPreview && previewRotation !== null && legalRotations.includes(previewRotation)
      ? previewRotation
      : inRotationPreview
        ? legalRotations[0]
        : null;
  const ancientsOnSelected =
    !!gameState &&
    selectedSectorId > 0 &&
    gameState.unit_registry.some((u) => u.sector_id === selectedSectorId && u.player_id === 255);

  const cyclePreviewRotation = (direction: -1 | 1) => {
    if (!legalRotations.length) return;
    const currentIndex = Math.max(0, legalRotations.indexOf(currentPreviewRotation ?? legalRotations[0]));
    const nextIndex = (currentIndex + direction + legalRotations.length) % legalRotations.length;
    setPreviewRotation(legalRotations[nextIndex]);
  };

  const confirmPreviewRotation = () => {
    if (currentPreviewRotation === null || !legalRotations.includes(currentPreviewRotation)) return;
    submitAction(ACTION.EXPLORE_ROT_START + currentPreviewRotation);
  };

  // Bonus actions available this turn.
  const legalTradeActions = isMyTurn
    ? legalActions.filter(a => a >= ACTION.TRADE_START && a < ACTION.COLONY_SHIP_START)
    : [];
  const legalColonyShipActions = isMyTurn
    ? legalActions.filter(a => a >= ACTION.COLONY_SHIP_START)
    : [];

  // Decode colony ship actions into (sectorId, slotIdx, track) for display.
  const colonyShipPlacements = legalColonyShipActions.map(actionId => {
    const encoded = actionId - ACTION.COLONY_SHIP_START;
    const cellIdx = Math.floor(encoded / ACTION.COLONY_SHIP_CODES_PER_CELL);
    const rem = encoded % ACTION.COLONY_SHIP_CODES_PER_CELL;
    const slotIdx = Math.floor(rem / ACTION.COLONY_SHIP_TRACKS);
    const track = rem % ACTION.COLONY_SHIP_TRACKS;
    const qIdx = Math.floor(cellIdx / 15);
    const rIdx = cellIdx % 15;
    const sector = gameState?.galaxy[qIdx]?.[rIdx];
    return { actionId, sectorId: sector?.sector_id ?? 0, slotIdx, track };
  });

  // Legal explore zones → clickable hexes (action id 35 + hex_to_index(q,r)).
  const legalZones = inZoneSelect
    ? legalActions
        .filter((a) => a >= ACTION.EXPLORE_ZONE_START)
        .map((a) => {
          const idx = a - ACTION.EXPLORE_ZONE_START;
          const q = Math.floor(idx / 15) - 7;
          const r = (idx % 15) - 7;
          return { action: a, q, r, ...axialToPixel(q, r) };
        })
    : [];

  const previewZone =
    inRotationPreview && exploreState
      ? {
          q: exploreState.zone_q,
          r: exploreState.zone_r,
          ...axialToPixel(exploreState.zone_q, exploreState.zone_r),
        }
      : null;

  // Floating preview for drawn tiles during select_drawn_tile phase.
  const inSelectDrawnTile = isMyTurn && explorePhase === 'select_drawn_tile';
  const drawnTilePreviewZone = inSelectDrawnTile && exploreState && previewingDrawnTile !== null && previewingDrawnTileIndex !== null
    ? {
        q: exploreState.zone_q,
        r: exploreState.zone_r,
        ...axialToPixel(exploreState.zone_q, exploreState.zone_r),
        previewSectorId: previewingDrawnTile,
        previewTileIndex: previewingDrawnTileIndex,
      }
    : null;

  // Clear drawn tile preview state when it's no longer the select_drawn_tile phase.
  useEffect(() => {
    if (!inSelectDrawnTile) {
      setPreviewingDrawnTile(null);
      setPreviewingDrawnTileIndex(null);
    }
  }, [inSelectDrawnTile]);

  // Clear selected rare tech when no longer in the choose_tech research phase.
  useEffect(() => {
    if (!isResearchPhase) {
      setSelectedRareTech(null);
    }
  }, [isResearchPhase]);

  // Auto-fit the SVG viewBox to the populated region (sectors + explore zones)
  // so the board fills the panel and zooms in as the galaxy grows.
  const fitCells = previewZone ? [...activeSectors, ...legalZones, previewZone] : drawnTilePreviewZone ? [...activeSectors, ...legalZones, drawnTilePreviewZone] : [...activeSectors, ...legalZones];
  const viewBox = (() => {
    if (fitCells.length === 0) return '0 0 600 520';
    const xs = fitCells.map((c) => c.cx);
    const ys = fitCells.map((c) => c.cy);
    const pad = hexSize * 1.4;
    const minX = Math.min(...xs) - hexSize - pad;
    const minY = Math.min(...ys) - hexSize - pad;
    const w = Math.max(...xs) - Math.min(...xs) + 2 * (hexSize + pad);
    const h = Math.max(...ys) - Math.min(...ys) + 2 * (hexSize + pad);
    return `${minX} ${minY} ${w} ${h}`;
  })();

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
                    <span style={{ color: getPlayerColor(playerId) }}>{playerLabel(playerId)} ({player?.species_id || 'Choosing...'})</span>
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
                research={researchState}
                selectedRareTech={selectedRareTech}
                onClearSelectedRareTech={() => setSelectedRareTech(null)}
                legalActions={legalActions}
                isMyTurn={isMyTurn}
                isTerminal={isTerminal}
                busy={actionInProgress}
                currentPlayerLabel={currentPlayerLabel}
                ancientsOnSelected={ancientsOnSelected}
                previewRotation={currentPreviewRotation}
                sectorImages={sectorImages}
                onPreviewDrawnTile={(id, idx) => {
                  setPreviewingDrawnTile(id);
                  setPreviewingDrawnTileIndex(idx);
                }}
                onClearPreviewDrawnTile={() => {
                  setPreviewingDrawnTile(null);
                  setPreviewingDrawnTileIndex(null);
                }}
                onConfirmPreviewRotation={confirmPreviewRotation}
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
                            <span style={{ color: getPlayerColor(pid) }}>{playerLabel(pid)}</span>
                            <span className="economy-score">⭐ {p.score}</span>
                          </div>
                          <div className="economy-resources">
                            <span className="res gold" title="Money">
                              💰 {p.resources.gold}
                              <em> +{POPULATION_PRODUCTION_TABLE[Math.min(p.resources.gold_prod, POP_TRACK_MAX)]}</em>
                            </span>
                            <span className="res science" title="Science">
                              🔬 {p.resources.science}
                              <em> +{POPULATION_PRODUCTION_TABLE[Math.min(p.resources.science_prod, POP_TRACK_MAX)]}</em>
                            </span>
                            <span className="res materials" title="Materials">
                              ⚙️ {p.resources.materials}
                              <em> +{POPULATION_PRODUCTION_TABLE[Math.min(p.resources.materials_prod, POP_TRACK_MAX)]}</em>
                            </span>
                          </div>
                          <PopulationTracks resources={p.resources} playerColor={getPlayerColor(pid)} />
                          {isMine ? (
                            <>
                              <InfluenceTrack onSectors={onSectors} onActions={onActions} />
                              <ColonyShips
                                total={p.colony_ships_total}
                                available={p.colony_ships_available}
                                legalPlacements={colonyShipPlacements}
                                onPlace={submitAction}
                              />
                              <TradePanel
                                tradeRate={p.trade_rate}
                                legalTradeActions={legalTradeActions}
                                onTrade={submitAction}
                                resources={p.resources}
                              />
                            </>
                          ) : (
                            <div className="economy-meta">
                              <span title="Influence discs available">🔵 {discsLeft} discs</span>
                              {p.colony_ships_total > 0 && (
                                <span title="Colony ships">
                                  ◎ {p.colony_ships_available}/{p.colony_ships_total}
                                </span>
                              )}
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
              <svg width="100%" height="100%" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
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
                    : owned
                      ? getPlayerColor(sector.owner_id)
                      : hasHostiles
                        ? '#ef4444'
                        : '#475569';
                  const strokeWidth = isCenter || hasHostiles || owned ? '2.5' : '1.5';

                  const imgUrl = sectorImageUrl(sector.sector_id);
                  const r = hexSize - 1.5;
                  const clipId = `hexclip-${sector.sector_id}-${sector.coords.q}-${sector.coords.r}`;
                  // Tile art (square) sized to cover the hex's circumdiameter.
                  const imgSize = 2 * r;
                  // Backend rotates wormholes CCW through the E,NE,NW,W,SW,SE
                  // edge order (galaxy.h); SVG rotate() is CW, so negate.
                  const imgDeg = IMAGE_ROTATION_OFFSET - 60 * (sector.rotation ?? 0);

                  return (
                    <g
                      key={`${sector.sector_id}-${sector.coords.q}-${sector.coords.r}`}
                      onMouseEnter={() => {
                        setHoveredSector(sector);
                        if (imgUrl) {
                          beginPreview({
                            src: imgUrl,
                            label: sector.sector_id === 1 ? '001 (Center)' : `Sector ${sector.sector_id}`,
                          });
                        }
                      }}
                      onMouseLeave={() => {
                        setHoveredSector(null);
                        clearPreview();
                      }}
                    >
                      {imgUrl && (
                        <>
                          <clipPath id={clipId}>
                            <polygon points={getHexPoints(cx, cy, r)} />
                          </clipPath>
                          {/* clipPath lives on the (unrotated) group so the hex
                              clip stays put while only the image rotates. */}
                          <g clipPath={`url(#${clipId})`}>
                            <image
                              href={imgUrl}
                              x={cx - imgSize / 2}
                              y={cy - imgSize / 2}
                              width={imgSize}
                              height={imgSize}
                              preserveAspectRatio="xMidYMid slice"
                              transform={`rotate(${imgDeg} ${cx} ${cy})`}
                              onError={() =>
                                setBrokenImages((prev) => new Set(prev).add(sector.sector_id))
                              }
                              style={{ pointerEvents: 'none' }}
                            />
                          </g>
                        </>
                      )}

                      <polygon
                        points={getHexPoints(cx, cy, r)}
                        fill={imgUrl ? 'none' : fillColor}
                        fillOpacity={imgUrl ? 0 : owned || hasHostiles ? 1 : 0.65}
                        stroke={stroke}
                        strokeWidth={strokeWidth}
                        className="hex-polygon"
                        style={{ pointerEvents: 'all' }}
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

                      {/* ── sector overlay: planet cubes, influence disk, structures ── */}
                      {(() => {
                        const layout = sectorLayouts[sector.sector_id];
                        if (!layout || !imgUrl) return null;
                        // Image occupies imgSize×imgSize SVG pixels → same scale for overlays
                        const scale = imgSize / 1024;
                        const pColor = owned ? getPlayerColor(sector.owner_id) : '#ffffff';
                        return (
                          // Wrap in the same rotation as the tile image so overlays track with it
                          <g transform={`rotate(${imgDeg} ${cx} ${cy})`} style={{ pointerEvents: 'none' }}>

                            {/* Influence disk — small player-coloured disc at influence space */}
                            {owned && (
                              <circle
                                cx={cx + layout.influence_space.dx * scale}
                                cy={cy + layout.influence_space.dy * scale}
                                r={5}
                                fill={pColor}
                                fillOpacity={0.9}
                                stroke="#000"
                                strokeWidth={0.6}
                              />
                            )}

                            {/* Planet population cubes */}
                            {layout.planets.map((planet) => {
                              const occupied = Boolean((sector.occupied_slots_mask >> planet.slot) & 1);
                              const px = cx + planet.dx * scale;
                              const py = cy + planet.dy * scale;
                              return occupied ? (
                                <rect
                                  key={planet.slot}
                                  x={px - 4}
                                  y={py - 4}
                                  width={8}
                                  height={8}
                                  rx={1}
                                  fill={pColor}
                                  fillOpacity={0.9}
                                  stroke="#000"
                                  strokeWidth={0.5}
                                />
                              ) : (
                                <rect
                                  key={planet.slot}
                                  x={px - 3.5}
                                  y={py - 3.5}
                                  width={7}
                                  height={7}
                                  rx={1}
                                  fill="none"
                                  stroke="#ffffff"
                                  strokeWidth={0.6}
                                  strokeOpacity={0.35}
                                />
                              );
                            })}

                            {/* Monolith — purple rectangle */}
                            {sector.monolith_built && (
                              <rect
                                x={cx + layout.monolith_anchor.dx * scale - 4}
                                y={cy + layout.monolith_anchor.dy * scale - 6}
                                width={8}
                                height={12}
                                rx={1}
                                fill="#a855f7"
                                fillOpacity={0.9}
                                stroke="#000"
                                strokeWidth={0.5}
                              />
                            )}

                            {/* Orbital — cyan circle */}
                            {sector.orbital_built && (
                              <circle
                                cx={cx + layout.orbital_anchor.dx * scale}
                                cy={cy + layout.orbital_anchor.dy * scale}
                                r={4}
                                fill="#22d3ee"
                                fillOpacity={0.9}
                                stroke="#000"
                                strokeWidth={0.5}
                              />
                            )}
                          </g>
                        );
                      })()}
                    </g>
                  );
                })}

                {inRotationPreview && previewZone && currentPreviewRotation !== null && (() => {
                  const imgUrl = sectorImageUrl(selectedSectorId);
                  const r = hexSize - 1.5;
                  const imgSize = 2 * r;
                  const clipId = `explore-preview-clip-${selectedSectorId}-${previewZone.q}-${previewZone.r}`;
                  const imgDeg = IMAGE_ROTATION_OFFSET - 60 * currentPreviewRotation;
                  const leftControlX = previewZone.cx - hexSize - 18;
                  const rightControlX = previewZone.cx + hexSize + 18;

                  return (
                    <g className="explore-preview">
                      {imgUrl && (
                        <>
                          <clipPath id={clipId}>
                            <polygon points={getHexPoints(previewZone.cx, previewZone.cy, r)} />
                          </clipPath>
                          <g clipPath={`url(#${clipId})`}>
                            <image
                              href={imgUrl}
                              x={previewZone.cx - imgSize / 2}
                              y={previewZone.cy - imgSize / 2}
                              width={imgSize}
                              height={imgSize}
                              preserveAspectRatio="xMidYMid slice"
                              transform={`rotate(${imgDeg} ${previewZone.cx} ${previewZone.cy})`}
                              onError={() =>
                                setBrokenImages((prev) => new Set(prev).add(selectedSectorId))
                              }
                              style={{ pointerEvents: 'none' }}
                            />
                          </g>
                        </>
                      )}

                      <polygon
                        points={getHexPoints(previewZone.cx, previewZone.cy, r)}
                        fill={imgUrl ? 'none' : UNOWNED_COLOR}
                        fillOpacity={imgUrl ? 0 : 0.75}
                        className="explore-preview-hex"
                      />
                      <text
                        x={previewZone.cx}
                        y={previewZone.cy - 6}
                        textAnchor="middle"
                        className="explore-preview-id"
                      >
                        {selectedSectorId}
                      </text>
                      <text
                        x={previewZone.cx}
                        y={previewZone.cy + 9}
                        textAnchor="middle"
                        className="explore-preview-rotation"
                      >
                        rot {currentPreviewRotation}
                      </text>

                      <g
                        className="explore-preview-control"
                        onClick={(event) => {
                          event.stopPropagation();
                          cyclePreviewRotation(1);
                        }}
                      >
                        <circle cx={leftControlX} cy={previewZone.cy} r="14" />
                        <text x={leftControlX} y={previewZone.cy + 5} textAnchor="middle">&lt;</text>
                      </g>
                      <g
                        className="explore-preview-control"
                        onClick={(event) => {
                          event.stopPropagation();
                          cyclePreviewRotation(-1);
                        }}
                      >
                        <circle cx={rightControlX} cy={previewZone.cy} r="14" />
                        <text x={rightControlX} y={previewZone.cy + 5} textAnchor="middle">&gt;</text>
                      </g>
                    </g>
                  );
                })()}

                {drawnTilePreviewZone && (() => {
                  const sectorId = drawnTilePreviewZone.previewSectorId!;
                  const tileIndex = drawnTilePreviewZone.previewTileIndex!;
                  const imgUrl = sectorImageUrl(sectorId);
                  const r = hexSize - 1.5;
                  const imgSize = 2 * r;
                  const clipId = `drawn-preview-clip-${sectorId}-${tileIndex}-${drawnTilePreviewZone.q}-${drawnTilePreviewZone.r}`;

                  return (
                    <g className="drawn-tile-preview">
                      {imgUrl && (
                        <>
                          <clipPath id={clipId}>
                            <polygon points={getHexPoints(drawnTilePreviewZone.cx, drawnTilePreviewZone.cy, r)} />
                          </clipPath>
                          <g clipPath={`url(#${clipId})`}>
                            <image
                              href={imgUrl}
                              x={drawnTilePreviewZone.cx - imgSize / 2}
                              y={drawnTilePreviewZone.cy - imgSize / 2}
                              width={imgSize}
                              height={imgSize}
                              preserveAspectRatio="xMidYMid slice"
                              style={{ pointerEvents: 'none' }}
                            />
                          </g>
                        </>
                      )}

                      <polygon
                        points={getHexPoints(drawnTilePreviewZone.cx, drawnTilePreviewZone.cy, r)}
                        fill={imgUrl ? 'none' : UNOWNED_COLOR}
                        fillOpacity={imgUrl ? 0 : 0.75}
                        className="drawn-tile-preview-hex"
                      />
                      <text
                        x={drawnTilePreviewZone.cx}
                        y={drawnTilePreviewZone.cy - 6}
                        textAnchor="middle"
                        className="drawn-tile-preview-id"
                      >
                        {sectorId}
                      </text>
                      <text
                        x={drawnTilePreviewZone.cx}
                        y={drawnTilePreviewZone.cy + 9}
                        textAnchor="middle"
                        className="drawn-tile-preview-label"
                      >
                        Tile {tileIndex + 1}
                      </text>
                    </g>
                  );
                })()}

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
              <div className="tech-market">
                {TECH_CATEGORIES.map((category) => {
                  const techs = techRows[category];
                  if (!techs.length) return null;
                  return (
                    <div key={category} className="tech-row">
                      {techs.map(([techName, tech]) => {
                        let isResearchable = false;
                        let onTechClick: (() => void) | undefined = undefined;

                        if (isResearchPhase) {
                          if (tech.category !== 'Rare') {
                            const actionId = ACTION.RESEARCH_STANDARD_START + (tech.order ?? 0);
                            if (legalActions.includes(actionId)) {
                              isResearchable = true;
                              onTechClick = () => submitAction(actionId);
                            }
                          } else {
                            const rareIdx = (tech.order ?? 0) - 24;
                            const isAnyTrackLegal = [0, 1, 2].some((track) =>
                              legalActions.includes(ACTION.RESEARCH_RARE_START + rareIdx * 3 + track)
                            );
                            if (isAnyTrackLegal) {
                              isResearchable = true;
                              onTechClick = () => setSelectedRareTech({ name: techName, order: tech.order ?? 0 });
                            }
                          }
                        }

                        return (
                          <div
                            key={techName}
                            className={`tech-card ${tech.category} ${tech.count === 0 ? 'unavailable' : ''} ${
                              isResearchable ? 'researchable' : ''
                            } ${selectedRareTech?.order === tech.order ? 'selected-research' : ''}`}
                            onClick={onTechClick}
                            onMouseEnter={() =>
                              beginPreview({ src: techImageUrl(techName, tech.category), label: techName })
                            }
                            onMouseLeave={clearPreview}
                          >
                            <img
                              className="tech-image"
                              src={techImageUrl(techName, tech.category)}
                              alt=""
                              onError={(event) => {
                                event.currentTarget.style.display = 'none';
                              }}
                            />
                            <div className="tech-name">{techName}</div>
                            <div className="tech-meta">
                              <span className={`tech-category-badge ${tech.category}`}>{tech.category}</span>
                              <span className="tech-count">{tech.count}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {isStarted && isHost && (
            <div className="panel debug-state-panel">
              <div className="debug-state-header">
                <div>
                  <h3 className="panel-title">Debug State</h3>
                  <span className="text-xs text-[#94a3b8]">Canonical backend game blob</span>
                </div>
                <div className="debug-state-actions">
                  <button className="btn-secondary" onClick={dumpDebugState} disabled={debugBusy}>
                    Dump State
                  </button>
                  <button className="btn-primary" onClick={loadDebugState} disabled={debugBusy}>
                    Load State
                  </button>
                </div>
              </div>
              <textarea
                className="debug-state-textarea"
                value={debugStateText}
                onChange={(event) => setDebugStateText(event.target.value)}
                placeholder="Dump or paste a canonical game blob"
                spellCheck={false}
              />
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
      <ImageHoverPreview preview={preview} />
    </div>
  );
}

export default App;
