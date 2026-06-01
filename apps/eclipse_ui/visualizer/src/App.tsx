import { useState } from 'react';
import './App.css';

const API_BASE = "http://127.0.0.1:8000";

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
  colony_ships: boolean[];
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
  tech_tray: Record<string, number>;
  tech_bag: string[];
  gcds_difficulty: string;
  guardian_difficulty: string;
  ancient_difficulty: string;
  current_player: number;
  current_phase: number;
  current_round: number;
  turn_order: number[];
  pass_order: number[];
}

const SPECIES_OPTIONS = [
  { value: "Eridani Empire", label: "Eridani Empire (Red)" },
  { value: "Hydran Progress", label: "Hydran Progress (Purple)" },
  { value: "Planta", label: "Planta (Green)" },
  { value: "Orion Hegemony", label: "Orion Hegemony (Pink)" },
  { value: "Descendants of Draco", label: "Descendants of Draco (Orange)" },
  { value: "Mechanema", label: "Mechanema (Grey)" },
  { value: "Terran Factions", label: "Terran Factions (Blue)" }
];

function App() {
  const [seed, setSeed] = useState<number>(42);
  const [numPlayers, setNumPlayers] = useState<number>(4);
  const [difficulty, setDifficulty] = useState<string>("Easy");

  const [state, setState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [jsonExpanded, setJsonExpanded] = useState<boolean>(false);
  const [hoveredSector, setHoveredSector] = useState<Sector | null>(null);

  // Selected species choices for each pre-allocated player index
  const [speciesChoices, setSpeciesChoices] = useState<Record<number, string>>({
    0: "Terran Factions",
    1: "Planta",
    2: "Orion Hegemony",
    3: "Hydran Progress",
    4: "Mechanema",
    5: "Eridani Empire"
  });

  const [aiChoices, setAiChoices] = useState<Record<number, boolean>>({
    0: false,
    1: true,
    2: true,
    3: true,
    4: true,
    5: true
  });

  const handleInitializeStage1 = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/setup/pre-choice?seed=${seed}&num_players=${numPlayers}&difficulty=${difficulty}`);
      if (!response.ok) {
        throw new Error("FastAPI server is not responding. Make sure apps/eclipse_ui/run.sh or the backend is running.");
      }
      const data = await response.json();
      setState(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleFinalizeStage2 = async () => {
    if (!state) return;
    setLoading(true);
    setError(null);

    // Format player choices to match standard API PlayerConfig: [{'species': 'Planta', 'is_ai': false}, ...]
    const choicesList = state.players.map((p) => ({
      species: speciesChoices[p.id] || "Terran Factions",
      is_ai: aiChoices[p.id] ?? false
    }));

    try {
      const response = await fetch(`${API_BASE}/setup/finalize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed: seed,
          state: state,
          player_choices: choicesList
        })
      });
      if (!response.ok) {
        throw new Error("Stage 2 Setup failed. Check model bindings on FastAPI console.");
      }
      const data = await response.json();
      setState(data);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleSpeciesChange = (playerId: number, species: string) => {
    setSpeciesChoices((prev) => ({ ...prev, [playerId]: species }));
  };

  const handleAiChange = (playerId: number, isAi: boolean) => {
    setAiChoices((prev) => ({ ...prev, [playerId]: isAi }));
  };

  // Hex Math helpers
  const hexSize = 35;
  const viewBoxWidth = 600;
  const viewBoxHeight = 520;
  const centerX = viewBoxWidth / 2;
  const centerY = viewBoxHeight / 2;

  const getHexPoints = (cx: number, cy: number, r: number) => {
    const points = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 180) * (30 + 60 * i); // Pointy-topped
      points.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
    }
    return points.join(" ");
  };

  const getPlayerColorClass = (speciesName: string) => {
    switch (speciesName) {
      case "Eridani Empire": return "color-eridani";
      case "Hydran Progress": return "color-hydran";
      case "Planta": return "color-planta";
      case "Descendants of Draco": return "color-draco";
      case "Mechanema": return "color-mechanema";
      case "Orion Hegemony": return "color-orion";
      case "Terran Factions": return "color-terran";
      default: return "";
    }
  };

  const getPlayerHexColor = (ownerId: number, sectorId: number) => {
    if (ownerId === 255) {
      if (sectorId >= 271 && sectorId <= 274) return "#7c2d12"; // NPC Guardian Orange
      if (sectorId === 1) return "#3b0764"; // Center Dark Purple
      return "#1e293b"; // Unowned
    }
    // Colored by owner species
    const p = state?.players[ownerId];
    if (!p) return "#1e293b";
    switch (p.species_id) {
      case "Eridani Empire": return "#7f1d1d"; // Dark Red
      case "Hydran Progress": return "#581c87"; // Dark Purple
      case "Planta": return "#14532d"; // Dark Green
      case "Descendants of Draco": return "#7c2d12"; // Orange
      case "Mechanema": return "#334155"; // Gray
      case "Orion Hegemony": return "#9f1239"; // Rose
      case "Terran Factions": return "#1e3a8a"; // Dark Blue
      default: return "#1e293b";
    }
  };

  const getUnitsInSector = (sectorId: number) => {
    if (!state) return [];
    return state.unit_registry.filter((u) => u.sector_id === sectorId);
  };

  // Render the hexagonal grid
  const activeSectors: { sector: Sector; cx: number; cy: number }[] = [];
  if (state?.galaxy) {
    const MAP_SIZE = state.galaxy.length;
    const OFFSET = 7;
    for (let q_idx = 0; q_idx < MAP_SIZE; q_idx++) {
      for (let r_idx = 0; r_idx < MAP_SIZE; r_idx++) {
        const sector = state.galaxy[q_idx][r_idx];
        if (sector && sector.sector_id > 0) {
          const q = q_idx - OFFSET;
          const r = r_idx - OFFSET;
          const cx = centerX + hexSize * (Math.sqrt(3) * q + (Math.sqrt(3) / 2) * r);
          const cy = centerY + hexSize * (1.5 * r);
          activeSectors.push({ sector, cx, cy });
        }
      }
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>Eclipse setup & map visualizer</h1>
        <p>Interactive 2-Stage setup interface connecting React TS to High-Performance C++ Core API</p>
      </header>

      {error && (
        <div style={{ backgroundColor: "#7f1d1d", border: "1px solid #f87171", padding: "12px", borderRadius: "8px", marginBottom: "16px", fontSize: "14px" }}>
          <strong>Error connecting to backend:</strong> {error}
        </div>
      )}

      <div className="main-layout">
        {/* Left column: Setup controls */}
        <div className="panel">
          <div>
            <h3 className="panel-title">Stage 1: Pre-Choice Setup</h3>
            <span style={{ fontSize: "12px", color: "#94a3b8" }}>Initializes RNG, draws Tech Tray and NPCs</span>
          </div>

          <div className="form-group">
            <label>RNG Seed (0 for random)</label>
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </div>

          <div className="form-group">
            <label>Number of Players ({numPlayers})</label>
            <input type="range" min="2" max="6" value={numPlayers} onChange={(e) => setNumPlayers(Number(e.target.value))} />
          </div>

          <div className="form-group">
            <label>NPC Difficulty (GCDS & Guardians)</label>
            <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
              <option value="Easy">Easy (Standard)</option>
              <option value="Medium">Medium (Advanced)</option>
              <option value="Hard">Hard (Expert)</option>
            </select>
          </div>

          <button className="btn-primary" onClick={handleInitializeStage1} disabled={loading}>
            {loading ? "Processing C++..." : "Generate Pre-Choice Board"}
          </button>

          {state && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px", borderTop: "1px solid #2d313f", paddingTop: "16px", marginTop: "8px" }}>
              <div>
                <h3 className="panel-title">Stage 2: Commit Species</h3>
                <span style={{ fontSize: "12px", color: "#94a3b8" }}>Selection executes in reverse turn order</span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {state.players.map((p) => {
                  const currentTurnIdx = state.turn_order.indexOf(p.id);
                  const isFirstPicker = currentTurnIdx === state.players.length - 1; // last in turn order picks first!
                  return (
                    <div key={p.id} className={`player-choice-card ${isFirstPicker ? "active" : ""}`}>
                      <div className="player-card-header">
                        <span style={{ fontSize: "13px", fontWeight: "bold" }}>Player {p.id}</span>
                        <span className="player-badge">
                          {isFirstPicker ? "🔥 Picker 1" : `Turn Position: ${currentTurnIdx + 1}`}
                        </span>
                      </div>

                      <div className="form-group" style={{ gap: "4px" }}>
                        <select
                          value={speciesChoices[p.id] || "Terran Factions"}
                          onChange={(e) => handleSpeciesChange(p.id, e.target.value)}
                        >
                          {SPECIES_OPTIONS.map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      </div>

                      <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px" }}>
                        <input
                          type="checkbox"
                          id={`ai-${p.id}`}
                          checked={aiChoices[p.id] ?? false}
                          onChange={(e) => handleAiChange(p.id, e.target.checked)}
                        />
                        <label htmlFor={`ai-${p.id}`} style={{ color: "#cbd5e1", cursor: "pointer" }}>Control as AI Agent</label>
                      </div>
                    </div>
                  );
                })}
              </div>

              <button className="btn-secondary" onClick={handleFinalizeStage2} disabled={loading}>
                {loading ? "Processing C++..." : "Finalize Species & Spawn Map"}
              </button>
            </div>
          )}
        </div>

        {/* Right column: board visualizer */}
        <div className="board-container">
          {state && (
            <div className="turn-order-bar">
              <span style={{ fontSize: "12px", textTransform: "uppercase", color: "#94a3b8", fontWeight: "bold", marginRight: "8px" }}>Active Turn Order:</span>
              {state.turn_order.map((pid, idx) => {
                if (pid === 255) return null;
                const player = state.players[pid];
                const isActive = state.current_player === pid;
                return (
                  <div key={pid} className={`turn-badge ${isActive ? "active" : ""}`}>
                    <span className="turn-num">{idx + 1}</span>
                    <span className={getPlayerColorClass(player?.species_id)}>P{pid} ({player?.species_id || "Choosing..."})</span>
                    {player?.is_ai && <span style={{ fontSize: "9px", backgroundColor: "#334155", color: "#93c5fd", padding: "1px 4px", borderRadius: "4px" }}>AI</span>}
                  </div>
                );
              })}
            </div>
          )}

          <div className="map-viewport">
            {state ? (
              <svg width="100%" height="100%" viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}>
                {/* Render active sector hexes */}
                {activeSectors.map(({ sector, cx, cy }) => {
                  const fillColor = getPlayerHexColor(sector.owner_id, sector.sector_id);
                  const isCenter = sector.sector_id === 1;
                  const units = getUnitsInSector(sector.sector_id);
                  
                  return (
                    <g
                      key={sector.sector_id}
                      onMouseEnter={() => setHoveredSector(sector)}
                      onMouseLeave={() => setHoveredSector(null)}
                    >
                      {/* Polygon hex */}
                      <polygon
                        points={getHexPoints(cx, cy, hexSize - 1.5)}
                        fill={fillColor}
                        stroke={isCenter ? "#eab308" : "#475569"}
                        strokeWidth={isCenter ? "2.5" : "1.5"}
                        className="hex-polygon"
                      />

                      {/* Text details inside hex */}
                      <text
                        x={cx}
                        y={cy - 6}
                        textAnchor="middle"
                        fill="#f8fafc"
                        fontSize="11px"
                        fontWeight="bold"
                        style={{ pointerEvents: "none" }}
                      >
                        {sector.sector_id === 1 ? "GCDS" : sector.sector_id}
                      </text>

                      {/* Coords helper */}
                      <text
                        x={cx}
                        y={cy + 8}
                        textAnchor="middle"
                        fill="#94a3b8"
                        fontSize="8px"
                        style={{ pointerEvents: "none" }}
                      >
                        ({sector.coords.q},{sector.coords.r})
                      </text>

                      {/* Spawns indicators / Ships count inside sector */}
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
              </svg>
            ) : (
              <div style={{ textAlign: "center", color: "#64748b" }}>
                <svg style={{ width: "64px", height: "64px", marginBottom: "16px", color: "#334155" }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
                </svg>
                <h3>Galaxy Map Uninitialized</h3>
                <p style={{ fontSize: "14px" }}>Click "Generate Pre-Choice Board" to spin up the Stage 1 C++ engine</p>
              </div>
            )}

            {/* Hover details overlay */}
            {hoveredSector && (
              <div className="galaxy-info-overlay">
                <div className="galaxy-info-item">
                  <span>Sector ID</span>
                  <span>{hoveredSector.sector_id === 1 ? "001 (Center)" : hoveredSector.sector_id}</span>
                </div>
                <div className="galaxy-info-item">
                  <span>Coordinates</span>
                  <span>({hoveredSector.coords.q}, {hoveredSector.coords.r})</span>
                </div>
                <div className="galaxy-info-item">
                  <span>Owner</span>
                  <span>
                    {hoveredSector.owner_id === 255 ? "Unowned (Neutral)" : `Player ${hoveredSector.owner_id} (${state?.players[hoveredSector.owner_id]?.species_id})`}
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
                      {getUnitsInSector(hoveredSector.sector_id).map((u) => `${u.type} (P${u.player_id})`).join(", ")}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Tech market display */}
          {state && (
            <div className="panel">
              <h3 className="panel-title">Round 1 Technology Market</h3>
              <div className="tech-grid">
                {Object.entries(state.tech_tray).map(([techName, count]) => {
                  // Standard Eclipse Second Dawn Tech Classification matching C++ Table
                  let category = "Military";
                  if (["Gauss Shield", "Fusion Source", "Improved Hull", "Positron Computer", "Advanced Economy", "Tachyon Drive", "Antimatter Cannon", "Quantum Grid"].includes(techName)) {
                    category = "Grid";
                  } else if (["Nanorobots", "Fusion Drive", "Orbital", "Advanced Robotics", "Advanced Labs", "Monolith", "Wormhole Generator", "Artifact Key"].includes(techName)) {
                    category = "Nano";
                  } else if (["Absorption Shield", "Ancient Labs", "Antimatter Splitter", "Cloaking Device", "Conifold Field", "Flux Missile", "Improved Logistics", "Metasynthesis", "Neutron Absorber", "Pico Modulator", "Sentient Hull", "Soliton Cannon", "Transition Drive", "Warp Portal", "Zero Point Source", "Rift Cannon"].includes(techName)) {
                    category = "Rare";
                  }

                  return (
                    <div key={techName} className={`tech-card ${category}`}>
                      <div className="tech-name">{techName}</div>
                      <span className={`tech-category-badge ${category}`}>{category}</span>
                      <div className="tech-count" style={{ marginTop: "6px" }}>
                        Qty Available: <strong>{count}</strong>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Collapsible raw JSON response inspector */}
          {state && (
            <div className="json-inspector">
              <div className="json-title" onClick={() => setJsonExpanded(!jsonExpanded)}>
                <span>🔍 Inspect Raw C++ State JSON payload</span>
                <span>{jsonExpanded ? "Collapse ▲" : "Expand ▼"}</span>
              </div>
              {jsonExpanded && (
                <pre className="json-code">
                  {JSON.stringify(state, null, 2)}
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
