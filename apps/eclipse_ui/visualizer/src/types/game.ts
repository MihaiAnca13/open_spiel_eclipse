import type { ExploreState, ResearchState } from '../ActionPanel';
import type { TechMarketEntry } from './lobby';

export interface HexCoord {
  q: number;
  r: number;
}

export interface Resources {
  gold: number;
  science: number;
  materials: number;
  gold_prod: number;
  science_prod: number;
  materials_prod: number;
}

export interface ShipStats {
  initiative: number;
  computer: number;
  shield: number;
  energy_net: number;
  hull: number;
  movement: number;
  cannons: number[];
  missiles: number[];
}

export interface Blueprint {
  slots: string[];
  capacity: number;
  base_stats: ShipStats;
  total_stats: ShipStats;
}

export interface BuildCosts {
  Interceptor: number;
  Cruiser: number;
  Dreadnought: number;
  Starbase: number;
  Orbital: number;
  Monolith: number;
}

export interface Player {
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
  researched_techs_military: number;
  researched_techs_grid: number;
  researched_techs_nano: number;
}

export interface Sector {
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

export interface PlanetLayout {
  slot: number;
  type: string;
  dx: number;
  dy: number;
}

export interface SectorLayout {
  influence_space: { dx: number; dy: number };
  planets: PlanetLayout[];
  monolith_anchor: { dx: number; dy: number };
  orbital_anchor: { dx: number; dy: number };
}

export interface Unit {
  player_id: number;
  type: string;
  sector_id: number;
  damage: number;
}

export interface PendingReturn {
  type: string;
  is_orbital: boolean;
}

export interface InfluenceState {
  phase: 'inactive' | 'choose_influence' | 'choose_return_track';
  player_id: number;
  activations_remaining: number;
  pending_returns: PendingReturn[];
}

export interface BuildState {
  phase: 'inactive' | 'choose_build';
  player_id: number;
  activations_remaining: number;
}

export interface MoveState {
  phase: 'inactive' | 'choose_move' | 'choose_warp_destination';
  player_id: number;
  activations_remaining: number;
  active_unit_idx: number;
  steps_remaining: number;
  warp_unit_idx: number;
}

export interface PlayerScoreBreakdown {
  reputation_vp: number;
  ambassador_vp: number;
  sector_vp: number;
  monolith_vp: number;
  discovery_vp: number;
  tech_track_vp: number;
  traitor_vp: number;
  species_vp: number;
  total_vp: number;
}

export interface GameState {
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
  build_costs_by_player?: Record<string, BuildCosts>;
  scores?: Record<string, PlayerScoreBreakdown>;
  explore_state?: ExploreState;
  research_state?: ResearchState;
  influence_state?: InfluenceState;
  build_state?: BuildState;
  move_state?: MoveState;
  sector_bag_inner: number;
  sector_bag_middle: number;
  sector_bag_outer: number;
}

export interface StagedPlayerConfig {
  species: string | null;
  is_ai: boolean;
  starting_sector?: number;
}

export interface SetupConfig {
  players: number;
  rng_seed: number;
  npc_difficulty: string;
  staged_players: StagedPlayerConfig[];
}

export interface SetupSnapshot {
  config: SetupConfig;
  state: GameState;
  finalized: boolean;
  legal_actions?: number[];
  action_strings?: Record<string, string>;
  current_player?: number;
  is_terminal?: boolean;
}
