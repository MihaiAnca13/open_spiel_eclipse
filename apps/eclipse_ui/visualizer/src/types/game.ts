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

export type DieColorName = 'yellow' | 'orange' | 'blue' | 'red' | 'purple' | 'none';

export interface ShipPartDefinition {
  id: number;
  name: string;
  is_discovery: boolean;
  is_missile: boolean;
  external: boolean;
  die_color: DieColorName;
  die_amount: number;
  added_computer: number;
  added_shield: number;
  added_hull: number;
  net_energy: number;
  net_initiative: number;
  added_movement: number;
}

export type ShipPartCatalog = Record<string, ShipPartDefinition>;

export interface GameTechDefinition {
  category: string;
  order: number;
  base_cost: number;
  min_cost?: number;
  copies?: number;
}

export interface GameMetadata {
  species?: string[];
  tech_catalog?: Record<string, GameTechDefinition>;
  ship_part_catalog?: ShipPartCatalog;
  discovery_catalog?: DiscoveryCatalog;
  npc_difficulties?: string[];
}

export interface DiscoveryTileDefinition {
  id: number;
  name: string;
  copies: number;
  slug: string;
}

export type DiscoveryCatalog = Record<string, DiscoveryTileDefinition>;

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
  parts_inventory?: string[];
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
  discovery_tile?: number | string;
  orbital_built: boolean;
  monolith_built: boolean;
  player_warp_portal_vp?: number;
}

export interface PlanetLayout {
  slot: number;
  type: string;
  dx: number;
  dy: number;
}

export interface LayoutAnchor {
  dx: number;
  dy: number;
}

export interface SectorLayout {
  influence_space: LayoutAnchor;
  planets: PlanetLayout[];
  monolith_anchor: LayoutAnchor;
  orbital_anchor: LayoutAnchor;
  ship_anchors?: LayoutAnchor[];
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

export interface UpgradeState {
  phase: 'inactive' | 'choose_upgrade';
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

// ── Combat phase ── mirrors open_spiel/games/eclipse/systems/combat.h.
export type CombatPhase =
  | 'inactive'
  | 'determine_battles'
  | 'missile_phase'
  | 'choose_engagement_action'
  | 'engagement_firing'
  | 'select_reputation_tile'
  | 'attack_population'
  | 'influence_sectors'
  | 'discovery_award'
  | 'repair';

export interface InitiativeGroup {
  player_id: number;
  type: string;
  initiative: number;
  is_npc: boolean;
  destroyed: boolean;
  retreating: boolean;
  alive_count: number;
  destroyed_count: number;
}

export interface CombatSectorInfo {
  sector_id: number;
  participant_count: number;
  participant_ids: number[];
  latest_arrival: number[];
  defender_idx: number;
}

export interface DestroyedShipRecord {
  player_id: number;
  type: string;
  count: number;
  destroyed_by: number;
}

export interface CombatState {
  phase: CombatPhase;
  active_sector_id: number;
  battle_queue: CombatSectorInfo[];
  battle_queue_size: number;
  current_battle_idx: number;
  engagement_round: number;
  current_attacker_id: number;
  current_defender_id: number;
  pending_player: number;
  active_ship_type: string;
  initiative_timeline: InitiativeGroup[];
  initiative_size: number;
  initiative_idx: number;
  retreat_destinations: number[];
  retreat_destinations_size: number;
  drawn_tiles: string[];
  drawn_tiles_size: number;
  tile_select_player: number;
  reputation_earned: number;
  pending_target_group_player: number;
  pending_target_group_type: string;
  pending_target_indices: number[];
  pending_target_count: number;
  pending_die_values: number[];
  pending_die_colors: number[];
  pending_die_count: number;
  pending_die_index: number;
  pending_dice_are_missiles: boolean;
  pending_dice_pop_attack: boolean;
  destroyed_ships: DestroyedShipRecord[];
  destroyed_ships_size: number;
  pop_attack_sector_id: number;
  pop_attack_player: number;
  pop_attack_owner: number;
  pop_attack_damage_remaining: number;
  influence_decision_player: number;
  influence_decision_sector: number;
  discovery_decision_player: number;
  discovery_decision_sector: number;
}

// ── Upkeep / Cleanup phase ── mirrors UpkeepState in state.h.
export interface UpkeepState {
  step: 'inactive' | 'colony_ships' | 'bankruptcy' | 'cleanup_graveyards' | 'choose_return_track';
  player_id: number;
  pending_returns: PendingReturn[];
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
  current_phase: 'action' | 'combat' | 'upkeep' | 'cleanup';
  current_round: number;
  turn_order: number[];
  pass_order: number[];
  build_costs_by_player?: Record<string, BuildCosts>;
  scores?: Record<string, PlayerScoreBreakdown>;
  explore_state?: ExploreState;
  research_state?: ResearchState;
  influence_state?: InfluenceState;
  build_state?: BuildState;
  upgrade_state?: UpgradeState;
  move_state?: MoveState;
  combat_state?: CombatState;
  upkeep_state?: UpkeepState;
  warped_universe?: boolean;
  layout_kinds?: number[];
  warp_link_dest_cell?: number[];
  warp_link_dest_dir?: number[];
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
  warped_universe?: boolean;
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
