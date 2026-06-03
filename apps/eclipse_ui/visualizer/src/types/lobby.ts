export const API_BASE = 'http://127.0.0.1:8000';
export const WS_BASE = 'ws://127.0.0.1:8000';
export const SECTOR_ASSETS_BASE = `${API_BASE}/assets/sectors`;

export type SeatState = 'empty' | 'human' | 'ai';
export type LobbyPhase = 'waiting' | 'setup' | 'started';

export interface LobbySeat {
  state: SeatState;
  player_id: string | null;
  player_name: string | null;
  species: string;
  last_player_id: string | null;
}

export interface LobbyData {
  host_player_id: string | null;
  num_players: number;
  seats: LobbySeat[];
  difficulty: string;
  rng_seed: number;
  phase: LobbyPhase;
  picker_order: number[];
  current_picker_idx: number;
  stage1_snapshot?: unknown;
  snapshot?: unknown;
}
