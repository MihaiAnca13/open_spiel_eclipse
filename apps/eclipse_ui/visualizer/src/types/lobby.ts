export const API_BASE = 'http://127.0.0.1:8000';
export const WS_BASE = 'ws://127.0.0.1:8000';
export const SECTOR_ASSETS_BASE = `${API_BASE}/assets/sectors`;
export const TECH_ASSETS_BASE = `${API_BASE}/assets/tech`;
export const SHIP_ASSETS_BASE = `${API_BASE}/assets/ships`;

export function techImageUrl(name: string, category: string): string {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return `${TECH_ASSETS_BASE}/${category.toLowerCase()}/${slug}.png`;
}

export function shipImageUrl(type: string): string {
  return `${SHIP_ASSETS_BASE}/${type.toLowerCase()}.png`;
}

export interface TechMarketEntry {
  count: number;
  category: string;
  order?: number;
  base_cost: number;
  min_cost?: number;
  copies?: number;
}

export type TechCatalog = Record<string, Omit<TechMarketEntry, 'count'> & { count?: number }>;

export const TECH_CATEGORIES = ['Military', 'Grid', 'Nano', 'Rare'] as const;

const TECH_CATEGORY_ORDER: Record<string, number> = {
  Military: 0,
  Grid: 1,
  Nano: 2,
  Rare: 3,
};

export function sortTechMarketEntries<T extends TechMarketEntry>(
  entries: [string, T][]
): [string, T][] {
  return [...entries].sort(
    ([leftName, left], [rightName, right]) =>
      (TECH_CATEGORY_ORDER[left.category] ?? Number.MAX_SAFE_INTEGER) -
        (TECH_CATEGORY_ORDER[right.category] ?? Number.MAX_SAFE_INTEGER) ||
      left.base_cost - right.base_cost ||
      (left.order ?? Number.MAX_SAFE_INTEGER) - (right.order ?? Number.MAX_SAFE_INTEGER) ||
      leftName.localeCompare(rightName)
  );
}

export function buildTechMarketRows(
  techCatalog: TechCatalog,
  techTray: Record<string, TechMarketEntry>
): Record<string, [string, TechMarketEntry][]> {
  const merged: Record<string, TechMarketEntry> = {};

  for (const [name, tech] of Object.entries(techCatalog)) {
    merged[name] = {
      ...tech,
      count: techTray[name]?.count ?? 0,
    };
  }

  for (const [name, tech] of Object.entries(techTray)) {
    if (!merged[name]) {
      merged[name] = tech;
    }
  }

  const rows: Record<string, [string, TechMarketEntry][]> = {};
  for (const category of TECH_CATEGORIES) {
    rows[category] = sortTechMarketEntries(
      Object.entries(merged).filter(([, tech]) => tech.category === category)
    );
  }
  return rows;
}

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
