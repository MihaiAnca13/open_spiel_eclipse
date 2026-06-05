import {
  getPlayerColor,
  NPC_COLOR_GCDS,
  NPC_COLOR_GUARDIAN,
  UNOWNED_COLOR,
} from '../theme';

export const EMPTY_LEGAL_ACTIONS: number[] = [];

// Remaining tiles in a ring bag = set bits in its bitmask.
export function bagCount(mask: number | undefined): number {
  let m = (mask ?? 0) >>> 0;
  let c = 0;
  while (m) {
    m &= m - 1;
    c++;
  }
  return c;
}

// Influence disc model — mirrors open_spiel/games/eclipse: total discs and the
// upkeep cost exposed as discs leave the track (state.h INFLUENCE_UPKEEP_TABLE).
export const INFLUENCE_TOTAL = 12;
export const INFLUENCE_UPKEEP = [0, 0, 1, 2, 3, 5, 7, 10, 13, 17, 21, 25, 30];

// Production table: index = cubes remaining on track (12=full=0 prod, 0=empty=28 prod)
export const POPULATION_PRODUCTION_TABLE = [28, 24, 21, 18, 15, 12, 10, 8, 6, 4, 3, 2, 0] as const;
export const POP_TRACK_MAX = 12;
export const POP_TRACK_STEPS = POP_TRACK_MAX - 1;

export function getPlayerHexColor(ownerId: number, sectorId: number): string {
  if (ownerId === 255) {
    if (sectorId >= 271 && sectorId <= 274) return NPC_COLOR_GUARDIAN;
    if (sectorId === 1) return NPC_COLOR_GCDS;
    return UNOWNED_COLOR;
  }
  return getPlayerColor(ownerId);
}
