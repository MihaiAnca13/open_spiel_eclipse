export interface SpeciesTheme {
  displayLabel: string;
  cssClass: string;
  hexColor: string;
  isTerran: boolean;
}

export const SPECIES_THEME: Record<string, SpeciesTheme> = {
  'Eridani Empire': {
    displayLabel: 'Eridani Empire (Red)',
    cssClass: 'color-eridani',
    hexColor: '#7f1d1d',
    isTerran: false,
  },
  'Hydran Progress': {
    displayLabel: 'Hydran Progress (Purple)',
    cssClass: 'color-hydran',
    hexColor: '#581c87',
    isTerran: false,
  },
  'Planta': {
    displayLabel: 'Planta (Green)',
    cssClass: 'color-planta',
    hexColor: '#14532d',
    isTerran: false,
  },
  'Orion Hegemony': {
    displayLabel: 'Orion Hegemony (Pink)',
    cssClass: 'color-orion',
    hexColor: '#9f1239',
    isTerran: false,
  },
  'Descendants of Draco': {
    displayLabel: 'Descendants of Draco (Orange)',
    cssClass: 'color-draco',
    hexColor: '#7c2d12',
    isTerran: false,
  },
  'Mechanema': {
    displayLabel: 'Mechanema (Grey)',
    cssClass: 'color-mechanema',
    hexColor: '#334155',
    isTerran: false,
  },
  'Terran Factions': {
    displayLabel: 'Terran Factions (Blue)',
    cssClass: 'color-terran',
    hexColor: '#1e3a8a',
    isTerran: true,
  },
};

// Seat colors: identity comes from the player slot, not the species (two
// players can pick the same species). Indexed by player/owner id (0-5).
export const PLAYER_COLORS = [
  '#3b82f6', // 0 blue
  '#ec4899', // 1 pink
  '#22c55e', // 2 green
  '#f97316', // 3 orange
  '#a855f7', // 4 purple
  '#06b6d4', // 5 cyan
];

export function getPlayerColor(playerId: number | undefined): string {
  return PLAYER_COLORS[playerId ?? -1] ?? DEFAULT_SPECIES_COLOR;
}

export const DEFAULT_SPECIES_COLOR = '#1e293b';
export const UNOWNED_COLOR = '#1e293b';
export const NPC_COLOR_GCDS = '#3b0764';
export const NPC_COLOR_GUARDIAN = '#7c2d12';

export function getSpeciesHexColor(speciesName: string | undefined): string {
  return SPECIES_THEME[speciesName ?? '']?.hexColor ?? DEFAULT_SPECIES_COLOR;
}
