export interface SpeciesTheme {
  displayLabel: string;
  cssClass: string;
  hexColor: string;
}

export const SPECIES_THEME: Record<string, SpeciesTheme> = {
  'Eridani Empire': {
    displayLabel: 'Eridani Empire (Red)',
    cssClass: 'color-eridani',
    hexColor: '#7f1d1d',
  },
  'Hydran Progress': {
    displayLabel: 'Hydran Progress (Purple)',
    cssClass: 'color-hydran',
    hexColor: '#581c87',
  },
  'Planta': {
    displayLabel: 'Planta (Green)',
    cssClass: 'color-planta',
    hexColor: '#14532d',
  },
  'Orion Hegemony': {
    displayLabel: 'Orion Hegemony (Pink)',
    cssClass: 'color-orion',
    hexColor: '#9f1239',
  },
  'Descendants of Draco': {
    displayLabel: 'Descendants of Draco (Orange)',
    cssClass: 'color-draco',
    hexColor: '#7c2d12',
  },
  'Mechanema': {
    displayLabel: 'Mechanema (Grey)',
    cssClass: 'color-mechanema',
    hexColor: '#334155',
  },
  'Terran Factions': {
    displayLabel: 'Terran Factions (Blue)',
    cssClass: 'color-terran',
    hexColor: '#1e3a8a',
  },
};

export const DEFAULT_SPECIES_COLOR = '#1e293b';
export const UNOWNED_COLOR = '#1e293b';
export const NPC_COLOR_GCDS = '#3b0764';
export const NPC_COLOR_GUARDIAN = '#7c2d12';

export function getSpeciesHexColor(speciesName: string | undefined): string {
  return SPECIES_THEME[speciesName ?? '']?.hexColor ?? DEFAULT_SPECIES_COLOR;
}
