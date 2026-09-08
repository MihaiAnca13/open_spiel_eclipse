import type { MinorSpeciesDef } from '../types/game';

const ABILITY_TEXT: Record<number, (amount: number) => string> = {
  0: () => '',
  1: () => '1 VP per Reputation tile at game end',
  2: (amount) => `−${amount} Materials for Dreadnoughts`,
  3: () => '1 VP per Ambassador tile at game end',
  4: (amount) => `−${amount} Materials for Orbitals`,
  5: () => '',
  6: (amount) => `−${amount} Materials for Monoliths`,
  7: () => 'Place 1 Population Cube',
  8: (amount) => `−${amount} Science for Research`,
  9: (amount) => `−${amount} Materials for Cruisers`,
};

export function minorSpeciesEffectText({
  ability,
  ability_param: abilityParam,
  end_vp: endVp,
}: MinorSpeciesDef): string {
  return [ABILITY_TEXT[ability]?.(abilityParam), endVp > 0 ? `${endVp} VP at game end` : '']
    .filter(Boolean)
    .join(' · ');
}
