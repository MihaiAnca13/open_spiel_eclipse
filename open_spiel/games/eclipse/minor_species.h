//
// Created by Mihai on 28/05/2026.
//

#ifndef ECLIPSE_MINOR_SPECIES_H
#define ECLIPSE_MINOR_SPECIES_H

#include <cstddef>
#include <cstdint>

// Ability types for each Minor Species Ambassador Tile.
// Source: apps/eclipse_ui/minor_species.md
enum class MinorSpeciesAbility : uint8_t {
    NONE = 0,
    VP_PER_REPUTATION,    // 1 VP per Reputation Tile at game end
    DISCOUNT_DREADNOUGHT, // -N Materials discount for building Dreadnoughts
    VP_PER_AMBASSADOR,    // 1 VP per Ambassador Tile (including itself) at game end
    DISCOUNT_ORBITAL,     // -N Materials discount for building Orbitals
    FLAT_VP,              // Flat VP at game end (ability_param = amount)
    DISCOUNT_MONOLITH,    // -N Materials discount for building Monoliths
    PLACE_POP_CUBE,       // Immediately place 1 Pop Cube from a chosen track
    DISCOUNT_RESEARCH,    // -N Science discount for Researching Techs (min cost still applies)
    DISCOUNT_CRUISER,     // -N Materials discount for building Cruisers
};

struct MinorSpeciesData {
    const char* name;            // Display name (placeholder until verified)
    uint8_t cost;                // Money cost to form Diplomatic Relations
    uint8_t end_vp;              // Flat end-of-game VP
    MinorSpeciesAbility ability; // Triggered ability
    uint8_t ability_param;       // Discount magnitude (1 or 2), or N/A when unused
};

// Data extracted from apps/eclipse_ui/minor_species.md.
// Money costs are not visible in the markdown text (they live in icon
// images in the rulebook); left as 0 here pending verification.
//   1. 1 VP per Reputation Tile
//   2. -2 Materials for Dreadnoughts; 1 VP
//   3. 1 VP per Ambassador Tile (including itself)
//   4. -1 Materials for Orbitals; 1 VP
//   5. 3 VP
//   6. -2 Materials for Monoliths; 1 VP
//   7. Place 1 Pop Cube from a chosen track; 1 VP
//   8. -1 Science for Researching Techs (min cost still applies); 1 VP
//   9. -1 Materials for Cruisers; 1 VP
static const MinorSpeciesData MINOR_SPECIES_TABLE[] = {
    { "Minor Species 1", 0, 0, MinorSpeciesAbility::VP_PER_REPUTATION,    0 },
    { "Minor Species 2", 0, 1, MinorSpeciesAbility::DISCOUNT_DREADNOUGHT, 2 },
    { "Minor Species 3", 0, 0, MinorSpeciesAbility::VP_PER_AMBASSADOR,    0 },
    { "Minor Species 4", 0, 1, MinorSpeciesAbility::DISCOUNT_ORBITAL,     1 },
    { "Minor Species 5", 0, 3, MinorSpeciesAbility::FLAT_VP,              3 },
    { "Minor Species 6", 0, 1, MinorSpeciesAbility::DISCOUNT_MONOLITH,    2 },
    { "Minor Species 7", 0, 1, MinorSpeciesAbility::PLACE_POP_CUBE,       0 },
    { "Minor Species 8", 0, 1, MinorSpeciesAbility::DISCOUNT_RESEARCH,    1 },
    { "Minor Species 9", 0, 1, MinorSpeciesAbility::DISCOUNT_CRUISER,     1 },
};

static constexpr size_t MINOR_SPECIES_COUNT =
    sizeof(MINOR_SPECIES_TABLE) / sizeof(MINOR_SPECIES_TABLE[0]);

#endif // ECLIPSE_MINOR_SPECIES_H
