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

static constexpr uint8_t MINOR_SPECIES_COUNT = 9;

static const MinorSpeciesData MINOR_SPECIES_TABLE[] = {
    { "Reputation Savants",       8, 0, MinorSpeciesAbility::VP_PER_REPUTATION,    0 },
    { "Dreadnought Allies",       4, 1, MinorSpeciesAbility::DISCOUNT_DREADNOUGHT, 2 },
    { "Ambassador Glyphs",        4, 0, MinorSpeciesAbility::VP_PER_AMBASSADOR,    0 },
    { "Orbital Architects",       4, 1, MinorSpeciesAbility::DISCOUNT_ORBITAL,     1 },
    { "Neutron Star Cult",        8, 3, MinorSpeciesAbility::FLAT_VP,              3 },
    { "Monolith Patrons",         6, 1, MinorSpeciesAbility::DISCOUNT_MONOLITH,    2 },
    { "Populous Traders",         9, 1, MinorSpeciesAbility::PLACE_POP_CUBE,       0 },
    { "Quantum Sages",            4, 1, MinorSpeciesAbility::DISCOUNT_RESEARCH,    1 },
    { "Cruiser Cartel",           4, 1, MinorSpeciesAbility::DISCOUNT_CRUISER,     1 },
};



#endif // ECLIPSE_MINOR_SPECIES_H
