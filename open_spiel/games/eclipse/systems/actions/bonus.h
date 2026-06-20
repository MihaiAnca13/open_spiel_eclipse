//
// Bonus actions: free actions usable at any time during a player's turn
// without spending an influence disc and without advancing the turn.
// Currently: Trade and Colony Ship placement.
//

#ifndef ECLIPSE_BONUS_H
#define ECLIPSE_BONUS_H

#include <vector>
#include <utility>
#include <cstdint>

#include "open_spiel/games/eclipse/state.h"

// ── Minor Species Diplomatic Relations ────────────────────────────────────────
// TODO: Forming Diplomatic Relations with a Minor Species is a free action
// usable at any time during a player's turn (similar to Trade / Colony Ships).
// Look up the tile's data in open_spiel/games/eclipse/minor_species.h
// (MINOR_SPECIES_TABLE) for the Money cost, end-game VP, and triggered
// ability. Effects trigger immediately on formation and the tile cannot be
// discarded afterwards.
// Pay trade_rate of one resource, gain 1 of another.

enum class TradeConversion : uint8_t {
    GOLD_TO_SCIENCE    = 0,
    GOLD_TO_MATERIALS  = 1,
    SCIENCE_TO_GOLD    = 2,
    SCIENCE_TO_MATERIALS = 3,
    MATERIALS_TO_GOLD  = 4,
    MATERIALS_TO_SCIENCE = 5,
};

constexpr int TRADE_CONVERSION_COUNT = 6;

bool can_trade(const ::Player& player, TradeConversion conv);
bool execute_trade(State& state, uint8_t player_id, TradeConversion conv);

// ── Colony Ships ──────────────────────────────────────────────────────────────
// Flip a colony ship facedown to move one Population Cube from a chosen
// Population Track to an empty slot in a Sector you Control.
//
// Cubes are all the player's colour: the track a cube comes from is the
// player's choice at placement. For a fixed-type slot (MONEY/SCIENCE/MATERIALS
// and their advanced variants) only the matching track is legal. For an
// ANY/ADV_ANY slot any non-empty track may be chosen.
//
// galaxy_cell_idx: flat index into the 15×15 galaxy grid (hex_to_index result)
// slot_idx:        index into SectorDefinition::slots for the target sector
// track:           0=Money(gold), 1=Science, 2=Materials

enum class PopTrack : uint8_t { MONEY = 0, SCIENCE = 1, MATERIALS = 2 };
constexpr int POP_TRACK_COUNT = 3;

bool can_use_colony_ship(const State& state, uint8_t player_id,
                         uint8_t galaxy_cell_idx, uint8_t slot_idx,
                         PopTrack track);
bool use_colony_ship(State& state, uint8_t player_id,
                     uint8_t galaxy_cell_idx, uint8_t slot_idx,
                     PopTrack track);

// Flip up to `count` facedown colony ships faceup (called by Influence action).
void refresh_colony_ships(::Player& player, uint8_t count = 2);

struct ColonyPlacement {
    uint8_t cell;
    uint8_t slot;
    PopTrack track;
};

// Enumerate all legal (cell, slot, track) placements for the given player.
std::vector<ColonyPlacement> legal_colony_ship_placements(
    const State& state, uint8_t player_id);

// ── Player Warp Portal Placement ──────────────────────────────────────────────
// TODO: Implement Warp Portal placement logic later.
// 1. Verify player is eligible (either by claiming the Warp Portal Discovery Tile
//    or having researched the Warp Portal Rare Tech).
// 2. Consume the eligibility flag/tile.
// 3. Mark the target sector with a 1 VP Rare Tech Warp Portal tile.
bool can_place_warp_portal(const State& state, uint8_t player_id, uint8_t galaxy_cell_idx);
bool place_warp_portal(State& state, uint8_t player_id, uint8_t galaxy_cell_idx);

// ── Population Track helpers ──────────────────────────────────────────────────
// Track state is stored in Resources::gold_prod / science_prod / materials_prod
// as the number of cubes REMAINING ON THE TRACK (0=empty, 12=full).
// Production = POPULATION_PRODUCTION_TABLE[cubes_on_track].

constexpr uint8_t POP_TRACK_MAX = 12; // cubes per track

#endif // ECLIPSE_BONUS_H
