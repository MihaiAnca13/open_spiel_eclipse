//
// Created by Mihai on 28/05/2026.
//

#include "research.h"
#include <algorithm>

#include "../../state.h"
#include "../../tech.h"


namespace open_spiel::eclipse {

uint8_t get_track_tile_count(const ::Player& player, TechCategory category) {
    uint64_t mask;
    uint64_t range;
    switch (category) {
        case TechCategory::MILITARY:
            mask = player.researched_techs_military;
            range = MIL_STANDARD_RANGE;
            break;
        case TechCategory::GRID:
            mask = player.researched_techs_grid;
            range = GRID_STANDARD_RANGE;
            break;
        case TechCategory::NANO:
            mask = player.researched_techs_nano;
            range = NANO_STANDARD_RANGE;
            break;
        case TechCategory::RARE:
            return 0;  // rare techs are tracked via track masks
    }
    return static_cast<uint8_t>(__builtin_popcount(mask & (RARE_BITMASK | range)));
}

uint8_t calculate_research_cost(const ::Player& player, const TechDefinition& tech_def, TechCategory target_track) {
    // Discount = number of tiles already on target track (standard + rare).
    // Rare techs both give and receive discounts just like standard techs.
    uint8_t discount = get_track_tile_count(player, target_track);

    int16_t cost = static_cast<int16_t>(tech_def.base_cost) - discount;
    return static_cast<uint8_t>(std::max(static_cast<int16_t>(tech_def.min_cost), cost));
}

bool research_tech(::State& state, uint8_t player_id, const TechDefinition& tech_def, TechCategory target_track) {
    if (player_id >= state.players.size()) return false;
    ::Player& player = state.players[player_id];

    // Already owned
    if (player.has_tech(tech_def.bit)) return false;

    // Check availability in tech tray
    if (state.get_tech_tray_count(tech_def.bit) == 0) return false;

    // Validate target track
    if (tech_def.category != TechCategory::RARE && tech_def.category != target_track) {
        return false; // Standard tech must go to its own track
    }
    if (tech_def.category == TechCategory::RARE) {
        size_t track_idx = static_cast<size_t>(target_track);
        if (track_idx >= 3) return false; // Invalid track index
    }

    // Check track capacity (max 8 tiles per track)
    if (get_track_tile_count(player, target_track) >= 8) {
        return false;
    }

    uint8_t cost = calculate_research_cost(player, tech_def, target_track);
    if (player.resources.science < cost) return false;

    // Subtract cost and place the tile on exactly one tech track.
    player.resources.science -= cost;
    uint64_t bit = static_cast<uint64_t>(tech_def.bit);
    switch (target_track) {
        case TechCategory::MILITARY: player.researched_techs_military |= bit; break;
        case TechCategory::GRID:     player.researched_techs_grid |= bit; break;
        case TechCategory::NANO:     player.researched_techs_nano |= bit; break;
        case TechCategory::RARE:     return false;  // rejected above
    }

    // Remove from market tray
    state.remove_from_tech_tray(tech_def.bit);

    // Rulebook page 7: researching the Warp Portal Rare Tech grants the player
    // a Warp Portal Tile that may be placed on any Controlled Sector.
    if (tech_def.bit == TechBit::WARP_PORTAL) {
        player.warp_portal_eligible = true;
    }

    return true;
}

} // namespace open_spiel::eclipse

