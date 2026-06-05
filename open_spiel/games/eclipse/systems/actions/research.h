//
// Created by Mihai on 28/05/2026.
//

#ifndef ECLIPSE_RESEARCH_H
#define ECLIPSE_RESEARCH_H

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>

#include "../../tech.h"

// The raw game state and player structs live in the global namespace (state.h);
// forward-declare them so this header does not depend on state.h.
struct State;
struct Player;

// Research state for multi-activation research actions
struct ResearchState {
    enum class Phase : uint8_t { inactive = 0, choose_tech = 1 };
    Phase phase = Phase::inactive;
    uint8_t player_id = 255;
    uint8_t activations_remaining = 0;
};

NLOHMANN_JSON_SERIALIZE_ENUM(ResearchState::Phase, {
    {ResearchState::Phase::inactive, "inactive"},
    {ResearchState::Phase::choose_tech, "choose_tech"},
});

inline std::ostream& operator<<(std::ostream& os, ResearchState::Phase phase) {
    switch (phase) {
        case ResearchState::Phase::inactive:
            return os << "inactive";
        case ResearchState::Phase::choose_tech:
            return os << "choose_tech";
        default:
            return os << "unknown";
    }
}

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ResearchState, phase, player_id, activations_remaining);


namespace open_spiel::eclipse {

// Returns the number of tiles (standard + rare) on a given track for the player.
uint8_t get_track_tile_count(const ::Player& player, TechCategory category);

// Returns the final cost to research the given technology for the player.
// For Rare Techs, target_track determines which track's discount to apply.
// For standard Techs, target_track must match the tech's category.
// Takes a pointer to the tech definition to avoid redundant TECH_TABLE lookups.
uint8_t calculate_research_cost(const ::Player& player, const TechDefinition& tech_def, TechCategory target_track);

// Researches a technology for a player.
// Subtracts science resources, adds the technology to the player's bitmask, and decrements/updates the tech tray.
// Takes a pointer to the tech definition to avoid redundant TECH_TABLE lookups.
// For Rare Techs, target_track controls which track receives the tile.
// Returns true on success, false if the tech is not researchable (already researched, too expensive, not in tech tray, or track full).
bool research_tech(::State& state, uint8_t player_id, const TechDefinition& tech_def, TechCategory target_track);

} // namespace open_spiel::eclipse


#endif //ECLIPSE_RESEARCH_H
