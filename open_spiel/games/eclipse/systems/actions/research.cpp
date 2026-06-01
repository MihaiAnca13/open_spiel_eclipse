//
// Created by Mihai on 28/05/2026.
//

#include "research.h"
#include <algorithm>

namespace open_spiel {
namespace eclipse {

uint8_t calculate_research_cost(const Player& player, TechBit tech_bit) {
    const TechDefinition* tech_def = nullptr;
    for (const auto& def : TECH_TABLE) {
        if (def.bit == tech_bit) {
            tech_def = &def;
            break;
        }
    }
    if (!tech_def) return 0;

    // Same-category discount is -1 per tech already owned in that category
    uint8_t discount = 0;
    for (const auto& def : TECH_TABLE) {
        if (def.category == tech_def->category && player.has_tech(def.bit)) {
            discount++;
        }
    }

    int16_t cost = static_cast<int16_t>(tech_def->base_cost) - discount;
    return static_cast<uint8_t>(std::max(static_cast<int16_t>(tech_def->min_cost), cost));
}

bool research_tech(State& state, uint8_t player_id, TechBit tech_bit) {
    if (player_id >= state.players.size()) return false;
    Player& player = state.players[player_id];

    // Already owned
    if (player.has_tech(tech_bit)) return false;

    // Check availability in tech tray
    if (state.get_tech_tray_count(tech_bit) == 0) return false;

    uint8_t cost = calculate_research_cost(player, tech_bit);
    if (player.resources.science < cost) return false;

    // Subtract cost and add tech to masks
    player.resources.science -= cost;
    player.researched_techs |= static_cast<uint64_t>(tech_bit);

    // Remove from market tray
    state.remove_from_tech_tray(tech_bit);

    return true;
}

} // namespace eclipse
} // namespace open_spiel
