//
// Created by Mihai on 28/05/2026.
//

#ifndef ECLIPSE_RESEARCH_H
#define ECLIPSE_RESEARCH_H

#include "../../state.h"
#include "../../tech.h"

namespace open_spiel {
namespace eclipse {

// Returns the final cost to research the given technology for the player.
uint8_t calculate_research_cost(const Player& player, TechBit tech_bit);

// Researches a technology for a player.
// Subtracts science resources, adds the technology to the player's bitmask, and decrements/updates the tech tray.
// Returns true on success, false if the tech is not researchable (already researched, too expensive, or not in the tech tray).
bool research_tech(State& state, uint8_t player_id, TechBit tech_bit);

} // namespace eclipse
} // namespace open_spiel

#endif //ECLIPSE_RESEARCH_H
