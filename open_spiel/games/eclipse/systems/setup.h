//
// Created by Mihai on 28/05/2026.
//

#ifndef ECLIPSE_SETUP_H
#define ECLIPSE_SETUP_H
#include "../state.h"
#include <cstdint>
#include <vector>
#include <random>

struct PlayerConfig {
    Species species;
    bool is_ai;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PlayerConfig, species, is_ai);

// Stage 1: Shared board setup before players choose species
State initialize_pre_choice_state(std::mt19937_64& rng, uint8_t num_players, NPCDifficulty difficulty = NPCDifficulty::EASY);

// Stage 2: Finalize setup once player species selection is committed
void finalize_game_setup(std::mt19937_64& rng, State& state, const std::vector<PlayerConfig>& player_choices);

#endif //ECLIPSE_SETUP_H
