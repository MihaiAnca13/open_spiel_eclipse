//
// Created by Mihai on 28/05/2026.
//

#ifndef ECLIPSE_SETUP_H
#define ECLIPSE_SETUP_H

#include <random>
#include <string>
#include "../state.h"

struct PlayerConfig {
    Species species;
    bool is_ai;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PlayerConfig, species, is_ai);

struct StagedPlayerConfig {
    Species species = Species::TERRAN_FACTIONS;
    bool is_ai = false;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(StagedPlayerConfig, species, is_ai);

struct SetupConfig {
    uint8_t players = 4;
    uint64_t rng_seed = 0;
    NPCDifficulty npc_difficulty = NPCDifficulty::EASY;
    std::vector<StagedPlayerConfig> staged_players;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SetupConfig, players, rng_seed, npc_difficulty, staged_players);

struct SetupSnapshot {
    SetupConfig config;
    State state;
    bool finalized = false;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SetupSnapshot, config, state, finalized);

SetupConfig NormalizeSetupConfig(SetupConfig config);
State InitializeDeterministicSetupState(const SetupConfig& config);
void ResolveInitialSetupRandomness(std::mt19937_64& rng,
                                   const SetupConfig& config,
                                   State& state);
void FinalizeGameSetup(State& state,
                       const std::vector<PlayerConfig>& player_choices);
SetupSnapshot CreatePreChoiceSnapshot(const SetupConfig& config);
SetupSnapshot FinalizeSetupSnapshot(
    const SetupSnapshot& snapshot,
    const std::vector<PlayerConfig>& player_choices);


#endif //ECLIPSE_SETUP_H
