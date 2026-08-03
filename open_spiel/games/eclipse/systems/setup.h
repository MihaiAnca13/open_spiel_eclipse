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
    bool warped_universe = false;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SetupConfig, players, rng_seed, npc_difficulty, staged_players, warped_universe);

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

// Per-episode setup randomization (opt-in via game params). Draws a fresh
// race assignment (unique alien draft, Terran as filler), NPC difficulty, and
// warped-universe flag from ``rng`` into ``config``:
//   - Races: the 6 alien species are drawn without replacement (each used at
//     most once); each seat is an alien with probability ``alien_prob``,
//     otherwise Terran. Terran fills in whatever slots remain when the alien
//     pool is exhausted, so it may appear any number of times.
//   - Difficulty: uniform over EASY/MEDIUM/HARD when ``randomize_difficulty``.
//   - Warped universe: drawn with probability ``warped_prob`` when
//     ``randomize_warped``.
void RandomizeSetupForEpisode(std::mt19937_64& rng, SetupConfig& config,
                              double alien_prob, bool randomize_difficulty,
                              bool randomize_warped, double warped_prob);
SetupSnapshot CreatePreChoiceSnapshot(const SetupConfig& config);
SetupSnapshot FinalizeSetupSnapshot(
    const SetupSnapshot& snapshot,
    const std::vector<PlayerConfig>& player_choices);


#endif //ECLIPSE_SETUP_H
