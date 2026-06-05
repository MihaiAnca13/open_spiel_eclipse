//
// Build actions: spending materials to construct ships and structures.
//

#ifndef ECLIPSE_BUILD_H
#define ECLIPSE_BUILD_H

#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>
#include "open_spiel/games/eclipse/state.h"

// Sub-state machine driving multi-activation build actions.
struct BuildState
{
    enum class Phase : uint8_t { inactive = 0, choose_build = 1 };

    Phase phase = Phase::inactive;
    uint8_t player_id = 255;
    uint8_t activations_remaining = 0;
};

NLOHMANN_JSON_SERIALIZE_ENUM(BuildState::Phase, {
    {BuildState::Phase::inactive, "inactive"},
    {BuildState::Phase::choose_build, "choose_build"},
});

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BuildState, phase, player_id, activations_remaining);

namespace open_spiel::eclipse
{
    // Available constructs to build.
    enum class BuildType : uint8_t
    {
        INTERCEPTOR = 0,
        CRUISER = 1,
        DREADNOUGHT = 2,
        STARBASE = 3,
        ORBITAL = 4,
        MONOLITH = 5
    };

    constexpr int BUILD_TYPE_COUNT = 6;

    // Returns the material cost for the specific construct adjusted for species rules.
    uint8_t calculate_build_cost(const Player& player, BuildType type);

    // Validates whether a player can execute a specific build action at a target galaxy cell hex.
    bool can_build(const State& state, uint8_t player_id, BuildType type, uint8_t galaxy_cell_idx);

    // Executes a single build validation and processing phase.
    // Deducts materials, updates structure flags or spawns ship units into the registry,
    // and handles multi-activation iteration decrement.
    bool execute_build(State& state, uint8_t player_id, BuildType type, uint8_t galaxy_cell_idx);

    // Initializes a multi-activation build cycle by pulling the species' innate icons.
    bool begin_build(State& state, uint8_t player_id);
} // namespace open_spiel::eclipse

#endif // ECLIPSE_BUILD_H
