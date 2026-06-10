//
// Upgrade actions: Modifying ship blueprints with upgraded parts.
//

#ifndef ECLIPSE_UPGRADE_H
#define ECLIPSE_UPGRADE_H

#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>
#include "open_spiel/games/eclipse/tech.h"

// Forward declarations to prevent circular dependencies
struct State;
struct Player;
enum class ShipType;

// Tracks the structural sub-state loop of a multi-activation Upgrade action phase.
struct UpgradeState
{
    enum class Phase : uint8_t { inactive = 0, choose_upgrade = 1 };

    Phase phase = Phase::inactive;
    uint8_t player_id = 255;
    uint8_t activations_remaining = 0;
};

NLOHMANN_JSON_SERIALIZE_ENUM(UpgradeState::Phase, {
    {UpgradeState::Phase::inactive, "inactive"},
    {UpgradeState::Phase::choose_upgrade, "choose_upgrade"},
});

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UpgradeState, phase, player_id, activations_remaining);

namespace open_spiel::eclipse
{

    // Validates whether a player can assign a specific part to a target slot index on a ship blueprint type.
    bool can_upgrade(const ::State& state, uint8_t player_id, ShipType ship_type, uint8_t slot_idx, ShipPartId part_id);

    // Executes a single component modification activation pass over a ship blueprint.
    bool execute_upgrade(::State& state, uint8_t player_id, ShipType ship_type, uint8_t slot_idx, ShipPartId part_id);

    // Initializes a multi-activation upgrade cycle using species traits and permanent modifiers.
    bool begin_upgrade(::State& state, uint8_t player_id);

} // namespace open_spiel::eclipse

#endif // ECLIPSE_UPGRADE_H