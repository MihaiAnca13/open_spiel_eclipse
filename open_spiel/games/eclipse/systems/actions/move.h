//
// Move actions: relocating ships along the wormhole / warp portal network.
//

#ifndef ECLIPSE_MOVE_H
#define ECLIPSE_MOVE_H

#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>

// Forward declarations to avoid circular dependency
struct State;
struct Player;
enum class ShipType;

// Sub-state machine driving multi-activation move actions.
// Each activation moves one ship up to its Movement Value, one hex per step,
// so the action space stays compact (unit x direction) for MCTS/RL.
struct MoveState
{
    enum class Phase : uint8_t
    {
        inactive = 0,
        choose_move = 1,
        choose_warp_destination = 2
    };

    Phase phase = Phase::inactive;
    uint8_t player_id = 255;
    uint8_t activations_remaining = 0;
    // Ship currently mid-activation (255 = none). Re-selecting another ship
    // ends the current activation early and starts a new one.
    uint8_t active_unit_idx = 255;
    // Hex steps the active ship may still take this activation.
    uint8_t steps_remaining = 0;
    // Ship awaiting a warp portal destination choice (255 = none).
    uint8_t warp_unit_idx = 255;
};

NLOHMANN_JSON_SERIALIZE_ENUM(MoveState::Phase, {
    {MoveState::Phase::inactive, "inactive"},
    {MoveState::Phase::choose_move, "choose_move"},
    {MoveState::Phase::choose_warp_destination, "choose_warp_destination"},
});

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MoveState, phase, player_id, activations_remaining, active_unit_idx, steps_remaining, warp_unit_idx);

namespace open_spiel::eclipse
{
    // One legal single-hex movement step (or warp network entry) for a unit.
    struct MoveStepOption
    {
        uint8_t unit_idx;   // Index into State::unit_registry
        uint8_t direction;  // 0-5 = HEX_DIRECTIONS, MOVE_WARP_DIRECTION = warp entry
    };

    // Direction code reserved for entering the warp portal network.
    constexpr uint8_t MOVE_WARP_DIRECTION = 6;
    constexpr int MOVE_CODES_PER_UNIT = 7;  // 6 hex directions + warp entry

    // Returns the Movement Value (total drive value) for a player's ship type.
    uint8_t ship_movement_value(const Player& player, ShipType ship_type);

    // Pinning rule: each opponent ship in a sector pins one of yours (two with
    // Cloaking Device); the GCDS pins all ships. True if one more ship may leave.
    bool can_leave_sector(const ::State& state, uint8_t player_id, uint16_t sector_id);

    // Validates a single-hex step of the in-flight Move action.
    bool can_move_step(const ::State& state, uint8_t player_id, uint8_t unit_idx, uint8_t direction);

    // Validates entering the warp portal network with a unit.
    bool can_begin_warp_move(const ::State& state, uint8_t player_id, uint8_t unit_idx);

    // All legal steps for the current Move sub-phase, computed in one pass.
    std::vector<MoveStepOption> legal_move_steps(const ::State& state, uint8_t player_id);

    // Galaxy cell indices of warp portal sectors reachable by the pending warp unit.
    std::vector<uint8_t> legal_warp_destination_cells(const ::State& state, uint8_t player_id);

    // Executes a single-hex movement step, handling activation bookkeeping.
    bool execute_move_step(::State& state, uint8_t player_id, uint8_t unit_idx, uint8_t direction);

    // Starts a warp transit: switches to the destination choice sub-phase.
    bool begin_warp_move(::State& state, uint8_t player_id, uint8_t unit_idx);

    // Completes a warp transit to the chosen warp portal sector.
    bool execute_warp_move(::State& state, uint8_t player_id, uint8_t galaxy_cell_idx);

    // Initializes a multi-activation move cycle from species traits and techs.
    // Stays inactive (returns false) when the player has no legal move at all.
    bool begin_move(::State& state, uint8_t player_id);
} // namespace open_spiel::eclipse

#endif // ECLIPSE_MOVE_H
