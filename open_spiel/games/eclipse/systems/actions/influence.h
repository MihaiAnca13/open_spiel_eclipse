//
// Influence actions: Modifying control of sectors and reclaiming discs.
//

#ifndef ECLIPSE_INFLUENCE_H
#define ECLIPSE_INFLUENCE_H

#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>
#include "open_spiel/games/eclipse/sectors.h"

// Forward declaration to avoid circular dependency
struct State;

struct PendingReturn
{
    PlanetType type;
    bool is_orbital;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PendingReturn, type, is_orbital);

// Sub-state machine driving multi-activation influence actions.
struct InfluenceState
{
    enum class Phase : uint8_t { inactive = 0, choose_influence = 1, choose_return_track = 2 };

    Phase phase = Phase::inactive;
    uint8_t player_id = 255;
    uint8_t activations_remaining = 0;
    std::vector<PendingReturn> pending_returns;
};

NLOHMANN_JSON_SERIALIZE_ENUM(InfluenceState::Phase, {
    {InfluenceState::Phase::inactive, "inactive"},
    {InfluenceState::Phase::choose_influence, "choose_influence"},
    {InfluenceState::Phase::choose_return_track, "choose_return_track"},
});

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InfluenceState, phase, player_id, activations_remaining, pending_returns);

namespace open_spiel::eclipse
{
    // Validates whether a player can move an influence disc to a target cell
    // or reclaim a disc back to their track.
    bool can_influence_to_sector(const ::State& state, uint8_t player_id, uint8_t galaxy_cell_idx);
    bool can_reclaim_from_sector(const ::State& state, uint8_t player_id, uint8_t galaxy_cell_idx);

    // Executes placement of an influence disc from the player's track onto a sector.
    bool execute_influence_to_sector(::State& state, uint8_t player_id, uint8_t galaxy_cell_idx);

    // Executes reclamation of an influence disc from a sector back to the track.
    // Automatically pulls population cubes off the sector back into their tracks.
    bool execute_reclaim_from_sector(::State& state, uint8_t player_id, uint8_t galaxy_cell_idx);

    // Executes the player's choice of track to return a pending cube to.
    // track: 0=Money, 1=Science, 2=Materials
    bool execute_choose_return_track(::State& state, uint8_t player_id, uint8_t track);

    // Returns the legal return tracks (0=Money, 1=Science, 2=Materials) for the current pending return cube.
    std::vector<uint8_t> get_legal_return_tracks_for_current_pending(const ::State& state);

    // Initializes a multi-activation influence cycle and automatically restores
    // up to two colony ships faceup.
    bool begin_influence(::State& state, uint8_t player_id);
} // namespace open_spiel::eclipse

#endif // ECLIPSE_INFLUENCE_H
