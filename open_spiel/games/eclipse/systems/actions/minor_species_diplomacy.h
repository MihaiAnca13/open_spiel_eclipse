#ifndef ECLIPSE_MINOR_SPECIES_DIPLOMACY_H
#define ECLIPSE_MINOR_SPECIES_DIPLOMACY_H

#include <cstdint>

#include "open_spiel/games/eclipse/types.h"

struct State;
struct Player;
enum class PopTrack : uint8_t;

namespace open_spiel::eclipse {

    // True if the player can form diplomatic relations with the given minor
    // species: has enough gold, species is in the pool, not already owned.
    bool can_form_minor_species(const ::State& state, uint8_t player_id,
                                uint8_t ms_idx);

    // Initiate formation. Pays gold, adds to owned_minor_species, removes from
    // pool, triggers immediate ability. For PLACE_POP_CUBE sets
    // state.minor_species_pending_track so the player must pick a track next
    // action. Returns false if precondition fails.
    bool begin_minor_species_formation(::State& state, uint8_t player_id,
                                       uint8_t ms_idx);

    // Resolves the PLACE_POP_CUBE track choice. Places one cube on the chosen
    // track and clears minor_species_pending_track.
    bool execute_minor_species_pick_track(::State& state, uint8_t player_id,
                                          PopTrack track);

} // namespace open_spiel::eclipse

#endif // ECLIPSE_MINOR_SPECIES_DIPLOMACY_H
