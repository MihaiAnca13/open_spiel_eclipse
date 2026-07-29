#include "open_spiel/games/eclipse/systems/actions/minor_species_diplomacy.h"

#include "open_spiel/games/eclipse/state.h"
#include "open_spiel/games/eclipse/minor_species.h"
#include "open_spiel/games/eclipse/systems/actions/bonus.h"

namespace open_spiel::eclipse {

    bool can_form_minor_species(const ::State& state, uint8_t player_id,
                                uint8_t ms_idx) {
        if (player_id >= state.players.size()) return false;
        if (ms_idx >= MINOR_SPECIES_COUNT) return false;
        const ::Player& player = state.players[player_id];
        if (player.eliminated) return false;

        // Check if this species is in the pool.
        bool in_pool = false;
        for (uint8_t pool_ms : state.minor_species_pool) {
            if (pool_ms == ms_idx) { in_pool = true; break; }
        }
        if (!in_pool) return false;

        // Check if already owned.
        for (uint8_t owned : player.owned_minor_species) {
            if (owned == ms_idx) return false;
        }

        const MinorSpeciesData& data = MINOR_SPECIES_TABLE[ms_idx];
        if (player.resources.gold < data.cost) return false;

        // For PLACE_POP_CUBE: must have at least one non-empty pop track.
        if (data.ability == MinorSpeciesAbility::PLACE_POP_CUBE) {
            if (player.resources.gold_prod == 0 &&
                player.resources.science_prod == 0 &&
                player.resources.materials_prod == 0) {
                return false;
            }
        }

        return true;
    }

    bool begin_minor_species_formation(::State& state, uint8_t player_id,
                                       uint8_t ms_idx) {
        if (!can_form_minor_species(state, player_id, ms_idx)) return false;

        ::Player& player = state.players[player_id];
        const MinorSpeciesData& data = MINOR_SPECIES_TABLE[ms_idx];

        // Pay gold cost.
        player.resources.gold -= data.cost;

        // Add to owned list.
        player.owned_minor_species.push_back(ms_idx);

        // Remove from pool (FixedVector has no erase; shift then pop).
        for (size_t i = 0; i < state.minor_species_pool.size(); ++i) {
            if (state.minor_species_pool[i] == ms_idx) {
                for (size_t j = i + 1; j < state.minor_species_pool.size(); ++j) {
                    state.minor_species_pool[j - 1] = state.minor_species_pool[j];
                }
                state.minor_species_pool.pop_back();
                break;
            }
        }

        // Trigger immediate ability.
        if (data.ability == MinorSpeciesAbility::PLACE_POP_CUBE) {
            state.minor_species_pending_track = player_id;
        }

        return true;
    }

    bool execute_minor_species_pick_track(::State& state, uint8_t player_id,
                                          PopTrack track) {
        if (player_id >= state.players.size()) return false;
        if (state.minor_species_pending_track != player_id) return false;

        ::Player& player = state.players[player_id];

        // Validate track has cubes.
        switch (track) {
            case PopTrack::MONEY:
                if (player.resources.gold_prod == 0) return false;
                --player.resources.gold_prod;
                break;
            case PopTrack::SCIENCE:
                if (player.resources.science_prod == 0) return false;
                --player.resources.science_prod;
                break;
            case PopTrack::MATERIALS:
                if (player.resources.materials_prod == 0) return false;
                --player.resources.materials_prod;
                break;
        }

        state.minor_species_pending_track = 255;
        return true;
    }

} // namespace open_spiel::eclipse
