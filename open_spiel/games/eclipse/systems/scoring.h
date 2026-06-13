//
// Created by Mihai on 13-06-2026.
//

#ifndef OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_SCORING_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_SCORING_H_

#include <cstdint>
#include <array>
#include "../state.h"

namespace open_spiel {
    namespace eclipse {

        struct PlayerScoreBreakdown {
            int16_t reputation_vp = 0;
            int16_t ambassador_vp = 0;
            int16_t sector_vp     = 0;
            int16_t monolith_vp   = 0;
            int16_t discovery_vp  = 0;
            int16_t tech_track_vp = 0;
            int16_t traitor_vp    = 0;
            int16_t species_vp    = 0;
            int16_t total_vp      = 0;
        };

        // Computes absolute total and itemized breakdown values for a specific player
        PlayerScoreBreakdown compute_player_score(const ::State& state, uint8_t player_id);

        // Populates raw utility vector for Open Spiel interface (State::Returns())
        std::array<double, MAX_PLAYERS> evaluate_final_returns(const ::State& state);

    } // namespace eclipse
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_SCORING_H_