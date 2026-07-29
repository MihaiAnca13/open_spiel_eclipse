//
// Created by Mihai on 13-06-2026.
//

#ifndef OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_SCORING_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_SCORING_H_

#include <cstdint>
#include <array>
#include <nlohmann/json.hpp>

struct State;

#ifndef MAX_PLAYERS
#define MAX_PLAYERS 6
#endif

namespace open_spiel {
    namespace eclipse {

        struct PlayerScoreBreakdown {
            int16_t reputation_vp    = 0;
            int16_t ambassador_vp    = 0;
            int16_t sector_vp        = 0;
            int16_t monolith_vp      = 0;
            int16_t discovery_vp     = 0;
            int16_t tech_track_vp    = 0;
            int16_t traitor_vp       = 0;
            int16_t species_vp       = 0;
            int16_t minor_species_vp = 0;
            int16_t total_vp         = 0;
        };

        inline void to_json(nlohmann::json& j, const PlayerScoreBreakdown& sb) {
            j = nlohmann::json{
                {"reputation_vp", sb.reputation_vp},
                {"ambassador_vp", sb.ambassador_vp},
                {"sector_vp", sb.sector_vp},
                {"monolith_vp", sb.monolith_vp},
                {"discovery_vp", sb.discovery_vp},
                {"tech_track_vp", sb.tech_track_vp},
                {"traitor_vp", sb.traitor_vp},
                {"species_vp", sb.species_vp},
                {"minor_species_vp", sb.minor_species_vp},
                {"total_vp", sb.total_vp}
            };
        }

        // Computes absolute total and itemized breakdown values for a specific player
        PlayerScoreBreakdown compute_player_score(const ::State& state, uint8_t player_id);

        // Populates raw utility vector for Open Spiel interface (State::Returns())
        std::array<double, MAX_PLAYERS> evaluate_final_returns(const ::State& state);

    } // namespace eclipse
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_SCORING_H_