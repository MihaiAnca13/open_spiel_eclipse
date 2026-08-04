//
// Created by Mihai on 2026.
//

#include "scoring.h"
#include "../state.h"
#include "../galaxy.h"
#include "../types.h"
#include "../species.h"
#include "../minor_species.h"

using open_spiel::eclipse::ReputationSlot;

namespace open_spiel {
namespace eclipse {

namespace {

// Maps the sequential amount of uncovered tech items on a given category track to its VP value.
// Rulebook: 4 tiles = 1VP, 5 = 2VP, 6 = 3VP, 7 = 5VP (and 8 = 5VP max base game).
inline int16_t get_tech_track_vp(uint8_t count) {
    if (count >= 7) return 5;
    if (count == 6) return 3;
    if (count == 5) return 2;
    if (count == 4) return 1;
    return 0;
}

// Counts total precalculated population blocks mapped in active masks
inline uint8_t pop_count_64(uint64_t mask) {
    return static_cast<uint8_t>(__builtin_popcountll(mask));
}

} // namespace

namespace {

// Adds all per-player (non-geometry) score components that don't require the
// galaxy sweep or unit registry, i.e. the parts identical for every seat.
void AddPlayerStaticComponents(PlayerScoreBreakdown& score, const ::Player& player,
                               int16_t active_ancients_count) {
    // 1. REPUTATION TILES
    for (size_t i = 0; i < player.reputation_track.size(); ++i) {
        const ReputationSlot& slot = player.reputation_track[i];
        if (slot.holds_ambassador) continue;  // Ambassador tile, not a Rep tile.
        switch (slot.rep_value) {
            case ReputationTiles::ONE:   score.reputation_vp += 1; break;
            case ReputationTiles::TWO:   score.reputation_vp += 2; break;
            case ReputationTiles::THREE: score.reputation_vp += 3; break;
            case ReputationTiles::FOUR:  score.reputation_vp += 4; break;
            case ReputationTiles::NONE:  break;  // empty slot, no tile
        }
    }

    // 2. AMBASSADOR TILES
    score.ambassador_vp = player.ambassador_tiles_held;

    // 4. TECH PROGRESS TRACKS
    uint8_t mil_count  = pop_count_64(player.researched_techs_military);
    uint8_t grid_count = pop_count_64(player.researched_techs_grid);
    uint8_t nano_count = pop_count_64(player.researched_techs_nano);

    score.tech_track_vp += get_tech_track_vp(mil_count);
    score.tech_track_vp += get_tech_track_vp(grid_count);
    score.tech_track_vp += get_tech_track_vp(nano_count);

    // 5. DISCOVERY TILES KEPT AS VP
    score.discovery_vp = static_cast<int16_t>(player.discovery_vp_tiles_kept) * 2;

    // 6. TRAITOR PENALTY CARD
    score.traitor_vp = player.traitor_held ? -2 : 0;

    // 7. DESCENDANTS OF DRACO SPECIFIC BONUS
    if (player.species_id == Species::DESCENDANTS_OF_DRACO) {
        score.species_vp += active_ancients_count;
    }

    // 8. MINOR SPECIES AMBASSADOR TILES
    for (uint8_t ms_idx : player.owned_minor_species) {
        const MinorSpeciesData& ms = MINOR_SPECIES_TABLE[ms_idx];
        switch (ms.ability) {
            case MinorSpeciesAbility::VP_PER_REPUTATION:
                for (const auto& slot : player.reputation_track) {
                    if (!slot.holds_ambassador && slot.rep_value != ReputationTiles::NONE) {
                        score.minor_species_vp += 1;
                    }
                }
                break;
            case MinorSpeciesAbility::VP_PER_AMBASSADOR:
                score.minor_species_vp +=
                    static_cast<int16_t>(player.ambassador_tiles_held) + 1;
                break;
            default:
                score.minor_species_vp += static_cast<int16_t>(ms.end_vp);
                break;
        }
    }

    // Total absolute compilation
    score.total_vp = score.reputation_vp + score.ambassador_vp + score.sector_vp +
                     score.monolith_vp + score.discovery_vp + score.tech_track_vp +
                     score.traitor_vp + score.species_vp + score.minor_species_vp;
}

} // namespace

std::array<PlayerScoreBreakdown, MAX_PLAYERS> compute_all_player_scores(const ::State& state) {
    std::array<PlayerScoreBreakdown, MAX_PLAYERS> scores{};
    const size_t num_players = state.players.size();
    if (num_players == 0 || num_players > MAX_PLAYERS) return scores;

    // 3. SECTORS & STRUCTURES -- one galaxy sweep shared by every seat.
    // Count live Ancient units from unit_registry for Draco scoring (one pass).
    int16_t active_ancients_count = 0;
    for (const Unit& unit : state.unit_registry) {
        if (unit.type == ShipType::ANCIENT) {
            active_ancients_count++;
        }
    }

    for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
        for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
            if (!in_galaxy_bounds(q, r)) continue;

            const auto& sector = state.galaxy.at(q, r);
            if (sector.sector_id == 0) continue; // Unmapped void

            const SectorDefinition* def = get_sector_definition(sector.sector_id);
            if (!def) continue;

            const uint8_t owner = sector.owner_id;
            if (owner == 255 || owner >= num_players) continue;
            if (state.players[owner].eliminated) continue;

            PlayerScoreBreakdown& sc = scores[owner];
            sc.sector_vp += sector.points;
            if (sector.monolith_built) {
                sc.monolith_vp += 3;
            }
            sc.sector_vp += sector.player_warp_portal_vp;
            // Planta Species Bonus (+1 VP per sector owned)
            if (state.players[owner].species_id == Species::PLANTA) {
                sc.species_vp += 1;
            }
        }
    }

    // Per-player components that don't touch the galaxy / unit registry.
    for (size_t i = 0; i < num_players; ++i) {
        if (state.players[i].eliminated) continue;
        AddPlayerStaticComponents(scores[i], state.players[i], active_ancients_count);
    }

    return scores;
}

PlayerScoreBreakdown compute_player_score(const ::State& state, uint8_t player_id) {
    return compute_all_player_scores(state)[player_id];
}

std::array<double, MAX_PLAYERS> evaluate_final_returns(const ::State& state) {
    std::array<double, MAX_PLAYERS> returns;
    returns.fill(0.0);

    std::array<int16_t, MAX_PLAYERS> raw_scores;
    raw_scores.fill(-999);

    const std::array<PlayerScoreBreakdown, MAX_PLAYERS> all =
        compute_all_player_scores(state);

    int16_t highest_score = -999;

    // Compute all base values inside a sequence window
    for (size_t i = 0; i < state.players.size(); ++i) {
        if (state.players[i].eliminated) continue;

        raw_scores[i] = all[i].total_vp;

        if (raw_scores[i] > highest_score) {
            highest_score = raw_scores[i];
        }
    }

    // Resolve ties natively using raw storage metrics
    for (size_t i = 0; i < state.players.size(); ++i) {
        if (state.players[i].eliminated) {
            // Rulebook (PLAYER ELIMINATION): eliminated players still count
            // their score. Their components are off the board by now, so use the
            // total snapshotted at the moment of elimination. Clamped at 0 to
            // respect MinUtility() (the traitor -2 penalty can make a snapshot
            // negative; that pre-existing MinUtility violation is tracked
            // separately for the live-player path).
            const int16_t snapshot = state.players[i].vp_at_elimination;
            returns[i] = snapshot > 0 ? static_cast<double>(snapshot) : 0.0;
            continue;
        }

        if (raw_scores[i] == highest_score) {
            // Check if multiple players share the max value to trigger a deeper comparison
            bool tie_exists = false;
            for (size_t j = 0; j < state.players.size(); ++j) {
                if (i != j && raw_scores[j] == highest_score) {
                    tie_exists = true;
                    break;
                }
            }

            if (tie_exists) {
                // Tiebreaker: Sum total remaining material stock resources
                const auto& res = state.players[i].resources;
                uint32_t total_resources = static_cast<uint32_t>(res.gold) +
                                           static_cast<uint32_t>(res.science) +
                                           static_cast<uint32_t>(res.materials);

                // Add minor fractional scaling to give MCTS clear paths without changing base value
                returns[i] = static_cast<double>(raw_scores[i]) + (static_cast<double>(total_resources) * 0.001);
            } else {
                returns[i] = static_cast<double>(raw_scores[i]);
            }
        } else {
            returns[i] = static_cast<double>(raw_scores[i]);
        }
    }

    return returns;
}

} // namespace eclipse
} // namespace open_spiel
