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

PlayerScoreBreakdown compute_player_score(const ::State& state, uint8_t player_id) {
    PlayerScoreBreakdown score;

    // Safety check: bound checking loop arrays natively
    if (player_id >= state.players.size()) return score;
    const auto& player = state.players[player_id];

    if (player.eliminated) {
        // Eliminated players retain 0 value across domains
        return score;
    }

    // 1. REPUTATION TILES
    // Using explicit loop over fixed-capacity storage structure
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
    // Rulebook page 25: 1 VP per Ambassador tile held.
    score.ambassador_vp = player.ambassador_tiles_held;

    // 3. SECTORS & STRUCTURES
    // Iterating over the hex axial coordinates matrix space without vector mutations
    uint8_t active_ancients_count = 0;

    // Count live Ancient units from unit_registry for Draco scoring
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

            if (sector.owner_id == player_id) {
                // Base sector control value
                score.sector_vp += sector.points;

                // Monolith tracking
                if (sector.monolith_built) {
                    score.monolith_vp += 3;
                }

                score.sector_vp += sector.player_warp_portal_vp;

                // Planta Species Bonus (+1 VP per sector owned)
                if (player.species_id == Species::PLANTA) {
                    score.species_vp += 1;
                }
            }
        }
    }

    // 4. TECH PROGRESS TRACKS
    uint8_t mil_count  = pop_count_64(player.researched_techs_military);
    uint8_t grid_count = pop_count_64(player.researched_techs_grid);
    uint8_t nano_count = pop_count_64(player.researched_techs_nano);

    score.tech_track_vp += get_tech_track_vp(mil_count);
    score.tech_track_vp += get_tech_track_vp(grid_count);
    score.tech_track_vp += get_tech_track_vp(nano_count);

    // 5. DISCOVERY TILES KEPT AS VP
    // Rulebook page 21: 2 VP per Discovery Tile kept VP-side up.
    // Variable VP tiles (VP_PER_3REP, VP_PER_ARTIFACT) are counted here at a flat
    // 2 VP each until variable scoring is implemented.
    score.discovery_vp = static_cast<int16_t>(player.discovery_vp_tiles_kept) * 2;

    // 6. TRAITOR PENALTY CARD
    // Rulebook page 25: holder gets -2 VP.
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
                // Count non-ambassador Reputation Tiles on the track.
                for (const auto& slot : player.reputation_track) {
                    if (!slot.holds_ambassador && slot.rep_value != ReputationTiles::NONE) {
                        score.minor_species_vp += 1;
                    }
                }
                break;
            case MinorSpeciesAbility::VP_PER_AMBASSADOR:
                // Count Ambassador Tiles including itself.
                score.minor_species_vp +=
                    static_cast<int16_t>(player.ambassador_tiles_held) + 1;
                break;
            case MinorSpeciesAbility::FLAT_VP:
                score.minor_species_vp += static_cast<int16_t>(ms.ability_param);
                break;
            default:
                // All other tiles grant end_vp flat points.
                score.minor_species_vp += static_cast<int16_t>(ms.end_vp);
                break;
        }
    }

    // Total absolute compilation
    score.total_vp = score.reputation_vp + score.ambassador_vp + score.sector_vp +
                     score.monolith_vp + score.discovery_vp + score.tech_track_vp +
                     score.traitor_vp + score.species_vp + score.minor_species_vp;

    return score;
}

std::array<double, MAX_PLAYERS> evaluate_final_returns(const ::State& state) {
    std::array<double, MAX_PLAYERS> returns;
    returns.fill(0.0);

    std::array<int16_t, MAX_PLAYERS> raw_scores;
    raw_scores.fill(-999);

    int16_t highest_score = -999;

    // Compute all base values inside a sequence window
    for (size_t i = 0; i < state.players.size(); ++i) {
        if (state.players[i].eliminated) continue;

        PlayerScoreBreakdown breakdown = compute_player_score(state, static_cast<uint8_t>(i));
        raw_scores[i] = breakdown.total_vp;

        if (breakdown.total_vp > highest_score) {
            highest_score = breakdown.total_vp;
        }
    }

    // Resolve ties natively using raw storage metrics
    for (size_t i = 0; i < state.players.size(); ++i) {
        if (state.players[i].eliminated) {
            returns[i] = 0.0;
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
