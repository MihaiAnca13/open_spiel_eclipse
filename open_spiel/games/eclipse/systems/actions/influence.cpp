//
// Created by Mihai on 05/06/2026.
//

#include "open_spiel/games/eclipse/systems/actions/influence.h"

#include <algorithm>
#include <bitset>

#include "open_spiel/games/eclipse/state.h"
#include "open_spiel/games/eclipse/galaxy.h"
#include "open_spiel/games/eclipse/sectors.h"
#include "open_spiel/games/eclipse/species.h"
#include "open_spiel/games/eclipse/tech.h"
#include "open_spiel/games/eclipse/systems/actions/bonus.h" // For refresh_colony_ships

namespace open_spiel::eclipse
{
    namespace
    {
        // Validates wormhole connections to a target hex from any block where the player anchors.
        bool has_influence_wormhole_access(const ::State& state, uint8_t player_id, int q, int r)
        {
            const Sector& dest = state.galaxy.at(q, r);
            const SectorDefinition* dest_def = get_sector_definition(dest.sector_id);
            if (!dest_def) return false;

            if ((dest.has_player_warp_portal || dest_def->has_warp_portal) && player_has_warp_portal_anchor(state, player_id))
            {
                return true;
            }

            const bool wormhole_generator =
                state.players[player_id].has_tech(TechBit::WORMHOLE_GENERATOR);
            uint8_t dest_mask = 0;
            if (!wormhole_generator)
            {
                dest_mask = rotate_edge_mask(dest_def->wormholes_mask, dest.rotation);
            }

            for (uint8_t d = 0; d < 6; ++d)
            {
                int nq = q + HEX_DIRECTIONS[d].first;
                int nr = r + HEX_DIRECTIONS[d].second;

                // Ensure neighbor is inside bounds
                if (!in_galaxy_bounds(nq, nr)) continue;

                const Sector& nb = state.galaxy.at(nq, nr);
                if (nb.sector_id == 0) continue;

                // Neighbor must be controlled or hold an unpinned ship to source the connection [cite: 854]
                if (!is_sector_anchor(state, player_id, nb)) continue;

                if (wormhole_generator)
                {
                    return true;
                }

                const SectorDefinition* ndef = get_sector_definition(nb.sector_id);
                if (!ndef) continue;

                // Check if neighbor facing edge contains a valid wormhole gateway
                uint8_t nb_mask = rotate_edge_mask(ndef->wormholes_mask, nb.rotation);
                bool my_edge = has_edge(dest_mask, d);
                bool their_edge = has_edge(nb_mask, (d + 3) % 6);
                if (my_edge && their_edge)
                {
                    return true;
                }
            }
            return false;
        }

        // Counts active enemy ships in a given sector.
        bool enemy_ships_present(const std::bitset<512>& enemy_ships, uint16_t sector_id)
        {
            return enemy_ships.test(sector_id);
        }

        // Returns true if ONLY the active player has ships in the unowned sector.
        bool only_own_ships_present(const std::bitset<512>& own_ships, const std::bitset<512>& enemy_ships, uint16_t sector_id)
        {
            return own_ships.test(sector_id) && !enemy_ships.test(sector_id);
        }

        // Advances or terminates the structural action machine loop tracking context.
        void end_influence_activation(State& state)
        {
            if (state.influence_state.activations_remaining > 0)
            {
                --state.influence_state.activations_remaining;
            }

            if (state.influence_state.activations_remaining == 0)
            {
                state.influence_state.phase = InfluenceState::Phase::inactive;
                state.influence_state.player_id = 255;
            }
            else
            {
                state.influence_state.phase = InfluenceState::Phase::choose_influence;
            }
        }

        PopTrack get_matching_track(PlanetType type)
        {
            if (type == PlanetType::SCIENCE || type == PlanetType::ADV_SCIENCE)
            {
                return PopTrack::SCIENCE;
            }
            if (type == PlanetType::MATERIALS || type == PlanetType::ADV_MATERIALS)
            {
                return PopTrack::MATERIALS;
            }
            return PopTrack::MONEY;
        }

        bool return_requires_choice(const Player& player, PlanetType type, bool is_orbital)
        {
            if (is_orbital) return true;
            if (type == PlanetType::ANY || type == PlanetType::ADV_ANY) return true;

            PopTrack matching = get_matching_track(type);
            uint8_t current_val = 0;
            if (matching == PopTrack::MONEY) current_val = player.resources.gold_prod;
            else if (matching == PopTrack::SCIENCE) current_val = player.resources.science_prod;
            else current_val = player.resources.materials_prod;

            return (current_val >= 12);
        }

        std::vector<PopTrack> get_legal_return_tracks(const Player& player, PlanetType type, bool is_orbital)
        {
            std::vector<PopTrack> tracks;

            if (is_orbital)
            {
                if (player.resources.gold_prod < 12) {
                    tracks.push_back(PopTrack::MONEY);
                }
                if (player.resources.science_prod < 12) {
                    tracks.push_back(PopTrack::SCIENCE);
                }
                if (tracks.empty() && player.resources.materials_prod < 12) {
                    tracks.push_back(PopTrack::MATERIALS);
                }
                return tracks;
            }

            if (type == PlanetType::ANY || type == PlanetType::ADV_ANY)
            {
                if (player.resources.gold_prod < 12) tracks.push_back(PopTrack::MONEY);
                if (player.resources.science_prod < 12) tracks.push_back(PopTrack::SCIENCE);
                if (player.resources.materials_prod < 12) tracks.push_back(PopTrack::MATERIALS);
                return tracks;
            }

            PopTrack matching = get_matching_track(type);
            uint8_t current_val = 0;
            if (matching == PopTrack::MONEY) current_val = player.resources.gold_prod;
            else if (matching == PopTrack::SCIENCE) current_val = player.resources.science_prod;
            else current_val = player.resources.materials_prod;

            if (current_val < 12)
            {
                tracks.push_back(matching);
            }
            else
            {
                if (matching != PopTrack::MONEY && player.resources.gold_prod < 12) {
                    tracks.push_back(PopTrack::MONEY);
                }
                if (matching != PopTrack::SCIENCE && player.resources.science_prod < 12) {
                    tracks.push_back(PopTrack::SCIENCE);
                }
                if (matching != PopTrack::MATERIALS && player.resources.materials_prod < 12) {
                    tracks.push_back(PopTrack::MATERIALS);
                }
            }
            return tracks;
        }

        void process_pending_returns(::State& state)
        {
            InfluenceState& is = state.influence_state;
            Player& player = state.players[is.player_id];

            while (!is.pending_returns.empty())
            {
                const auto& pending = is.pending_returns.front();

                if (return_requires_choice(player, pending.type, pending.is_orbital))
                {
                    is.phase = InfluenceState::Phase::choose_return_track;
                    return;
                }

                PopTrack matching = get_matching_track(pending.type);
                if (matching == PopTrack::SCIENCE) player.resources.science_prod++;
                else if (matching == PopTrack::MATERIALS) player.resources.materials_prod++;
                else player.resources.gold_prod++;

                is.pending_returns.erase(is.pending_returns.begin());
            }

            end_influence_activation(state);
        }

        void return_cube_to_track(Player& player, PlanetType type)
        {
            if (return_requires_choice(player, type, false))
            {
                if (player.resources.gold_prod < 12) player.resources.gold_prod++;
                else if (player.resources.science_prod < 12) player.resources.science_prod++;
                else if (player.resources.materials_prod < 12) player.resources.materials_prod++;
            }
            else
            {
                PopTrack matching = get_matching_track(type);
                if (matching == PopTrack::SCIENCE) player.resources.science_prod++;
                else if (matching == PopTrack::MATERIALS) player.resources.materials_prod++;
                else player.resources.gold_prod++;
            }
        }
    } // namespace

    bool can_influence_to_sector(const ::State& state, uint8_t player_id, uint8_t galaxy_cell_idx)
    {
        if (player_id >= state.players.size()) return false;
        const Player& player = state.players[player_id];
        // Must have discs available on your internal sheet tracker track [cite: 460]
        if (player.available_influence_discs() == 0) return false;

        HexCoord coord = index_to_hex(galaxy_cell_idx);
        const Sector& sector = state.galaxy.at(coord.q, coord.r);

        // Sector must be an explored tile and completely unowned/vacant [cite: 848]
        if (sector.sector_id == 0 || sector.owner_id != 255) return false;

        // Build bitsets for O(1) ship presence checks (matches explore.cpp pattern)
        constexpr int kMaxSectorId = 512;
        std::bitset<kMaxSectorId> own_ships;
        std::bitset<kMaxSectorId> enemy_ships;
        for (const Unit& unit : state.unit_registry) {
            if (unit.sector_id < kMaxSectorId) {
                if (unit.player_id == player_id) {
                    own_ships.set(unit.sector_id);
                } else if (unit.player_id != NPC_PLAYER_ID) {
                    enemy_ships.set(unit.sector_id);
                }
            }
        }

        // Condition A: Vacant, no opponent ships, has wormhole connection to control/ship anchor [cite: 854]
        if (!enemy_ships_present(enemy_ships, sector.sector_id) &&
            has_influence_wormhole_access(state, player_id, coord.q, coord.r))
        {
            return true;
        }

        // Condition B: Uncontrolled sector where ONLY you have a ship present [cite: 855]
        if (only_own_ships_present(own_ships, enemy_ships, sector.sector_id))
        {
            return true;
        }

        return false;
    }

    bool can_reclaim_from_sector(const ::State& state, uint8_t player_id, uint8_t galaxy_cell_idx)
    {
        if (player_id >= state.players.size()) return false;

        HexCoord coord = index_to_hex(galaxy_cell_idx);
        const Sector& sector = state.galaxy.at(coord.q, coord.r);

        // Sector must be explored and currently possessed by the active player [cite: 848]
        if (sector.sector_id == 0 || sector.owner_id != player_id) return false;

        return true;
    }

    bool execute_influence_to_sector(::State& state, uint8_t player_id, uint8_t galaxy_cell_idx)
    {
        if (!can_influence_to_sector(state, player_id, galaxy_cell_idx)) return false;

        Player& player = state.players[player_id];
        HexCoord coord = index_to_hex(galaxy_cell_idx);
        Sector& sector = state.galaxy.at(coord.q, coord.r);

        // Drop leftmost disc onto the planet node map [cite: 327, 559]
        sector.owner_id = player_id;
        player.disks_on_sectors++;

        end_influence_activation(state);
        return true;
    }

    bool execute_reclaim_from_sector(::State& state, uint8_t player_id, uint8_t galaxy_cell_idx)
    {
        if (!can_reclaim_from_sector(state, player_id, galaxy_cell_idx)) return false;

        Player& player = state.players[player_id];
        HexCoord coord = index_to_hex(galaxy_cell_idx);
        Sector& sector = state.galaxy.at(coord.q, coord.r);

        // Lift ownership disc and return to your track pool [cite: 856]
        sector.owner_id = 255;
        if (player.disks_on_sectors > 0)
        {
            player.disks_on_sectors--;
        }

        // Rule requirement: Removing an influence disc forces ALL population cubes
        // off that sector back onto the player track architecture [cite: 858]
        const SectorDefinition* def = get_sector_definition(sector.sector_id);

        state.influence_state.player_id = player_id;
        state.influence_state.pending_returns.clear();

        if (def)
        {
            for (size_t i = 0; i < def->slots.size(); ++i)
            {
                // Check if slot bit is flagged occupied inside the mask bitset
                if ((sector.occupied_slots_mask >> i) & 1u)
                {
                    state.influence_state.pending_returns.push_back({def->slots[i].type, false});
                }
            }
            if (sector.orbital_built)
            {
                size_t orbital_slot_idx = def->slots.size();
                if ((sector.occupied_slots_mask >> orbital_slot_idx) & 1u)
                {
                    state.influence_state.pending_returns.push_back({PlanetType::MONEY, true}); // Orbital cube
                }
            }
        }

        // Reset occupancy status of the vacated tile slots
        sector.occupied_slots_mask = 0;

        // Process the queued returns
        process_pending_returns(state);
        return true;
    }

    bool execute_choose_return_track(::State& state, uint8_t player_id, uint8_t track_code)
    {
        InfluenceState& is = state.influence_state;
        if (is.phase != InfluenceState::Phase::choose_return_track || is.player_id != player_id)
        {
            return false;
        }
        if (is.pending_returns.empty())
        {
            return false;
        }

        Player& player = state.players[player_id];
        const auto& pending = is.pending_returns.front();
        PopTrack track = static_cast<PopTrack>(track_code);

        // Validate choice
        std::vector<PopTrack> legal = get_legal_return_tracks(player, pending.type, pending.is_orbital);
        bool is_legal = false;
        for (PopTrack t : legal)
        {
            if (t == track)
            {
                is_legal = true;
                break;
            }
        }
        if (!is_legal)
        {
            return false;
        }

        // Apply choice
        if (track == PopTrack::SCIENCE) player.resources.science_prod++;
        else if (track == PopTrack::MATERIALS) player.resources.materials_prod++;
        else player.resources.gold_prod++;

        // Remove the pending return
        is.pending_returns.erase(is.pending_returns.begin());

        // Process the rest of the pending returns
        process_pending_returns(state);
        return true;
    }

    bool begin_influence(::State& state, uint8_t player_id)
    {
        if (player_id >= state.players.size()) return false;

        InfluenceState& is = state.influence_state;
        is = InfluenceState{};
        is.player_id = player_id;

        // Fetch core influence performance counters via species definition structure profiles
        uint8_t activations = SPECIES_TABLE[static_cast<size_t>(state.players[player_id].species_id)].activations.
            influence;
        is.activations_remaining = activations > 0 ? activations : 1;

        // Mandatory rule: Activating an Influence action automatically restores up to 2 spent colony ships faceup
        refresh_colony_ships(state.players[player_id], 2);

        is.phase = InfluenceState::Phase::choose_influence;
        return true;
    }
} // namespace open_spiel::eclipse
