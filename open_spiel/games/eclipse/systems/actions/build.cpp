//
// Created by Mihai on 05/06/2026.
//

#include "open_spiel/games/eclipse/systems/actions/build.h"

#include <algorithm>

#include "open_spiel/games/eclipse/galaxy.h"
#include "open_spiel/games/eclipse/sectors.h"
#include "open_spiel/games/eclipse/species.h"
#include "open_spiel/games/eclipse/tech.h"

namespace open_spiel::eclipse
{
    namespace
    {
        // Checks whether the player has any components remaining in their finite component pool supply.
        bool has_available_miniatures(const State& state, const uint8_t player_id, const BuildType type)
        {
            // Count active units of this type on the board currently in registry
            int active_count = 0;
            ShipType target_ship_type;
            int max_supply = 0;

            switch (type)
            {
            case BuildType::INTERCEPTOR:
                target_ship_type = ShipType::INTERCEPTOR;
                max_supply = 8;
                break;
            case BuildType::CRUISER:
                target_ship_type = ShipType::CRUISER;
                max_supply = 4;
                break;
            case BuildType::DREADNOUGHT:
                target_ship_type = ShipType::DREADNOUGHT;
                max_supply = 2;
                break;
            case BuildType::STARBASE:
                target_ship_type = ShipType::STARBASE;
                max_supply = 4;
                break;
            case BuildType::ORBITAL:
            case BuildType::MONOLITH:
                return true;
                // Infrastructure tracking is bounded per sector (max 1 each), pieces are unlimited or handled contextually
            }

            for (const auto& unit : state.unit_registry)
            {
                if (unit.player_id == player_id && unit.type == target_ship_type)
                {
                    active_count++;
                    if (active_count >= max_supply) return false; // Early exit
                }
            }
            return true;
        }

        // Advances or terminates the build lifecycle tracking loop.
        void end_build_activation(State& state)
        {
            if (state.build_state.activations_remaining > 0)
            {
                --state.build_state.activations_remaining;
            }

            if (state.build_state.activations_remaining == 0)
            {
                state.build_state.phase = BuildState::Phase::inactive;
                state.build_state.player_id = 255;
            }
            else
            {
                state.build_state.phase = BuildState::Phase::choose_build;
            }
        }
    } // namespace

    uint8_t calculate_build_cost(const Player& player, const BuildType type)
    {
        // Check if the species is Mechanema for discount structures
        bool is_mechanema = (player.species_id == Species::MECHANEMA);

        switch (type)
        {
        case BuildType::INTERCEPTOR: return is_mechanema ? 2 : 3;
        case BuildType::CRUISER: return is_mechanema ? 4 : 5;
        case BuildType::DREADNOUGHT: return is_mechanema ? 7 : 8;
        case BuildType::STARBASE: return is_mechanema ? 2 : 3;
        case BuildType::ORBITAL: return is_mechanema ? 3 : 4;
        case BuildType::MONOLITH: return is_mechanema ? 8 : 10;
        }
        return 255;
    }

    bool can_build(const State& state, const uint8_t player_id, const BuildType type, const uint8_t galaxy_cell_idx)
    {
        if (player_id >= state.players.size()) return false;
        const Player& player = state.players[player_id];

        // Out of bounds check on galaxy hex coordinates mapping
        HexCoord coord = index_to_hex(galaxy_cell_idx);
        const Sector& sector = state.galaxy.at(coord.q, coord.r);

        // Must be a valid explored sector owned/controlled by the active player
        if (sector.sector_id == 0 || sector.owner_id != player_id) return false;

        // Prerequisite technology checks
        switch (type)
        {
        case BuildType::STARBASE:
            if (!player.has_tech(TechBit::STARBASE)) return false;
            break;
        case BuildType::ORBITAL:
            if (!player.has_tech(TechBit::ORBITAL)) return false;
            if (sector.orbital_built) return false; // Max 1 orbital per sector rule
            break;
        case BuildType::MONOLITH:
            if (!player.has_tech(TechBit::MONOLITH)) return false;
            if (sector.monolith_built) return false; // Max 1 monolith per sector rule
            break;
        default:
            break; // Standard ships (Interceptor, Cruiser, Dreadnought) require no unlock techs to build
        }

        // Physical unit plastic piece limitation checks
        if (!has_available_miniatures(state, player_id, type)) return false;

        // Financial checking (Materials inventory pool)
        uint8_t cost = calculate_build_cost(player, type);
        if (player.resources.materials < cost) return false;

        return true;
    }

    bool execute_build(State& state, uint8_t player_id, BuildType type, uint8_t galaxy_cell_idx)
    {
        if (!can_build(state, player_id, type, galaxy_cell_idx)) return false;

        Player& player = state.players[player_id];
        HexCoord coord = index_to_hex(galaxy_cell_idx);
        Sector& sector = state.galaxy.at(coord.q, coord.r);

        // Spend resource allocation
        uint8_t cost = calculate_build_cost(player, type);
        player.resources.materials -= cost;

        // Deploy infrastructure flags or register units inside the sector spatial configuration
        switch (type)
        {
        case BuildType::ORBITAL:
            sector.orbital_built = true;
            break;
        case BuildType::MONOLITH:
            sector.monolith_built = true;
            break;
        case BuildType::INTERCEPTOR:
            state.unit_registry.push_back(Unit{player_id, ShipType::INTERCEPTOR, sector.sector_id, 0});
            break;
        case BuildType::CRUISER:
            state.unit_registry.push_back(Unit{player_id, ShipType::CRUISER, sector.sector_id, 0});
            break;
        case BuildType::DREADNOUGHT:
            state.unit_registry.push_back(Unit{player_id, ShipType::DREADNOUGHT, sector.sector_id, 0});
            break;
        case BuildType::STARBASE:
            state.unit_registry.push_back(Unit{player_id, ShipType::STARBASE, sector.sector_id, 0});
            break;
        }

        end_build_activation(state);
        return true;
    }

    bool begin_build(State& state, uint8_t player_id)
    {
        if (player_id >= state.players.size()) return false;

        BuildState& bs = state.build_state;
        bs = BuildState{};
        bs.player_id = player_id;

        // Resolve structural modifications from base species table configurations (e.g. Nanorobots checking)
        uint8_t activations = SPECIES_TABLE[static_cast<size_t>(state.players[player_id].species_id)].activations.build;

        // Additional build modifier check from active permanent technologies (Nanorobots grants +1 Activation)
        if (state.players[player_id].has_tech(TechBit::NANOROBOTS))
        {
            activations += 1;
        }

        bs.activations_remaining = activations > 0 ? activations : 1;
        bs.phase = BuildState::Phase::choose_build;
        return true;
    }
} // namespace open_spiel::eclipse
