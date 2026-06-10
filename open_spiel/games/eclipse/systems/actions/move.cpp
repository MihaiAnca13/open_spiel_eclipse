//
// Move actions: relocating ships along the wormhole / warp portal network.
//

#include "open_spiel/games/eclipse/systems/actions/move.h"

#include <array>

#include "open_spiel/games/eclipse/state.h"
#include "open_spiel/games/eclipse/galaxy.h"
#include "open_spiel/games/eclipse/species.h"

namespace open_spiel::eclipse
{
    namespace
    {
        constexpr uint16_t MAX_SECTOR_ID = 395;
        constexpr uint8_t INVALID_CELL = 255;

        // Dense sector_id -> galaxy cell index map, rebuilt in O(GALAXY_CELL_COUNT).
        using SectorCellMap = std::array<uint8_t, MAX_SECTOR_ID>;

        SectorCellMap build_sector_cell_map(const ::State& state)
        {
            SectorCellMap map;
            map.fill(INVALID_CELL);
            for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q)
            {
                for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r)
                {
                    const Sector& sector = state.galaxy.at(q, r);
                    if (sector.sector_id != 0 && sector.sector_id < MAX_SECTOR_ID)
                    {
                        map[sector.sector_id] = static_cast<uint8_t>(hex_to_index(q, r));
                    }
                }
            }
            return map;
        }

        // Only player-built mobile ships may move; starbases and NPCs never do.
        bool is_movable_ship(const Unit& unit, uint8_t player_id)
        {
            if (unit.player_id != player_id) return false;
            return unit.type == ShipType::INTERCEPTOR ||
                   unit.type == ShipType::CRUISER ||
                   unit.type == ShipType::DREADNOUGHT;
        }

        bool sector_has_warp_portal(const Sector& sector)
        {
            if (sector.sector_id == 0) return false;
            if (sector.has_player_warp_portal) return true;
            const SectorDefinition* def = get_sector_definition(sector.sector_id);
            return def != nullptr && def->has_warp_portal;
        }

        // A unit may start a fresh activation, or continue the one it is mid-way through.
        bool can_start_or_continue(const ::MoveState& ms, uint8_t unit_idx)
        {
            if (ms.active_unit_idx == unit_idx && ms.steps_remaining > 0) return true;
            return ms.activations_remaining > 0;
        }

        // Wormhole Connection between adjacent sectors: both facing edges need a
        // wormhole, or just one with the Wormhole Generator tech (half wormhole).
        bool has_wormhole_connection(const Sector& from, const Sector& to,
                                     uint8_t direction, bool wormhole_generator)
        {
            const SectorDefinition* from_def = get_sector_definition(from.sector_id);
            const SectorDefinition* to_def = get_sector_definition(to.sector_id);
            if (from_def == nullptr || to_def == nullptr) return false;

            const uint8_t from_mask = rotate_edge_mask(from_def->wormholes_mask, from.rotation);
            const uint8_t to_mask = rotate_edge_mask(to_def->wormholes_mask, to.rotation);
            const bool my_edge = has_edge(from_mask, direction);
            const bool their_edge = has_edge(to_mask, (direction + 3) % 6);
            return wormhole_generator ? (my_edge || their_edge) : (my_edge && their_edge);
        }

        // Consumes a movement step, opening a new activation when needed and
        // closing the whole action once every activation is spent.
        void apply_step_bookkeeping(::State& state, uint8_t player_id, uint8_t unit_idx)
        {
            ::MoveState& ms = state.move_state;
            if (ms.active_unit_idx != unit_idx || ms.steps_remaining == 0)
            {
                // Fresh activation (possibly ending a previous ship's early).
                --ms.activations_remaining;
                ms.active_unit_idx = unit_idx;
                ms.steps_remaining = ship_movement_value(
                    state.players[player_id], state.unit_registry[unit_idx].type);
            }

            --ms.steps_remaining;
            if (ms.steps_remaining == 0)
            {
                ms.active_unit_idx = 255;
            }

            if (ms.active_unit_idx == 255 && ms.activations_remaining == 0)
            {
                ms = ::MoveState{};
            }
            else
            {
                ms.phase = ::MoveState::Phase::choose_move;
                ms.warp_unit_idx = 255;
            }
        }
    } // namespace

    uint8_t ship_movement_value(const Player& player, ShipType ship_type)
    {
        const size_t bp_idx = static_cast<size_t>(ship_type);
        if (bp_idx >= player.blueprints.size()) return 0;
        return player.blueprints[bp_idx].total_stats.movement;
    }

    bool can_leave_sector(const ::State& state, uint8_t player_id, uint16_t sector_id)
    {
        int friendly = 0;
        int opponents = 0;
        for (const Unit& unit : state.unit_registry)
        {
            if (unit.sector_id != sector_id) continue;
            if (unit.type == ShipType::GCDS) return false;  // GCDS pins all ships
            if (unit.player_id == player_id) ++friendly;
            else ++opponents;
        }

        int pinned = opponents;
        if (state.players[player_id].has_tech(TechBit::CLOAKING_DEVICE))
        {
            pinned = opponents / 2;  // two ships are required to pin each of yours
        }
        return friendly > pinned;
    }

    bool can_move_step(const ::State& state, uint8_t player_id, uint8_t unit_idx, uint8_t direction)
    {
        if (player_id >= state.players.size()) return false;
        if (unit_idx >= state.unit_registry.size() || direction >= 6) return false;

        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_move || ms.player_id != player_id) return false;
        if (!can_start_or_continue(ms, unit_idx)) return false;

        const Unit& unit = state.unit_registry[unit_idx];
        if (!is_movable_ship(unit, player_id)) return false;
        if (ship_movement_value(state.players[player_id], unit.type) == 0) return false;
        if (!can_leave_sector(state, player_id, unit.sector_id)) return false;

        const SectorCellMap cell_map = build_sector_cell_map(state);
        if (unit.sector_id >= MAX_SECTOR_ID || cell_map[unit.sector_id] == INVALID_CELL) return false;
        const HexCoord from = index_to_hex(cell_map[unit.sector_id]);

        const int to_q = from.q + HEX_DIRECTIONS[direction].first;
        const int to_r = from.r + HEX_DIRECTIONS[direction].second;
        if (!in_galaxy_bounds(to_q, to_r)) return false;

        const Sector& dest = state.galaxy.at(to_q, to_r);
        if (dest.sector_id == 0) return false;  // Ships may not move to unexplored zones

        const bool wormhole_generator =
            state.players[player_id].has_tech(TechBit::WORMHOLE_GENERATOR);
        return has_wormhole_connection(state.galaxy.at(from.q, from.r), dest,
                                       direction, wormhole_generator);
    }

    bool can_begin_warp_move(const ::State& state, uint8_t player_id, uint8_t unit_idx)
    {
        if (player_id >= state.players.size()) return false;
        if (unit_idx >= state.unit_registry.size()) return false;

        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_move || ms.player_id != player_id) return false;
        if (!can_start_or_continue(ms, unit_idx)) return false;

        const Unit& unit = state.unit_registry[unit_idx];
        if (!is_movable_ship(unit, player_id)) return false;
        if (ship_movement_value(state.players[player_id], unit.type) == 0) return false;
        if (!can_leave_sector(state, player_id, unit.sector_id)) return false;

        const SectorCellMap cell_map = build_sector_cell_map(state);
        if (unit.sector_id >= MAX_SECTOR_ID || cell_map[unit.sector_id] == INVALID_CELL) return false;
        const HexCoord from = index_to_hex(cell_map[unit.sector_id]);
        if (!sector_has_warp_portal(state.galaxy.at(from.q, from.r))) return false;

        // Any other explored warp portal sector is a valid destination.
        for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q)
        {
            for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r)
            {
                const Sector& sector = state.galaxy.at(q, r);
                if (sector.sector_id != unit.sector_id && sector_has_warp_portal(sector))
                {
                    return true;
                }
            }
        }
        return false;
    }

    std::vector<MoveStepOption> legal_move_steps(const ::State& state, uint8_t player_id)
    {
        std::vector<MoveStepOption> options;
        if (player_id >= state.players.size()) return options;

        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_move || ms.player_id != player_id) return options;

        const Player& player = state.players[player_id];
        const bool wormhole_generator = player.has_tech(TechBit::WORMHOLE_GENERATOR);
        const SectorCellMap cell_map = build_sector_cell_map(state);

        // Whether any warp portal pair exists on the board (cheap pre-pass).
        int warp_portal_sectors = 0;
        for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q)
        {
            for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r)
            {
                if (sector_has_warp_portal(state.galaxy.at(q, r))) ++warp_portal_sectors;
            }
        }

        for (uint8_t unit_idx = 0; unit_idx < state.unit_registry.size(); ++unit_idx)
        {
            const Unit& unit = state.unit_registry[unit_idx];
            if (!is_movable_ship(unit, player_id)) continue;
            if (!can_start_or_continue(ms, unit_idx)) continue;
            if (ship_movement_value(player, unit.type) == 0) continue;
            if (unit.sector_id >= MAX_SECTOR_ID || cell_map[unit.sector_id] == INVALID_CELL) continue;
            if (!can_leave_sector(state, player_id, unit.sector_id)) continue;

            const HexCoord from = index_to_hex(cell_map[unit.sector_id]);
            const Sector& source = state.galaxy.at(from.q, from.r);

            for (uint8_t d = 0; d < 6; ++d)
            {
                const int to_q = from.q + HEX_DIRECTIONS[d].first;
                const int to_r = from.r + HEX_DIRECTIONS[d].second;
                if (!in_galaxy_bounds(to_q, to_r)) continue;
                const Sector& dest = state.galaxy.at(to_q, to_r);
                if (dest.sector_id == 0) continue;
                if (has_wormhole_connection(source, dest, d, wormhole_generator))
                {
                    options.push_back(MoveStepOption{unit_idx, d});
                }
            }

            if (warp_portal_sectors >= 2 && sector_has_warp_portal(source))
            {
                options.push_back(MoveStepOption{unit_idx, MOVE_WARP_DIRECTION});
            }
        }
        return options;
    }

    std::vector<uint8_t> legal_warp_destination_cells(const ::State& state, uint8_t player_id)
    {
        std::vector<uint8_t> cells;
        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_warp_destination ||
            ms.player_id != player_id || ms.warp_unit_idx >= state.unit_registry.size())
        {
            return cells;
        }

        const uint16_t source_sector_id = state.unit_registry[ms.warp_unit_idx].sector_id;
        for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q)
        {
            for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r)
            {
                const Sector& sector = state.galaxy.at(q, r);
                if (sector.sector_id != source_sector_id && sector_has_warp_portal(sector))
                {
                    cells.push_back(static_cast<uint8_t>(hex_to_index(q, r)));
                }
            }
        }
        return cells;
    }

    bool execute_move_step(::State& state, uint8_t player_id, uint8_t unit_idx, uint8_t direction)
    {
        if (!can_move_step(state, player_id, unit_idx, direction)) return false;

        Unit& unit = state.unit_registry[unit_idx];
        const SectorCellMap cell_map = build_sector_cell_map(state);
        const HexCoord from = index_to_hex(cell_map[unit.sector_id]);
        const Sector& dest = state.galaxy.at(from.q + HEX_DIRECTIONS[direction].first,
                                             from.r + HEX_DIRECTIONS[direction].second);

        unit.sector_id = dest.sector_id;
        apply_step_bookkeeping(state, player_id, unit_idx);
        return true;
    }

    bool begin_warp_move(::State& state, uint8_t player_id, uint8_t unit_idx)
    {
        if (!can_begin_warp_move(state, player_id, unit_idx)) return false;

        state.move_state.warp_unit_idx = unit_idx;
        state.move_state.phase = ::MoveState::Phase::choose_warp_destination;
        return true;
    }

    bool execute_warp_move(::State& state, uint8_t player_id, uint8_t galaxy_cell_idx)
    {
        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_warp_destination ||
            ms.player_id != player_id)
        {
            return false;
        }

        const std::vector<uint8_t> cells = legal_warp_destination_cells(state, player_id);
        bool is_legal = false;
        for (uint8_t cell : cells)
        {
            if (cell == galaxy_cell_idx)
            {
                is_legal = true;
                break;
            }
        }
        if (!is_legal) return false;

        const uint8_t unit_idx = ms.warp_unit_idx;
        const HexCoord dest_coord = index_to_hex(galaxy_cell_idx);
        state.unit_registry[unit_idx].sector_id =
            state.galaxy.at(dest_coord.q, dest_coord.r).sector_id;
        apply_step_bookkeeping(state, player_id, unit_idx);
        return true;
    }

    bool begin_move(::State& state, uint8_t player_id)
    {
        if (player_id >= state.players.size()) return false;

        MoveState& ms = state.move_state;
        ms = MoveState{};
        ms.player_id = player_id;

        uint8_t activations = SPECIES_TABLE[static_cast<size_t>(state.players[player_id].species_id)].activations.move;

        // Improved Logistics grants 1 additional Move Activation per Move action.
        if (state.players[player_id].has_tech(TechBit::IMPROVED_LOGISTICS))
        {
            activations += 1;
        }

        ms.activations_remaining = activations > 0 ? activations : 1;
        ms.phase = MoveState::Phase::choose_move;

        // Stay inactive when there is nothing to move so no disc is wasted.
        if (legal_move_steps(state, player_id).empty())
        {
            ms = MoveState{};
            return false;
        }
        return true;
    }
} // namespace open_spiel::eclipse
