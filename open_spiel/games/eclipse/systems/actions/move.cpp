//
// Move actions: relocating ships along the wormhole / warp portal network.
//
// Optimized for high-frequency legal-action generation in MCTS/RL:
// - board-derived movement data is built once per public query/execution path;
// - sector pinning availability is precomputed once for bulk legal move generation;
// - execution paths validate directly instead of allocating/scanning legal vectors.
//

#include "open_spiel/games/eclipse/systems/actions/move.h"

#include <array>
#include <cstddef>

#include "open_spiel/games/eclipse/state.h"
#include "open_spiel/games/eclipse/galaxy.h"
#include "open_spiel/games/eclipse/species.h"

namespace open_spiel::eclipse
{
    namespace
    {
        // Highest known valid sector id. The map needs MAX_SECTOR_ID + 1 slots
        // because sector ids are used directly as indices.
        constexpr uint16_t MAX_SECTOR_ID = 395;
        constexpr std::size_t SECTOR_ID_LIMIT = static_cast<std::size_t>(MAX_SECTOR_ID) + 1;
        constexpr uint8_t INVALID_CELL = 255;

        using SectorCellMap = std::array<uint8_t, SECTOR_ID_LIMIT>;
        using SectorFlagMap = std::array<uint8_t, SECTOR_ID_LIMIT>;
        using CellSectorMap = std::array<uint16_t, GALAXY_CELL_COUNT>;
        using CellMaskMap = std::array<uint8_t, GALAXY_CELL_COUNT>;
        using CellFlagMap = std::array<uint8_t, GALAXY_CELL_COUNT>;

        bool valid_sector_id(uint16_t sector_id)
        {
            return sector_id <= MAX_SECTOR_ID;
        }

        // Per-call board cache. This intentionally remains local to avoid
        // increasing State size or requiring all board-mutating systems to
        // maintain derived indexes. It still collapses several repeated O(C)
        // scans and get_sector_definition() calls into one pass.
        struct MoveBoardCache
        {
            SectorCellMap sector_to_cell{};
            CellSectorMap cell_to_sector{};
            CellMaskMap rotated_wormholes{};
            CellFlagMap has_sector_definition{};
            CellFlagMap has_warp_portal{};
            uint8_t warp_portal_count = 0;

            explicit MoveBoardCache(const ::State& state)
            {
                sector_to_cell.fill(INVALID_CELL);
                cell_to_sector.fill(0);
                rotated_wormholes.fill(0);
                has_sector_definition.fill(0);
                has_warp_portal.fill(0);

                for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q)
                {
                    for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r)
                    {
                        const uint8_t cell = static_cast<uint8_t>(hex_to_index(q, r));
                        const Sector& sector = state.galaxy.at(q, r);
                        const uint16_t sector_id = sector.sector_id;
                        cell_to_sector[cell] = sector_id;

                        if (sector_id == 0) continue;

                        if (valid_sector_id(sector_id))
                        {
                            sector_to_cell[sector_id] = cell;
                        }

                        const SectorDefinition* def = get_sector_definition(sector_id);
                        if (def != nullptr)
                        {
                            has_sector_definition[cell] = 1;
                            rotated_wormholes[cell] = rotate_edge_mask(def->wormholes_mask, sector.rotation);
                        }

                        if (sector.has_player_warp_portal || (def != nullptr && def->has_warp_portal))
                        {
                            has_warp_portal[cell] = 1;
                            ++warp_portal_count;
                        }
                    }
                }
            }

            [[nodiscard]] uint8_t cell_for_sector(uint16_t sector_id) const
            {
                if (!valid_sector_id(sector_id)) return INVALID_CELL;
                return sector_to_cell[sector_id];
            }
        };

        // Bulk pinning cache for legal_move_steps()/begin_move(). Public
        // can_leave_sector() remains a direct O(U) query to avoid building this
        // table for one-off validation calls.
        struct PinningAvailability
        {
            SectorFlagMap can_leave{};

            PinningAvailability(const ::State& state, const Player& player, uint8_t player_id)
            {
                std::array<uint8_t, SECTOR_ID_LIMIT> friendly{};
                std::array<uint8_t, SECTOR_ID_LIMIT> opponents{};
                SectorFlagMap has_gcds{};

                can_leave.fill(0);
                has_gcds.fill(0);

                for (const Unit& unit : state.unit_registry)
                {
                    if (!valid_sector_id(unit.sector_id)) continue;
                    const std::size_t sector_idx = unit.sector_id;

                    if (unit.type == ShipType::GCDS)
                    {
                        has_gcds[sector_idx] = 1;
                        continue;
                    }

                    if (unit.type == ShipType::STARBASE)
                    {
                        continue; // Starbases do not count for pinning.
                    }

                    if (unit.player_id == player_id)
                    {
                        ++friendly[sector_idx];
                    }
                    else
                    {
                        ++opponents[sector_idx];
                    }
                }

                const bool cloaking = player.has_tech(TechBit::CLOAKING_DEVICE);
                for (std::size_t sector_idx = 0; sector_idx < SECTOR_ID_LIMIT; ++sector_idx)
                {
                    if (has_gcds[sector_idx] != 0) continue;
                    const uint8_t pinned = cloaking
                        ? static_cast<uint8_t>(opponents[sector_idx] / 2)
                        : opponents[sector_idx];
                    can_leave[sector_idx] = friendly[sector_idx] > pinned ? 1 : 0;
                }
            }

            [[nodiscard]] bool can_leave_from(uint16_t sector_id) const
            {
                return valid_sector_id(sector_id) && can_leave[sector_id] != 0;
            }
        };

        bool is_movable_ship(const Unit& unit, uint8_t player_id)
        {
            if (unit.player_id != player_id) return false;
            switch (unit.type)
            {
                case ShipType::INTERCEPTOR:
                case ShipType::CRUISER:
                case ShipType::DREADNOUGHT:
                    return true;
                default:
                    return false;
            }
        }


        bool can_start_or_continue(const ::MoveState& ms, uint8_t unit_idx)
        {
            if (ms.active_unit_idx == unit_idx && ms.steps_remaining > 0) return true;
            return ms.activations_remaining > 0;
        }

        bool has_wormhole_connection(const MoveBoardCache& cache, uint8_t from_cell,
                                     uint8_t to_cell, uint8_t direction,
                                     bool wormhole_generator)
        {
            if (from_cell >= GALAXY_CELL_COUNT || to_cell >= GALAXY_CELL_COUNT) return false;
            if (cache.cell_to_sector[to_cell] == 0) return false;
            if (cache.has_sector_definition[from_cell] == 0 ||
                cache.has_sector_definition[to_cell] == 0)
            {
                return false;
            }

            const bool my_edge = has_edge(cache.rotated_wormholes[from_cell], direction);
            const bool their_edge = has_edge(cache.rotated_wormholes[to_cell], (direction + 3) % 6);
            return wormhole_generator ? (my_edge || their_edge) : (my_edge && their_edge);
        }

        bool has_other_warp_portal(const MoveBoardCache& cache, uint16_t source_sector_id)
        {
            if (cache.warp_portal_count < 2) return false;
            for (std::size_t cell = 0; cell < GALAXY_CELL_COUNT; ++cell)
            {
                if (cache.has_warp_portal[cell] != 0 && cache.cell_to_sector[cell] != source_sector_id)
                {
                    return true;
                }
            }
            return false;
        }

        struct UnitMoveContext
        {
            const Unit* unit = nullptr;
            uint8_t source_cell = INVALID_CELL;
            uint8_t movement_value = 0;
        };

        bool validate_unit_for_step(const ::State& state, uint8_t player_id, uint8_t unit_idx,
                                    const MoveBoardCache& cache, UnitMoveContext& ctx)
        {
            if (player_id >= state.players.size()) return false;
            if (unit_idx >= state.unit_registry.size()) return false;

            const ::MoveState& ms = state.move_state;
            if (ms.phase != ::MoveState::Phase::choose_move || ms.player_id != player_id) return false;
            if (!can_start_or_continue(ms, unit_idx)) return false;

            const Player& player = state.players[player_id];
            const Unit& unit = state.unit_registry[unit_idx];
            if (!is_movable_ship(unit, player_id)) return false;

            const uint8_t source_cell = cache.cell_for_sector(unit.sector_id);
            if (source_cell == INVALID_CELL) return false;

            const uint8_t movement_value = ship_movement_value(player, unit.type);
            if (movement_value == 0) return false;

            if (!can_leave_sector(state, player_id, unit.sector_id)) return false;

            ctx.unit = &unit;
            ctx.source_cell = source_cell;
            ctx.movement_value = movement_value;
            return true;
        }

        struct StepValidation
        {
            uint16_t dest_sector_id = 0;
            uint8_t movement_value = 0;
        };

        bool validate_regular_step(const ::State& state, uint8_t player_id, uint8_t unit_idx,
                                   uint8_t direction, const MoveBoardCache& cache,
                                   StepValidation& validation)
        {
            if (direction >= 6) return false;

            UnitMoveContext ctx;
            if (!validate_unit_for_step(state, player_id, unit_idx, cache, ctx)) return false;

            const HexCoord from = index_to_hex(ctx.source_cell);
            const int to_q = from.q + HEX_DIRECTIONS[direction].first;
            const int to_r = from.r + HEX_DIRECTIONS[direction].second;
            if (!in_galaxy_bounds(to_q, to_r)) return false;

            const uint8_t to_cell = static_cast<uint8_t>(hex_to_index(to_q, to_r));
            if (cache.cell_to_sector[to_cell] == 0) return false;

            const bool wormhole_generator =
                state.players[player_id].has_tech(TechBit::WORMHOLE_GENERATOR);
            if (!has_wormhole_connection(cache, ctx.source_cell, to_cell,
                                         direction, wormhole_generator))
            {
                return false;
            }

            validation.dest_sector_id = cache.cell_to_sector[to_cell];
            validation.movement_value = ctx.movement_value;
            return true;
        }

        bool validate_warp_entry(const ::State& state, uint8_t player_id, uint8_t unit_idx,
                                 const MoveBoardCache& cache)
        {
            UnitMoveContext ctx;
            if (!validate_unit_for_step(state, player_id, unit_idx, cache, ctx)) return false;
            if (cache.has_warp_portal[ctx.source_cell] == 0) return false;
            return has_other_warp_portal(cache, ctx.unit->sector_id);
        }

        bool unit_can_contribute_legal_step(const ::MoveState& ms, const Player& player,
                                           const Unit& unit, uint8_t player_id,
                                           uint8_t unit_idx,
                                           const MoveBoardCache& cache,
                                           const PinningAvailability& pinning,
                                           uint8_t& source_cell)
        {
            if (!is_movable_ship(unit, player_id)) return false;
            if (!can_start_or_continue(ms, unit_idx)) return false;
            if (ship_movement_value(player, unit.type) == 0) return false;

            source_cell = cache.cell_for_sector(unit.sector_id);
            if (source_cell == INVALID_CELL) return false;
            return pinning.can_leave_from(unit.sector_id);
        }

        bool has_any_legal_move(const ::State& state, uint8_t player_id,
                                const MoveBoardCache& cache,
                                const PinningAvailability& pinning)
        {
            const ::MoveState& ms = state.move_state;
            const Player& player = state.players[player_id];
            const bool wormhole_generator = player.has_tech(TechBit::WORMHOLE_GENERATOR);

            const std::size_t unit_count = state.unit_registry.size();
            for (std::size_t idx = 0; idx < unit_count; ++idx)
            {
                const uint8_t unit_idx = static_cast<uint8_t>(idx);
                const Unit& unit = state.unit_registry[idx];

                uint8_t source_cell = INVALID_CELL;
                if (!unit_can_contribute_legal_step(ms, player, unit, player_id, unit_idx,
                                                    cache, pinning, source_cell))
                {
                    continue;
                }

                const HexCoord from = index_to_hex(source_cell);
                for (uint8_t d = 0; d < 6; ++d)
                {
                    const int to_q = from.q + HEX_DIRECTIONS[d].first;
                    const int to_r = from.r + HEX_DIRECTIONS[d].second;
                    if (!in_galaxy_bounds(to_q, to_r)) continue;

                    const uint8_t to_cell = static_cast<uint8_t>(hex_to_index(to_q, to_r));
                    if (has_wormhole_connection(cache, source_cell, to_cell,
                                                d, wormhole_generator))
                    {
                        return true;
                    }
                }

                if (cache.has_warp_portal[source_cell] != 0 &&
                    has_other_warp_portal(cache, unit.sector_id))
                {
                    return true;
                }
            }
            return false;
        }

        // Consumes a movement step, opening a new activation when needed and
        // closing the whole action once every activation is spent.
        void apply_step_bookkeeping(::State& state, uint8_t unit_idx, uint8_t movement_value)
        {
            ::MoveState& ms = state.move_state;
            if (ms.active_unit_idx != unit_idx || ms.steps_remaining == 0)
            {
                // Fresh activation, possibly ending a previous ship's activation early.
                --ms.activations_remaining;
                ms.active_unit_idx = unit_idx;
                ms.steps_remaining = movement_value;
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
        const std::size_t bp_idx = static_cast<std::size_t>(ship_type);
        if (bp_idx >= player.blueprints.size()) return 0;
        return player.blueprints[bp_idx].total_stats.movement;
    }

    bool can_leave_sector(const ::State& state, uint8_t player_id, uint16_t sector_id)
    {
        if (player_id >= state.players.size()) return false;

        int friendly = 0;
        int opponents = 0;
        for (const Unit& unit : state.unit_registry)
        {
            if (unit.sector_id != sector_id) continue;
            if (unit.type == ShipType::GCDS) return false;  // GCDS pins all ships.
            if (unit.type == ShipType::STARBASE) continue;  // Starbases do not count for pinning.
            if (unit.player_id == player_id) ++friendly;
            else ++opponents;
        }

        int pinned = opponents;
        if (state.players[player_id].has_tech(TechBit::CLOAKING_DEVICE))
        {
            pinned = opponents / 2;  // Two opposing ships are required to pin each of yours.
        }
        return friendly > pinned;
    }

    bool can_move_step(const ::State& state, uint8_t player_id, uint8_t unit_idx, uint8_t direction)
    {
        const MoveBoardCache cache(state);
        StepValidation validation;
        return validate_regular_step(state, player_id, unit_idx, direction, cache, validation);
    }

    bool can_begin_warp_move(const ::State& state, uint8_t player_id, uint8_t unit_idx)
    {
        if (player_id >= state.players.size()) return false;
        if (unit_idx >= state.unit_registry.size()) return false;

        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_move || ms.player_id != player_id) return false;
        if (!can_start_or_continue(ms, unit_idx)) return false;

        const MoveBoardCache cache(state);
        return validate_warp_entry(state, player_id, unit_idx, cache);
    }

    std::vector<MoveStepOption> legal_move_steps(const ::State& state, uint8_t player_id)
    {
        std::vector<MoveStepOption> options;
        if (player_id >= state.players.size()) return options;

        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_move || ms.player_id != player_id) return options;

        const Player& player = state.players[player_id];
        const bool wormhole_generator = player.has_tech(TechBit::WORMHOLE_GENERATOR);
        const MoveBoardCache cache(state);
        const PinningAvailability pinning(state, player, player_id);

        // Public API still returns std::vector, so reserve the compact action upper bound
        // once rather than paying for repeated reallocations during push_back.
        options.reserve(state.unit_registry.size() * MOVE_CODES_PER_UNIT);

        const std::size_t unit_count = state.unit_registry.size();
        for (std::size_t idx = 0; idx < unit_count; ++idx)
        {
            const uint8_t unit_idx = static_cast<uint8_t>(idx);
            const Unit& unit = state.unit_registry[idx];

            uint8_t source_cell = INVALID_CELL;
            if (!unit_can_contribute_legal_step(ms, player, unit, player_id, unit_idx,
                                                cache, pinning, source_cell))
            {
                continue;
            }

            const HexCoord from = index_to_hex(source_cell);
            for (uint8_t d = 0; d < 6; ++d)
            {
                const int to_q = from.q + HEX_DIRECTIONS[d].first;
                const int to_r = from.r + HEX_DIRECTIONS[d].second;
                if (!in_galaxy_bounds(to_q, to_r)) continue;

                const uint8_t to_cell = static_cast<uint8_t>(hex_to_index(to_q, to_r));
                if (has_wormhole_connection(cache, source_cell, to_cell,
                                            d, wormhole_generator))
                {
                    options.push_back(MoveStepOption{unit_idx, d});
                }
            }

            if (cache.has_warp_portal[source_cell] != 0 &&
                has_other_warp_portal(cache, unit.sector_id))
            {
                options.push_back(MoveStepOption{unit_idx, MOVE_WARP_DIRECTION});
            }
        }
        return options;
    }

    std::vector<uint8_t> legal_warp_destination_cells(const ::State& state, uint8_t player_id)
    {
        std::vector<uint8_t> cells;
        if (player_id >= state.players.size()) return cells;

        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_warp_destination ||
            ms.player_id != player_id || ms.warp_unit_idx >= state.unit_registry.size())
        {
            return cells;
        }

        const Unit& unit = state.unit_registry[ms.warp_unit_idx];
        const MoveBoardCache cache(state);
        const uint8_t source_cell = cache.cell_for_sector(unit.sector_id);
        if (source_cell == INVALID_CELL || cache.has_warp_portal[source_cell] == 0) return cells;
        if (!has_other_warp_portal(cache, unit.sector_id)) return cells;

        cells.reserve(cache.warp_portal_count > 0 ? cache.warp_portal_count - 1 : 0);
        for (std::size_t cell = 0; cell < GALAXY_CELL_COUNT; ++cell)
        {
            if (cache.has_warp_portal[cell] != 0 && cache.cell_to_sector[cell] != unit.sector_id)
            {
                cells.push_back(static_cast<uint8_t>(cell));
            }
        }
        return cells;
    }

    bool execute_move_step(::State& state, uint8_t player_id, uint8_t unit_idx, uint8_t direction)
    {
        const MoveBoardCache cache(state);
        StepValidation validation;
        if (!validate_regular_step(state, player_id, unit_idx, direction, cache, validation)) return false;

        state.unit_registry[unit_idx].sector_id = validation.dest_sector_id;
        state.unit_registry[unit_idx].arrival_order = state.AllocateArrivalOrder();
        apply_step_bookkeeping(state, unit_idx, validation.movement_value);
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
        if (player_id >= state.players.size()) return false;
        if (galaxy_cell_idx >= GALAXY_CELL_COUNT) return false;

        const ::MoveState& ms = state.move_state;
        if (ms.phase != ::MoveState::Phase::choose_warp_destination ||
            ms.player_id != player_id || ms.warp_unit_idx >= state.unit_registry.size())
        {
            return false;
        }

        const uint8_t unit_idx = ms.warp_unit_idx;
        Unit& unit = state.unit_registry[unit_idx];
        if (!is_movable_ship(unit, player_id)) return false;

        const uint8_t movement_value = ship_movement_value(state.players[player_id], unit.type);
        if (movement_value == 0) return false;

        const MoveBoardCache cache(state);
        const uint8_t source_cell = cache.cell_for_sector(unit.sector_id);
        if (source_cell == INVALID_CELL || cache.has_warp_portal[source_cell] == 0) return false;
        if (cache.has_warp_portal[galaxy_cell_idx] == 0) return false;

        const uint16_t dest_sector_id = cache.cell_to_sector[galaxy_cell_idx];
        if (dest_sector_id == 0 || dest_sector_id == unit.sector_id) return false;

        unit.sector_id = dest_sector_id;
        unit.arrival_order = state.AllocateArrivalOrder();
        apply_step_bookkeeping(state, unit_idx, movement_value);
        return true;
    }

    bool begin_move(::State& state, uint8_t player_id)
    {
        if (player_id >= state.players.size()) return false;

        ::MoveState& ms = state.move_state;
        ms = ::MoveState{};
        ms.player_id = player_id;

        uint8_t activations =
            SPECIES_TABLE[static_cast<std::size_t>(state.players[player_id].species_id)].activations.move;

        // Improved Logistics grants 1 additional Move Activation per Move action.
        if (state.players[player_id].has_tech(TechBit::IMPROVED_LOGISTICS))
        {
            ++activations;
        }

        ms.activations_remaining = activations > 0 ? activations : 1;
        ms.phase = ::MoveState::Phase::choose_move;

        // Stay inactive when there is nothing to move so no disc is wasted.
        const MoveBoardCache cache(state);
        const PinningAvailability pinning(state, state.players[player_id], player_id);
        if (!has_any_legal_move(state, player_id, cache, pinning))
        {
            ms = ::MoveState{};
            return false;
        }
        return true;
    }
} // namespace open_spiel::eclipse
