//
// Created by Gemini AI on 06/09/2026.
//

#include "open_spiel/games/eclipse/systems/actions/upgrade.h"

#include <algorithm>
#include "open_spiel/games/eclipse/state.h"
#include "open_spiel/games/eclipse/tech.h"
#include "open_spiel/games/eclipse/species.h"

namespace open_spiel::eclipse
{
    namespace
    {
        // Internal structural processing step to tick down tracking or close state sub-phase loop safely.
        void end_upgrade_activation(::State& state)
        {
            if (state.upgrade_state.activations_remaining > 0)
            {
                --state.upgrade_state.activations_remaining;
            }

            if (state.upgrade_state.activations_remaining == 0)
            {
                state.upgrade_state.phase = ::UpgradeState::Phase::inactive;
                state.upgrade_state.player_id = 255;
            }
            else
            {
                state.upgrade_state.phase = ::UpgradeState::Phase::choose_upgrade;
            }
        }
    } // namespace

    bool can_upgrade(const ::State& state, const uint8_t player_id, const ShipType ship_type, const uint8_t slot_idx, const ShipPartId part_id)
    {
        if (player_id >= state.players.size()) return false;
        const Player& player = state.players[player_id];

        // Safe index conversion from Blueprint indexing assumptions
        const size_t bp_idx = static_cast<size_t>(ship_type);
        if (bp_idx >= player.blueprints.size()) return false;

        const Blueprint& current_bp = player.blueprints[bp_idx];
        if (slot_idx >= current_bp.capacity) return false;

        // Reject no-op assignments (also keeps free removals from looping forever)
        if (current_bp.slots[slot_idx] == part_id) return false;

        // Extract required data if shifting structural items into empty or existing spaces
        if (part_id == ShipPartId::NONE)
        {
            // Validating clear/removal configuration changes safely.
            // Check structural invariant rule: Mobile units must retain at least 1 engine drive to move.
            if (ship_type != ShipType::STARBASE)
            {
                int drive_count = 0;
                for (uint8_t i = 0; i < current_bp.capacity; ++i)
                {
                    if (i == slot_idx) continue; // Assume removal target is omitted
                    ShipPartId existing_id = current_bp.slots[i];
                    if (existing_id != ShipPartId::NONE)
                    {
                        const ShipPart& existing_part = SHIP_PART_TABLE[static_cast<size_t>(existing_id) - 1];
                        if (existing_part.added_movement > 0) ++drive_count;
                    }
                }
                if (drive_count == 0) return false; // Thwarting engine-less mobile layouts
            }

            // Removing a part (e.g. an energy source) must keep the grid energy-positive
            Blueprint removal_bp = current_bp;
            removal_bp.slots[slot_idx] = ShipPartId::NONE;
            removal_bp.recompute();
            return removal_bp.total_stats.energy_net >= 0;
        }

        // Table index verification bounds guard
        const size_t part_table_idx = static_cast<size_t>(part_id) - 1;
        constexpr size_t total_parts = sizeof(SHIP_PART_TABLE) / sizeof(SHIP_PART_TABLE[0]);
        if (part_table_idx >= total_parts) return false;

        const ShipPart& targeted_part = SHIP_PART_TABLE[part_table_idx];

        // Starbase specific rule constraint filtering
        if (ship_type == ShipType::STARBASE && targeted_part.added_movement > 0)
        {
            return false; // Drive tiles cannot be assigned to structural starbases
        }

        // Technology requirement verification checks
        if (!targeted_part.is_discovery)
        {
            if (targeted_part.required_tech.tech != TechBit::NONE && !player.has_tech(targeted_part.required_tech.tech))
            {
                return false; // Blocked due to unlock prerequisites lack
            }
        }
        else
        {
            // Discovery tiles tracking checks are handled contextually or stored independently via separate state layers.
            // Ensure proper default allowance constraints behavior unless explicit mapping tracks individual token counts.
            return false;
        }

        // Financial & Energy Simulation Balance check via localized mutation footprinting
        // Emulate grid changes using a hypothetical balance sheet pass
        Blueprint test_bp = current_bp;
        test_bp.slots[slot_idx] = part_id;
        test_bp.recompute();

        // Prevent layout changes that plunge ship electrical networks into energy deficits
        if (test_bp.total_stats.energy_net < 0) return false;

        // Check layout rules: Mobile units must preserve layout integrity via 1 core propulsion unit minimum
        if (ship_type != ShipType::STARBASE && test_bp.total_stats.movement == 0)
        {
            return false;
        }

        return true;
    }

    bool execute_upgrade(::State& state, const uint8_t player_id, const ShipType ship_type, const uint8_t slot_idx, const ShipPartId part_id)
    {
        if (!can_upgrade(state, player_id, ship_type, slot_idx, part_id)) return false;

        Player& player = state.players[player_id];
        const size_t bp_idx = static_cast<size_t>(ship_type);
        Blueprint& target_bp = player.blueprints[bp_idx];

        // Apply grid change mutation cleanly
        target_bp.slots[slot_idx] = part_id;
        target_bp.recompute();

        // Returning Ship Parts costs nothing; only placements consume an activation.
        if (part_id != ShipPartId::NONE)
        {
            end_upgrade_activation(state);
        }
        return true;
    }

    bool begin_upgrade(::State& state, const uint8_t player_id)
    {
        if (player_id >= state.players.size()) return false;

        UpgradeState& us = state.upgrade_state;
        us = UpgradeState{};
        us.player_id = player_id;

        // Query default structural action values matching base specifications tables
        uint8_t activations = SPECIES_TABLE[static_cast<size_t>(state.players[player_id].species_id)].activations.upgrade;

        // Check for permanent tracking modifiers or technology augmentations bonuses (Pico Modulator yields +2 Upgrade Activations)
        if (state.players[player_id].has_tech(TechBit::PICO_MODULATOR))
        {
            activations += 2;
        }

        us.activations_remaining = activations > 0 ? activations : 1;
        us.phase = ::UpgradeState::Phase::choose_upgrade;
        return true;
    }
} // namespace open_spiel::eclipse