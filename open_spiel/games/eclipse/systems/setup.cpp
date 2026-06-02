//
// Created by Mihai on 28/05/2026.
//

#include "setup.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "../species.h"
#include "../sectors.h"

namespace {

// Pre-defined balanced starting positions in Ring II (distance 2 from center)
static constexpr HexCoord kBalancedPositions[] = {
    { 2, -2}, // Position 0
    { 0, -2}, // Position 1
    {-2,  0}, // Position 2
    {-2,  2}, // Position 3
    { 0,  2}, // Position 4
    { 2,  0}  // Position 5
};

std::vector<size_t> PlayerPositionIndices(size_t player_count) {
    if (player_count == 6) return {0, 1, 2, 3, 4, 5};
    if (player_count == 5) return {0, 1, 2, 3, 4};
    if (player_count == 4) return {0, 1, 3, 4};
    if (player_count == 3) return {0, 2, 4};
    return {0, 3};
}

std::vector<size_t> GuardianPositionIndices(size_t player_count) {
    if (player_count == 6) return {};
    if (player_count == 5) return {5};
    if (player_count == 4) return {2, 5};
    if (player_count == 3) return {1, 3, 5};
    return {1, 2, 4, 5};
}

}  // namespace

SetupConfig NormalizeSetupConfig(SetupConfig config) {
    if (config.players < 2 || config.players > MAX_PLAYERS) {
        throw std::invalid_argument("players must be between 2 and 6");
    }
    config.staged_players.resize(config.players);
    return config;
}


State InitializeDeterministicSetupState(const SetupConfig& raw_config) {
    SetupConfig config = NormalizeSetupConfig(raw_config);
    State state{};

    state.gcds_difficulty = config.npc_difficulty;
    state.guardian_difficulty = config.npc_difficulty;
    state.ancient_difficulty = config.npc_difficulty;

    state.players.clear();
    for (uint8_t i = 0; i < config.players; ++i) {
        Player player{};
        player.id = i;
        player.species_id = config.staged_players[i].species;
        player.is_ai = config.staged_players[i].is_ai;
        player.score = 0;
        player.has_passed = false;
        state.players.push_back(player);
    }

    state.reputation_tiles.clear();
    state.tech_bag.clear();
    state.unit_registry.clear();
    state.pass_order.clear();
    state.tech_tray.fill(0);
    state.current_player = 255;
    state.current_round = 1;
    state.current_phase = 0;
    for (uint8_t& player_id : state.turn_order) {
        player_id = 255;
    }
    return state;
}

void ResolveInitialSetupRandomness(std::mt19937_64& rng,
                                   const SetupConfig& raw_config,
                                   State& state) {
    const SetupConfig config = NormalizeSetupConfig(raw_config);

    // 1. Populate the tech bag with standard tiles from the TECH_TABLE.
    state.tech_bag.clear();
    for (const auto& tech_def : TECH_TABLE) {
        for (uint8_t c = 0; c < tech_def.copies; ++c) {
            state.tech_bag.push_back(tech_def.bit);
        }
    }
    std::shuffle(state.tech_bag.begin(), state.tech_bag.end(), rng);

    // 2. Draw initial tech tiles for the round 1 market based on player count.
    uint8_t target_regular_tiles = 12;
    if (config.players == 3) target_regular_tiles = 14;
    else if (config.players == 4) target_regular_tiles = 16;
    else if (config.players == 5) target_regular_tiles = 18;
    else if (config.players >= 6) target_regular_tiles = 20;

    uint8_t regular_drawn = 0;
    state.tech_tray.fill(0);
    while (regular_drawn < target_regular_tiles && !state.tech_bag.empty()) {
        TechBit drawn = state.tech_bag.back();
        state.tech_bag.pop_back();

        TechCategory category = TechCategory::MILITARY;
        for (const auto& def : TECH_TABLE) {
            if (def.bit == drawn) {
                category = def.category;
                break;
            }
        }

        state.add_to_tech_tray(drawn);
        if (category != TechCategory::RARE) {
            ++regular_drawn;
        }
    }

    // 3. Populate and shuffle the reputation tiles bag.
    state.reputation_tiles.clear();
    for (size_t i = 0; i < 4; ++i) {
        ReputationTiles tile = static_cast<ReputationTiles>(i);
        for (int copies = 0; copies < REPUTATION_TILE_COUNTS[i]; ++copies) {
            state.reputation_tiles.push_back(tile);
        }
    }
    std::shuffle(state.reputation_tiles.begin(), state.reputation_tiles.end(),
                 rng);

    // 4. Randomize initial turn order.
    std::vector<uint8_t> initial_turns(config.players);
    for (uint8_t i = 0; i < config.players; ++i) {
        initial_turns[i] = i;
    }
    std::shuffle(initial_turns.begin(), initial_turns.end(), rng);
    for (size_t i = 0; i < MAX_PLAYERS; ++i) {
        state.turn_order[i] =
            i < initial_turns.size() ? initial_turns[i] : static_cast<uint8_t>(255);
    }
    state.current_player = state.turn_order[0];
    state.current_round = 1;
    state.current_phase = 0;
    state.pass_order.clear();

    // 5. Populate sector bags as bitmasks.
    state.sector_bag_inner = (1U << 10) - 1;
    state.sector_bag_middle = (1U << 13) - 1;

    std::vector<uint8_t> outer_indices(22);
    std::iota(outer_indices.begin(), outer_indices.end(), 0);
    std::shuffle(outer_indices.begin(), outer_indices.end(), rng);

    size_t target_outer_tiles = 18;
    if (config.players == 2) target_outer_tiles = 5;
    else if (config.players == 3) target_outer_tiles = 8;
    else if (config.players == 4) target_outer_tiles = 14;
    else if (config.players == 5) target_outer_tiles = 16;

    state.sector_bag_outer = 0;
    for (size_t i = 0; i < target_outer_tiles && i < outer_indices.size(); ++i) {
        state.sector_bag_outer |= (1U << outer_indices[i]);
    }

    // 6. Populate the fixed galaxy sectors resolved during setup.
    state.galaxy = Galaxy{};
    state.unit_registry.clear();

    state.galaxy.at(0, 0) = Sector{
        .sector_id = 1,
        .owner_id = 255,
        .coords = {0, 0},
        .rotation = 0,
        .points = 4,
        .occupied_slots_mask = 0,
        .discovery_tile_present = true,
        .orbital_built = false,
        .monolith_built = false,
    };
    state.unit_registry.push_back(Unit{
        .player_id = NPC_PLAYER_ID,
        .type = ShipType::GCDS,
        .sector_id = 1,
        .damage = 0,
    });

    const auto guardian_positions = GuardianPositionIndices(config.players);
    if (!guardian_positions.empty()) {
        std::vector<uint16_t> guardian_sectors = {271, 272, 273, 274};
        std::shuffle(guardian_sectors.begin(), guardian_sectors.end(), rng);

        for (size_t i = 0; i < guardian_positions.size(); ++i) {
            const size_t pos_idx = guardian_positions[i];
            const HexCoord coord = kBalancedPositions[pos_idx];
            const uint16_t sector_id =
                guardian_sectors[i % guardian_sectors.size()];
            const SectorDefinition* sector_def = get_sector_definition(sector_id);

            state.galaxy.at(coord.q, coord.r) = Sector{
                .sector_id = sector_id,
                .owner_id = 255,
                .coords = coord,
                .rotation = 0,
                .points = sector_def ? sector_def->points : static_cast<uint8_t>(2),
                .occupied_slots_mask = 0,
                .discovery_tile_present = true,
                .orbital_built = false,
                .monolith_built = false,
            };
            state.unit_registry.push_back(Unit{
                .player_id = NPC_PLAYER_ID,
                .type = ShipType::GUARDIAN,
                .sector_id = sector_id,
                .damage = 0,
            });
        }
    }
}

void FinalizeGameSetup(State& state,
                       const std::vector<PlayerConfig>& player_choices) {
    const size_t player_count = player_choices.size();
    FixedVector<Unit, 128> preserved_units;
    for (const Unit& unit : state.unit_registry) {
        if (unit.player_id == NPC_PLAYER_ID) {
            preserved_units.push_back(unit);
        }
    }
    state.unit_registry = preserved_units;

    // 1. Initialize player boards, blueprints, and resource tracks.
    for (size_t i = 0; i < player_count; ++i) {
        Player& player = state.players[i];
        const PlayerConfig& config = player_choices[i];
        player.species_id = config.species;
        player.is_ai = config.is_ai;
        player.score = 0;
        player.has_passed = false;

        const SpeciesData& species_data =
            SPECIES_TABLE[static_cast<size_t>(config.species)];
        player.resources = species_data.starting_resources;
        player.trade_rate = species_data.trade_rate;
        player.disks_on_sectors = 1;
        player.disks_on_actions = 0;
        player.orbitals = 0;
        player.monoliths = 0;
        player.researched_techs = species_data.starting_techs;
        player.colony_ships_total = 0;
        player.colony_ships_available = 0;
        player.reputation_tiles.clear();

        for (size_t ship_idx = 0; ship_idx < 4; ++ship_idx) {
            Blueprint& blueprint = player.blueprints[ship_idx];
            ShipType ship_type = static_cast<ShipType>(ship_idx);

            if (config.species == Species::PLANTA) {
                if (ship_type == ShipType::INTERCEPTOR) blueprint.capacity = 3;
                else if (ship_type == ShipType::CRUISER) blueprint.capacity = 5;
                else if (ship_type == ShipType::DREADNOUGHT) blueprint.capacity = 7;
                else if (ship_type == ShipType::STARBASE) blueprint.capacity = 3;
            } else {
                if (ship_type == ShipType::INTERCEPTOR) blueprint.capacity = 4;
                else if (ship_type == ShipType::CRUISER) blueprint.capacity = 6;
                else if (ship_type == ShipType::DREADNOUGHT) blueprint.capacity = 8;
                else if (ship_type == ShipType::STARBASE) blueprint.capacity = 4;
            }

            for (size_t slot = 0; slot < 8; ++slot) {
                blueprint.slots[slot] = ShipPartId::NONE;
            }
            blueprint.base_stats = ShipStats{};

            if (ship_type == ShipType::INTERCEPTOR) {
                blueprint.base_stats.initiative = 2;
                blueprint.base_stats.hull = 1;
                blueprint.slots[0] = ShipPartId::ION_CANNON;
                blueprint.slots[1] = ShipPartId::NUCLEAR_SOURCE;
                blueprint.slots[2] = ShipPartId::NUCLEAR_DRIVE;
            } else if (ship_type == ShipType::CRUISER) {
                blueprint.base_stats.initiative = 1;
                blueprint.base_stats.hull = 1;
                blueprint.slots[0] = ShipPartId::ION_CANNON;
                blueprint.slots[1] = ShipPartId::ION_CANNON;
                blueprint.slots[2] = ShipPartId::NUCLEAR_SOURCE;
                blueprint.slots[3] = ShipPartId::NUCLEAR_DRIVE;
                blueprint.slots[4] = ShipPartId::HULL;
                blueprint.slots[5] = ShipPartId::ELECTRON_COMPUTER;
            } else if (ship_type == ShipType::DREADNOUGHT) {
                blueprint.base_stats.initiative = 0;
                blueprint.base_stats.hull = 1;
                blueprint.slots[0] = ShipPartId::ION_CANNON;
                blueprint.slots[1] = ShipPartId::ION_CANNON;
                blueprint.slots[2] = ShipPartId::ION_CANNON;
                blueprint.slots[3] = ShipPartId::NUCLEAR_SOURCE;
                blueprint.slots[4] = ShipPartId::NUCLEAR_SOURCE;
                blueprint.slots[5] = ShipPartId::NUCLEAR_DRIVE;
                blueprint.slots[6] = ShipPartId::HULL;
                blueprint.slots[7] = ShipPartId::ELECTRON_COMPUTER;
            } else if (ship_type == ShipType::STARBASE) {
                blueprint.base_stats.initiative = 4;
                blueprint.base_stats.hull = 1;
                blueprint.slots[0] = ShipPartId::ION_CANNON;
                blueprint.slots[1] = ShipPartId::ION_CANNON;
                blueprint.slots[2] = ShipPartId::NUCLEAR_SOURCE;
                blueprint.slots[3] = ShipPartId::HULL;
            }

            if (config.species == Species::PLANTA) {
                blueprint.base_stats.initiative -= 1;
                blueprint.base_stats.computer += 1;
                blueprint.base_stats.energy_net += 1;
            } else if (config.species == Species::ORION_HEGEMONY) {
                blueprint.base_stats.initiative += 1;
                if (ship_type != ShipType::STARBASE) {
                    blueprint.base_stats.energy_net += 1;
                }
            } else if (config.species == Species::ERIDANI_EMPIRE) {
                if (ship_type != ShipType::STARBASE) {
                    blueprint.base_stats.energy_net += 1;
                }
            }

            blueprint.recompute();
        }
    }

    // 2. Place player starting sectors and spawn their starting ships.
    // Each alien species has a unique home sector, but several players may be
    // Terran Factions — they must occupy distinct home tiles, otherwise a shared
    // sector_id collides in every sector_id-keyed lookup (anchors, ships, claim).
    static const uint16_t kTerranHomeSectors[] = {221, 223, 225, 227, 229, 231};
    constexpr size_t kTerranHomeCount =
        sizeof(kTerranHomeSectors) / sizeof(kTerranHomeSectors[0]);
    size_t terran_seen = 0;
    const auto player_positions = PlayerPositionIndices(player_count);
    for (size_t i = 0; i < player_count; ++i) {
        const Player& player = state.players[i];
        const SpeciesData& species_data =
            SPECIES_TABLE[static_cast<size_t>(player.species_id)];
        uint16_t start_sector_id = species_data.starting_sector;
        if (player.species_id == Species::TERRAN_FACTIONS) {
            if (terran_seen < kTerranHomeCount) {
                start_sector_id = kTerranHomeSectors[terran_seen];
            }
            ++terran_seen;
        }
        const SectorDefinition* sector_def = get_sector_definition(start_sector_id);
        const HexCoord coord = kBalancedPositions[player_positions[i]];

        state.galaxy.at(coord.q, coord.r) = Sector{
            .sector_id = start_sector_id,
            .owner_id = player.id,
            .coords = coord,
            .rotation = 0,
            .points = sector_def ? sector_def->points : static_cast<uint8_t>(3),
            .occupied_slots_mask = 0,
            .discovery_tile_present = false,
            .orbital_built = false,
            .monolith_built = false,
        };

        ShipType starting_ship = ShipType::INTERCEPTOR;
        if (player.species_id == Species::ORION_HEGEMONY) {
            starting_ship = ShipType::CRUISER;
        }
        state.unit_registry.push_back(Unit{
            .player_id = player.id,
            .type = starting_ship,
            .sector_id = start_sector_id,
            .damage = 0,
        });
    }
}

SetupSnapshot CreatePreChoiceSnapshot(const SetupConfig& raw_config) {
    const SetupConfig config = NormalizeSetupConfig(raw_config);
    std::mt19937_64 rng(config.rng_seed);

    SetupSnapshot snapshot;
    snapshot.config = config;
    snapshot.state = InitializeDeterministicSetupState(config);
    ResolveInitialSetupRandomness(rng, config, snapshot.state);
    snapshot.finalized = false;
    return snapshot;
}

SetupSnapshot FinalizeSetupSnapshot(
    const SetupSnapshot& snapshot,
    const std::vector<PlayerConfig>& player_choices) {
    if (player_choices.size() != snapshot.config.players) {
        throw std::invalid_argument("player choice count does not match config");
    }

    SetupSnapshot finalized_snapshot = snapshot;
    FinalizeGameSetup(finalized_snapshot.state, player_choices);
    finalized_snapshot.finalized = true;
    finalized_snapshot.config.staged_players.clear();
    finalized_snapshot.config.staged_players.reserve(player_choices.size());
    for (const PlayerConfig& choice : player_choices) {
        finalized_snapshot.config.staged_players.push_back(
            StagedPlayerConfig{.species = choice.species, .is_ai = choice.is_ai});
    }
    return finalized_snapshot;
}
