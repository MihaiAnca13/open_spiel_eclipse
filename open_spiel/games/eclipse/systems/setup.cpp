//
// Created by Mihai on 28/05/2026.
//

#include "setup.h"
#include "../species.h"
#include "../sectors.h"
#include <algorithm>
#include <random>

// Pre-defined balanced starting positions in Ring II (distance 2 from center)
static const HexCoord BALANCED_POSITIONS[] = {
    { 2, -2}, // Position 0
    { 0, -2}, // Position 1
    {-2,  0}, // Position 2
    {-2,  2}, // Position 3
    { 0,  2}, // Position 4
    { 2,  0}  // Position 5
};

State initialize_pre_choice_state(unsigned int seed, uint8_t num_players, NPCDifficulty difficulty) {
    State state;
    if (seed == 0) {
        state.rng.seed(std::random_device{}());
    }
    else {
        state.rng.seed(seed);
    }

    // TODO: allow difficulty to be set per NPC and/or random per NPC
    state.gcds_difficulty = difficulty;
    state.guardian_difficulty = difficulty;
    state.ancient_difficulty = difficulty;

    // 1. Populate the tech bag with standard tiles from the TECH_TABLE
    state.tech_bag.clear();
    for (const auto& tech_def : TECH_TABLE) {
        for (uint8_t c = 0; c < tech_def.copies; ++c) {
            state.tech_bag.push_back(tech_def.bit);
        }
    }

    // Shuffle tech bag
    std::shuffle(state.tech_bag.begin(), state.tech_bag.end(), state.rng);

    // 2. Draw initial tech tiles for the round 1 market based on player count
    //    Note: Rare techs drawn during setup are placed but do not count against the limit!
    uint8_t target_regular_tiles = 12;
    if (num_players == 3) target_regular_tiles = 14;
    else if (num_players == 4) target_regular_tiles = 16;
    else if (num_players == 5) target_regular_tiles = 18;
    else if (num_players >= 6) target_regular_tiles = 20;

    uint8_t regular_drawn = 0;
    state.tech_tray.clear();

    while (regular_drawn < target_regular_tiles && !state.tech_bag.empty()) {
        TechBit drawn = state.tech_bag.back();
        state.tech_bag.pop_back();

        // Identify category of drawn technology
        TechCategory cat = TechCategory::MILITARY;
        for (const auto& def : TECH_TABLE) {
            if (def.bit == drawn) {
                cat = def.category;
                break;
            }
        }

        state.tech_tray[drawn]++;
        if (cat != TechCategory::RARE) {
            regular_drawn++;
        }
    }

    // 3. Populate and shuffle reputation tiles bag
    state.reputation_tiles.clear();
    for (size_t i = 0; i < 4; ++i) {
        ReputationTiles tile = static_cast<ReputationTiles>(i);
        for (int c = 0; c < REPUTATION_TILE_COUNTS[i]; ++c) {
            state.reputation_tiles.push_back(tile);
        }
    }
    std::shuffle(state.reputation_tiles.begin(), state.reputation_tiles.end(), state.rng);

    // 4. Pre-generate player shells and randomize initial turn order
    state.players = std::vector<Player>();
    // state.players.clear();
    std::vector<uint8_t> initial_turns(num_players);
    for (uint8_t i = 0; i < num_players; ++i) {
        initial_turns[i] = i;

        Player p_shell;
        p_shell.id = i;
        p_shell.species_id = Species::TERRAN_FACTIONS; // Temporary default species
        p_shell.is_ai = false;
        p_shell.score = 0;
        p_shell.has_passed = false;
        state.players.push_back(p_shell);
    }
    std::shuffle(initial_turns.begin(), initial_turns.end(), state.rng);

    for (size_t i = 0; i < MAX_PLAYERS; ++i) {
        if (i < num_players) {
            state.turn_order[i] = initial_turns[i];
        } else {
            state.turn_order[i] = 255; // Unused player slot sentinel
        }
    }

    state.current_player = state.turn_order[0];
    state.current_round = 1;
    state.current_phase = 0; // Action Phase
    state.pass_order.clear();

    return state;
}

void finalize_game_setup(State& state, const std::vector<PlayerConfig>& player_choices) {
    size_t player_count = player_choices.size();
    // state.players.clear();
    state.unit_registry.clear();

    // 1. Initialize Player Boards, Blueprints, and Resource Tracks
    // Players are pre-allocated in Stage 1, we update them in-place.
    for (size_t i = 0; i < player_count; ++i) {
        Player& p = state.players[i];
        const auto& config = player_choices[i];
        // p.id is already assigned in Stage 1
        p.species_id = config.species;
        p.is_ai = config.is_ai;
        p.score = 0;
        p.has_passed = false;

        const SpeciesData& species_data = SPECIES_TABLE[static_cast<size_t>(config.species)];
        p.resources = species_data.starting_resources;
        p.trade_rate = species_data.trade_rate;
        p.disks_on_sectors = 1; // 1 disk used for starting sector
        p.disks_on_actions = 0;
        p.orbitals = 0;
        p.monoliths = 0;
        p.researched_techs = species_data.starting_techs;

        // Initialize Blueprints with pre-printed starting default parts and base_stats
        for (size_t ship_idx = 0; ship_idx < 4; ++ship_idx) {
            Blueprint& bp = p.blueprints[ship_idx];
            ShipType s_type = static_cast<ShipType>(ship_idx);

            // Set capacity (Planta has 1 less space across all designs)
            if (config.species == Species::PLANTA) {
                if (s_type == ShipType::INTERCEPTOR) bp.capacity = 3;
                else if (s_type == ShipType::CRUISER) bp.capacity = 5;
                else if (s_type == ShipType::DREADNOUGHT) bp.capacity = 7;
                else if (s_type == ShipType::STARBASE) bp.capacity = 3;
            } else {
                if (s_type == ShipType::INTERCEPTOR) bp.capacity = 4;
                else if (s_type == ShipType::CRUISER) bp.capacity = 6;
                else if (s_type == ShipType::DREADNOUGHT) bp.capacity = 8;
                else if (s_type == ShipType::STARBASE) bp.capacity = 4;
            }

            // Standard pre-printed default parts in slots
            for (size_t s = 0; s < 8; ++s) bp.slots[s] = ShipPartId::NONE;

            if (s_type == ShipType::INTERCEPTOR) {
                bp.base_stats.initiative = 2;
                bp.base_stats.hull = 1;
                bp.slots[0] = ShipPartId::ION_CANNON;
                bp.slots[1] = ShipPartId::NUCLEAR_SOURCE;
                bp.slots[2] = ShipPartId::NUCLEAR_DRIVE;
            } else if (s_type == ShipType::CRUISER) {
                bp.base_stats.initiative = 1;
                bp.base_stats.hull = 1;
                bp.slots[0] = ShipPartId::ION_CANNON;
                bp.slots[1] = ShipPartId::ION_CANNON;
                bp.slots[2] = ShipPartId::NUCLEAR_SOURCE;
                bp.slots[3] = ShipPartId::NUCLEAR_DRIVE;
                bp.slots[4] = ShipPartId::HULL;
                bp.slots[5] = ShipPartId::ELECTRON_COMPUTER;
            } else if (s_type == ShipType::DREADNOUGHT) {
                bp.base_stats.initiative = 0;
                bp.base_stats.hull = 1;
                bp.slots[0] = ShipPartId::ION_CANNON;
                bp.slots[1] = ShipPartId::ION_CANNON;
                bp.slots[2] = ShipPartId::ION_CANNON;
                bp.slots[3] = ShipPartId::NUCLEAR_SOURCE;
                bp.slots[4] = ShipPartId::NUCLEAR_SOURCE;
                bp.slots[5] = ShipPartId::NUCLEAR_DRIVE;
                bp.slots[6] = ShipPartId::HULL;
                bp.slots[7] = ShipPartId::ELECTRON_COMPUTER;
            } else if (s_type == ShipType::STARBASE) {
                bp.base_stats.initiative = 4;
                bp.base_stats.hull = 1;
                bp.slots[0] = ShipPartId::ION_CANNON;
                bp.slots[1] = ShipPartId::ION_CANNON;
                bp.slots[2] = ShipPartId::NUCLEAR_SOURCE;
                bp.slots[3] = ShipPartId::HULL;
            }

            // Apply species modifications to pre-printed base_stats
            if (config.species == Species::PLANTA) {
                bp.base_stats.initiative -= 1;
                bp.base_stats.computer += 1;
                bp.base_stats.energy_net += 1;
            } else if (config.species == Species::ORION_HEGEMONY) {
                bp.base_stats.initiative += 1;
                if (s_type != ShipType::STARBASE) {
                    bp.base_stats.energy_net += 1;
                }
            } else if (config.species == Species::ERIDANI_EMPIRE) {
                if (s_type != ShipType::STARBASE) {
                    bp.base_stats.energy_net += 1;
                }
            }

            bp.recompute();
        }
    }

    // 2. Populate spatial grid: center sector (GCDS) and player starting sectors
    // Setup Galactic Center
    state.galaxy.at(0, 0) = Sector{
        .sector_id = 1,
        .owner_id = 255,
        .coords = {0, 0},
        .rotation = 0,
        .points = 4,
        .occupied_slots_mask = 0,
        .discovery_tile_present = true,
        .orbital_built = false,
        .monolith_built = false
    };

    // Spawn Galactic Center Defense System (GCDS)
    Unit gcds_unit {
        .player_id = NPC_PLAYER_ID,
        .type = ShipType::GCDS,
        .sector_id = 1,
        .damage = 0
    };
    state.unit_registry.push_back(gcds_unit);

    // Determine balanced positions assigned to players vs. NPCs (Guardians)
    std::vector<size_t> player_position_indices;
    std::vector<size_t> guardian_position_indices;

    if (player_count == 6) {
        player_position_indices = {0, 1, 2, 3, 4, 5};
    } else if (player_count == 5) {
        player_position_indices = {0, 1, 2, 3, 4};
        guardian_position_indices = {5};
    } else if (player_count == 4) {
        player_position_indices = {0, 1, 3, 4};
        guardian_position_indices = {2, 5};
    } else if (player_count == 3) {
        player_position_indices = {0, 2, 4};
        guardian_position_indices = {1, 3, 5};
    } else { // 2 players
        player_position_indices = {0, 3};
        guardian_position_indices = {1, 2, 4, 5};
    }

    // Place player starting sectors and spawn their starting ships
    for (size_t i = 0; i < player_count; ++i) {
        const Player& player = state.players[i];
        const SpeciesData& species_data = SPECIES_TABLE[static_cast<size_t>(player.species_id)];
        uint16_t start_sector_id = species_data.starting_sector;

        const SectorDefinition* start_sector_def = get_sector_definition(start_sector_id);

        size_t pos_idx = player_position_indices[i];
        HexCoord coord = BALANCED_POSITIONS[pos_idx];

        state.galaxy.at(coord.q, coord.r) = Sector{
            .sector_id = start_sector_id,
            .owner_id = player.id,
            .coords = coord,
            .rotation = 0,
            .points = start_sector_def ? start_sector_def->points : static_cast<uint8_t>(3),
            .occupied_slots_mask = 0,
            .discovery_tile_present = false,
            .orbital_built = false,
            .monolith_built = false
        };

        // Orion starts with a Cruiser, others start with Interceptors
        ShipType start_unit_type = ShipType::INTERCEPTOR;
        if (player.species_id == Species::ORION_HEGEMONY) {
            start_unit_type = ShipType::CRUISER;
        }

        Unit start_unit {
            .player_id = player.id,
            .type = start_unit_type,
            .sector_id = start_sector_id,
            .damage = 0
        };
        state.unit_registry.push_back(start_unit);
    }

    // Place Guardian sectors in empty positions (< 6 players)
    if (!guardian_position_indices.empty()) {
        std::vector<uint16_t> guardian_sectors = {271, 272, 273, 274};
        std::shuffle(guardian_sectors.begin(), guardian_sectors.end(), state.rng);

        for (size_t i = 0; i < guardian_position_indices.size(); ++i) {
            size_t pos_idx = guardian_position_indices[i];
            HexCoord coord = BALANCED_POSITIONS[pos_idx];
            uint16_t g_sector_id = guardian_sectors[i % guardian_sectors.size()];

            const SectorDefinition* g_sector_def = get_sector_definition(g_sector_id);

            state.galaxy.at(coord.q, coord.r) = Sector{
                .sector_id = g_sector_id,
                .owner_id = 255,
                .coords = coord,
                .rotation = 0,
                .points = g_sector_def ? g_sector_def->points : static_cast<uint8_t>(2),
                .occupied_slots_mask = 0,
                .discovery_tile_present = true,
                .orbital_built = false,
                .monolith_built = false
            };

            // Spawn Guardian NPC ship
            Unit guardian_unit {
                .player_id = NPC_PLAYER_ID,
                .type = ShipType::GUARDIAN,
                .sector_id = g_sector_id,
                .damage = 0
            };
            state.unit_registry.push_back(guardian_unit);
        }
    }
}
