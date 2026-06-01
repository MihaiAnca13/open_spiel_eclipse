//
// Created by Mihai on 24/05/2026.
//

#ifndef ECLIPSE_STATE_H
#define ECLIPSE_STATE_H

#include <cstdint>
#include <vector>
#include <array>
#include <unordered_map>
#include <random>
#include <nlohmann/json.hpp>

#include "tech.h"
#include "galaxy.h"
#include "resources.h"
#include "npc.h"
#include "fixed_vector.h"
#include "absl/container/fixed_array.h"

using namespace open_spiel::eclipse;

#define MAX_PLAYERS 6

enum class ShipType { INTERCEPTOR, CRUISER, DREADNOUGHT, STARBASE, ANCIENT, GUARDIAN, GCDS };

NLOHMANN_JSON_SERIALIZE_ENUM( ShipType, {
    {ShipType::INTERCEPTOR, "Interceptor"},
    {ShipType::CRUISER, "Cruiser"},
    {ShipType::DREADNOUGHT, "Dreadnought"},
    {ShipType::STARBASE, "Starbase"},
    {ShipType::ANCIENT, "Ancient"},
    {ShipType::GUARDIAN, "Guardian"},
    {ShipType::GCDS, "GCDS"}
});

constexpr uint8_t NPC_PLAYER_ID = 255;

struct Unit {
    uint8_t player_id; // NPC_PLAYER_ID for non-player units
    ShipType type;
    uint16_t sector_id; // Current location (matches Sector::sector_id)
    uint8_t damage;     // Damage cubes currently assigned
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Unit, player_id, type, sector_id, damage);

enum ReputationTiles { ONE, TWO, THREE, FOUR };

constexpr int REPUTATION_TILE_COUNTS[] = { 12, 10, 7, 4 }; // values are 1, 2, 3, 4

// Index 0 means all cubes are on the board, Index 12 means all cubes are on the track.
static const int POPULATION_PRODUCTION_TABLE[] = { 28, 24, 21, 18, 15, 12, 10, 8, 6, 4, 3, 2, 0 };
// Index 0 means no influence disk is used
static const int INFLUENCE_UPKEEP_TABLE[] = { 0, 0, 1, 2, 3, 5, 7, 10, 13, 17, 21, 25, 30 };

struct Player {
    uint8_t id;
    uint8_t score;
    Species species_id;
    bool is_ai;
    bool has_passed;
    uint8_t disks_on_sectors, disks_on_actions;
    Resources resources;
    uint8_t colony_ships_total = 0;
    uint8_t colony_ships_available = 0;
    uint8_t orbitals, monoliths;
    std::array<Blueprint, 4> blueprints;
    FixedVector<ReputationTiles, 5> reputation_tiles;
    uint8_t trade_rate;
    uint64_t researched_techs = 0; // The actual bitmask

    // Helper to check if a tech is owned
    [[nodiscard]] bool has_tech(TechBit tech) const {
        return (researched_techs & static_cast<uint64_t>(tech)) != 0;
    }
};

static_assert(static_cast<size_t>(ShipType::STARBASE) + 1 == 4, "The first 4 ShipType values must map to blueprints index 0-3");

NLOHMANN_JSON_SERIALIZE_ENUM(ReputationTiles, {
    {ONE, "One"},
    {TWO, "Two"},
    {THREE, "Three"},
    {FOUR, "Four"}
});

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Player, id, score, species_id, is_ai, has_passed, disks_on_sectors, disks_on_actions, resources, colony_ships_total, colony_ships_available, orbitals, monoliths, blueprints, reputation_tiles, trade_rate, researched_techs);

struct State {
    FixedVector<Player, MAX_PLAYERS> players;
    Galaxy galaxy;
    FixedVector<ReputationTiles, 40> reputation_tiles;
    FixedVector<Unit, 128> unit_registry;

    // Tech market and NPC states
    std::array<uint8_t, 40> tech_tray = {0};
    FixedVector<TechBit, 130> tech_bag;
    uint16_t sector_bag_inner;
    uint16_t sector_bag_middle;
    uint32_t sector_bag_outer;
    NPCDifficulty gcds_difficulty = NPCDifficulty::EASY;
    NPCDifficulty guardian_difficulty = NPCDifficulty::EASY;
    NPCDifficulty ancient_difficulty = NPCDifficulty::EASY;

    // Turn tracking and passing queue
    uint8_t current_player;
    uint8_t current_phase;
    uint8_t current_round;
    uint8_t turn_order[MAX_PLAYERS];
    FixedVector<uint8_t, MAX_PLAYERS> pass_order;

    // Helper functions for tech market tray (allocation-free representation)
    uint8_t get_tech_tray_count(TechBit tech) const {
        for (size_t i = 0; i < 40; ++i) {
            if (TECH_TABLE[i].bit == tech) {
                return tech_tray[i];
            }
        }
        return 0;
    }

    void add_to_tech_tray(TechBit tech, uint8_t count = 1) {
        for (size_t i = 0; i < 40; ++i) {
            if (TECH_TABLE[i].bit == tech) {
                tech_tray[i] += count;
                return;
            }
        }
    }

    bool remove_from_tech_tray(TechBit tech, uint8_t count = 1) {
        for (size_t i = 0; i < 40; ++i) {
            if (TECH_TABLE[i].bit == tech) {
                if (tech_tray[i] >= count) {
                    tech_tray[i] -= count;
                    return true;
                }
                return false;
            }
        }
        return false;
    }
};

inline void to_json(nlohmann::json& j, const State& s) {
    nlohmann::json tray_j = nlohmann::json::object();
    for (size_t i = 0; i < 40; ++i) {
        if (s.tech_tray[i] > 0) {
            std::string name = TECH_TABLE[i].name;
            tray_j[name] = s.tech_tray[i];
        }
    }

    j = nlohmann::json{
        {"players", s.players},
        {"galaxy", s.galaxy},
        {"reputation_tiles", s.reputation_tiles},
        {"unit_registry", s.unit_registry},
        {"tech_tray", tray_j},
        {"tech_bag", s.tech_bag},
        {"sector_bag_inner", s.sector_bag_inner},
        {"sector_bag_middle", s.sector_bag_middle},
        {"sector_bag_outer", s.sector_bag_outer},
        {"gcds_difficulty", s.gcds_difficulty},
        {"guardian_difficulty", s.guardian_difficulty},
        {"ancient_difficulty", s.ancient_difficulty},
        {"current_player", s.current_player},
        {"current_phase", s.current_phase},
        {"current_round", s.current_round},
        {"turn_order", s.turn_order},
        {"pass_order", s.pass_order}
    };
}

inline void from_json(const nlohmann::json& j, State& s) {
    j.at("players").get_to(s.players);
    j.at("galaxy").get_to(s.galaxy);
    j.at("reputation_tiles").get_to(s.reputation_tiles);
    j.at("unit_registry").get_to(s.unit_registry);
    
    s.tech_tray.fill(0);
    if (j.contains("tech_tray")) {
        auto tray_j = j.at("tech_tray");
        for (auto& el : tray_j.items()) {
            const std::string& name = el.key();
            uint8_t count = el.value().get<uint8_t>();
            
            for (size_t i = 0; i < 40; ++i) {
                if (TECH_TABLE[i].name == name) {
                    s.tech_tray[i] = count;
                    break;
                }
            }
        }
    }

    j.at("tech_bag").get_to(s.tech_bag);
    if (j.contains("sector_bag_inner")) {
        j.at("sector_bag_inner").get_to(s.sector_bag_inner);
    }
    if (j.contains("sector_bag_middle")) {
        j.at("sector_bag_middle").get_to(s.sector_bag_middle);
    }
    if (j.contains("sector_bag_outer")) {
        j.at("sector_bag_outer").get_to(s.sector_bag_outer);
    }
    j.at("gcds_difficulty").get_to(s.gcds_difficulty);
    j.at("guardian_difficulty").get_to(s.guardian_difficulty);
    j.at("ancient_difficulty").get_to(s.ancient_difficulty);
    j.at("current_player").get_to(s.current_player);
    j.at("current_phase").get_to(s.current_phase);
    j.at("current_round").get_to(s.current_round);
    
    auto turn_order_j = j.at("turn_order");
    for (size_t i = 0; i < MAX_PLAYERS; ++i) {
        s.turn_order[i] = turn_order_j.at(i).get<uint8_t>();
    }
    
    j.at("pass_order").get_to(s.pass_order);
}

#endif //ECLIPSE_STATE_H
