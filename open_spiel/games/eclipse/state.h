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
    std::vector<bool> colony_ships;
    uint8_t orbitals, monoliths;
    std::array<Blueprint, 4> blueprints;
    std::vector<ReputationTiles> reputation_tiles;
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

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Player, id, score, species_id, is_ai, has_passed, disks_on_sectors, disks_on_actions, resources, colony_ships, orbitals, monoliths, blueprints, reputation_tiles, trade_rate, researched_techs);

struct State {
    std::vector<Player> players;
    Galaxy galaxy;
    std::vector<ReputationTiles> reputation_tiles;
    std::vector<Unit> unit_registry;

    // Tech market and NPC states
    std::unordered_map<TechBit, uint8_t> tech_tray;
    std::vector<TechBit> tech_bag;
    NPCDifficulty gcds_difficulty = NPCDifficulty::EASY;
    NPCDifficulty guardian_difficulty = NPCDifficulty::EASY;
    NPCDifficulty ancient_difficulty = NPCDifficulty::EASY;

    // Turn tracking and passing queue
    uint8_t current_player;
    uint8_t current_phase;
    uint8_t current_round;
    uint8_t turn_order[MAX_PLAYERS];
    std::vector<uint8_t> pass_order;
    std::mt19937_64 rng;
};

inline void to_json(nlohmann::json& j, const State& s) {
    nlohmann::json tray_j = nlohmann::json::object();
    for (const auto& [bit, count] : s.tech_tray) {
        std::string name = "Unknown Tech";
        for (const auto& def : TECH_TABLE) {
            if (def.bit == bit) {
                name = def.name;
                break;
            }
        }
        tray_j[name] = count;
    }

    j = nlohmann::json{
        {"players", s.players},
        {"galaxy", s.galaxy},
        {"reputation_tiles", s.reputation_tiles},
        {"unit_registry", s.unit_registry},
        {"tech_tray", tray_j},
        {"tech_bag", s.tech_bag},
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
    
    s.tech_tray.clear();
    if (j.contains("tech_tray")) {
        auto tray_j = j.at("tech_tray");
        for (auto& el : tray_j.items()) {
            const std::string& name = el.key();
            uint8_t count = el.value().get<uint8_t>();
            
            TechBit bit = TechBit::NONE;
            for (const auto& def : TECH_TABLE) {
                if (def.name == name) {
                    bit = def.bit;
                    break;
                }
            }
            if (bit != TechBit::NONE) {
                s.tech_tray[bit] = count;
            }
        }
    }

    j.at("tech_bag").get_to(s.tech_bag);
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
