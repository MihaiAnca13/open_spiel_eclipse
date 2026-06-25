//
// Created by Mihai on 24/05/2026.
//

#ifndef ECLIPSE_STATE_H
#define ECLIPSE_STATE_H

#include <cstdint>
#include <vector>
#include <array>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "types.h"
#include "tech.h"
#include "galaxy.h"
#include "resources.h"
#include "npc.h"
#include "fixed_vector.h"
#include "species.h"
#include "discovery_tiles.h"
#include "systems/actions/explore.h"
#include "systems/actions/research.h"
#include "systems/actions/build.h"
#include "systems/actions/diplomacy.h"
#include "systems/actions/influence.h"
#include "systems/actions/upgrade.h"
#include "systems/actions/move.h"
#include "systems/combat.h"
#include "systems/scoring.h"
#include "absl/container/fixed_array.h"

using open_spiel::eclipse::FixedVector;
using open_spiel::eclipse::ShipType;
using open_spiel::eclipse::ReputationTiles;
using open_spiel::eclipse::ReputationSlotKind;
using open_spiel::eclipse::CombatState;

#define MAX_PLAYERS 6

constexpr uint8_t NPC_PLAYER_ID = 255;
constexpr int total_influence_discs = 12;

struct Unit {
    uint8_t player_id; // NPC_PLAYER_ID for non-player units
    ShipType type;
    uint16_t sector_id; // Current location (matches Sector::sector_id)
    uint8_t damage;     // Damage cubes currently assigned
    uint32_t arrival_order = 0; // Monotonic counter from State::next_arrival_order; 0 = legacy/default
};

inline void to_json(nlohmann::json& j, const Unit& u) {
    j = nlohmann::json{
        {"player_id", u.player_id},
        {"type", u.type},
        {"sector_id", u.sector_id},
        {"damage", u.damage},
        {"arrival_order", u.arrival_order}
    };
}

inline void from_json(const nlohmann::json& j, Unit& u) {
    j.at("player_id").get_to(u.player_id);
    j.at("type").get_to(u.type);
    j.at("sector_id").get_to(u.sector_id);
    j.at("damage").get_to(u.damage);
    if (j.contains("arrival_order")) {
        j.at("arrival_order").get_to(u.arrival_order);
    } else {
        u.arrival_order = 0;
    }
}

constexpr int REPUTATION_TILE_COUNTS[] = { 12, 10, 7, 4 }; // values are 1, 2, 3, 4

// Index = cubes remaining on the track (12=full/0-production, 0=empty/max-production).
// Resources::gold_prod / science_prod / materials_prod store this index.
// Production value = POPULATION_PRODUCTION_TABLE[cubes_on_track].
static const int POPULATION_PRODUCTION_TABLE[] = { 28, 24, 21, 18, 15, 12, 10, 8, 6, 4, 3, 2, 0 };
// Index 0 means no influence disk is used
static const int INFLUENCE_UPKEEP_TABLE[] = { 0, 0, 1, 2, 3, 5, 7, 10, 13, 17, 21, 25, 30 };

enum class RoundPhase : uint8_t {
    ACTION = 0,
    COMBAT = 1,
    UPKEEP = 2,
    CLEANUP = 3
};

NLOHMANN_JSON_SERIALIZE_ENUM(RoundPhase, {
    {RoundPhase::ACTION, "action"},
    {RoundPhase::COMBAT, "combat"},
    {RoundPhase::UPKEEP, "upkeep"},
    {RoundPhase::CLEANUP, "cleanup"}
});

using open_spiel::eclipse::ReputationSlot;

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
    FixedVector<ReputationSlot, 5> reputation_track;
    uint8_t trade_rate;
    uint8_t extra_influence_discs = 0;
    std::array<uint8_t, 3> graveyard_counts = {0, 0, 0};
    bool eliminated = false;
    uint64_t researched_techs_military = 0;  // standard MIL bits + rare bits
    uint64_t researched_techs_grid = 0;      // standard GRID bits + rare bits
    uint64_t researched_techs_nano = 0;      // standard NANO bits + rare bits

    // End-of-game scoring state
    uint8_t ambassador_tiles_held = 0;     // 1 VP per tile at game end
    uint8_t ambassador_tiles_pending_return = 0;  // tiles freed by a break awaiting track choice
    bool traitor_held = false;             // -2 VP at game end
    uint8_t discovery_vp_tiles_kept = 0;   // 2 VP per tile at game end

    // Ancient / discovery ship parts collected from exploration. Available for
    // placement onto blueprints via the Upgrade action.
    FixedVector<ShipPartId, 24> parts_inventory;

    // True when the player may immediately place a Warp Portal tile on a sector
    // they Control. Triggered by the WARP_PORTAL discovery tile or by researching
    // the WARP_PORTAL Rare Tech. Cleared once the tile is placed.
    bool warp_portal_eligible = false;

    // Helper to check if a tech is owned (checks all 3 track masks).
    [[nodiscard]] bool has_tech(TechBit tech) const {
        uint64_t b = static_cast<uint64_t>(tech);
        return (researched_techs_military & b) ||
               (researched_techs_grid & b) ||
               (researched_techs_nano & b);
    }

    // Helper to calculate available influence discs.
    [[nodiscard]] uint8_t available_influence_discs() const {
        int penalty = SPECIES_TABLE[static_cast<size_t>(species_id)].starting_disk_penalty;
        int available = total_influence_discs + static_cast<int>(extra_influence_discs) - penalty -
                        static_cast<int>(disks_on_sectors) -
                        static_cast<int>(disks_on_actions);
        return available > 0 ? static_cast<uint8_t>(available) : 0;
    }
};

static_assert(static_cast<size_t>(ShipType::STARBASE) + 1 == 4, "The first 4 ShipType values must map to blueprints index 0-3");

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Player, id, score, species_id, is_ai, has_passed, disks_on_sectors, disks_on_actions, resources, colony_ships_total, colony_ships_available, orbitals, monoliths, blueprints, reputation_track, trade_rate, extra_influence_discs, graveyard_counts, eliminated, researched_techs_military, researched_techs_grid, researched_techs_nano, ambassador_tiles_held, ambassador_tiles_pending_return, traitor_held, discovery_vp_tiles_kept, parts_inventory, warp_portal_eligible);

struct UpkeepState {
    enum class Step : uint8_t {
        inactive = 0,
        colony_ships = 1,
        bankruptcy = 2,
        cleanup_graveyards = 3,
        choose_return_track = 4
    };

    Step step = Step::inactive;
    uint8_t player_id = 255;
    std::vector<PendingReturn> pending_returns;
};

NLOHMANN_JSON_SERIALIZE_ENUM(UpkeepState::Step, {
    {UpkeepState::Step::inactive, "inactive"},
    {UpkeepState::Step::colony_ships, "colony_ships"},
    {UpkeepState::Step::bankruptcy, "bankruptcy"},
    {UpkeepState::Step::cleanup_graveyards, "cleanup_graveyards"},
    {UpkeepState::Step::choose_return_track, "choose_return_track"},
});

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UpkeepState, step, player_id, pending_returns);

struct State {
    FixedVector<Player, MAX_PLAYERS> players;
    Galaxy galaxy;
    FixedVector<ReputationTiles, 40> reputation_tiles;
    FixedVector<Unit, 128> unit_registry;

    // Tech market and NPC states
    std::array<uint8_t, 40> tech_tray = {0};
    FixedVector<TechBit, 130> tech_bag;
    uint16_t sector_bag_inner = 0;
    uint16_t sector_bag_middle = 0;
    uint32_t sector_bag_outer = 0;
    NPCDifficulty gcds_difficulty = NPCDifficulty::EASY;
    NPCDifficulty guardian_difficulty = NPCDifficulty::EASY;
    NPCDifficulty ancient_difficulty = NPCDifficulty::EASY;

    // Discovery tile bag
    FixedVector<DiscoveryBit, 40> discovery_bag;

    // Turn tracking and passing queue
    uint8_t current_player = 255;
    RoundPhase current_phase = RoundPhase::ACTION;
    uint8_t current_round = 0;
    uint8_t turn_order[MAX_PLAYERS] = {255, 255, 255, 255, 255, 255};
    FixedVector<uint8_t, MAX_PLAYERS> pass_order;

    // In-flight Explore action (inactive when no Explore is being resolved).
    ExploreState explore_state;

    // In-flight Research action (inactive when no Research is being resolved).
    // ResearchState tracks multi-activation research actions (e.g., Hydran's 2 activations).
    // Unlike Explore, Research does not require a multi-step sub-state machine:
    // each activation is a single choice of a tech (and track for Rare Techs).
    ResearchState research_state;

    // In-flight Build action (inactive when no Build is being resolved).
    BuildState build_state;

    // In-flight Influence action (inactive when no Influence is being resolved).
    InfluenceState influence_state;

    // In-flight Upgrade action (inactive when no Upgrade is being resolved).
    UpgradeState upgrade_state;

    // In-flight Move action (inactive when no Move is being resolved).
    MoveState move_state;

    // In-flight Upkeep / Cleanup round-end state.
    UpkeepState upkeep_state;

    // In-flight Combat phase state.
    CombatState combat_state;

    // In-flight Diplomacy (Diplomatic Relations formation / rearrange / deferred
    // return-track choice after a break). Sub-state machine, free bonus action:
    // does not advance the turn until it returns to inactive.
    DiplomacyState diplomacy_state;

    // Monotonic counter for Unit::arrival_order. Bumped whenever a unit is
    // added to unit_registry (setup, build, explore, move, warp). 0 is the
    // "uninitialised" sentinel for legacy units lacking the field.
    uint32_t next_arrival_order = 0;

    // Allocates the next arrival_order value. Call sites wrap push_back into
    // unit_registry so participant ordering in combat reflects the rulebook's
    // "reverse entry order" rule deterministically.
    uint32_t AllocateArrivalOrder() { return ++next_arrival_order; }

    // Helper functions for tech market tray (allocation-free representation)
    uint8_t get_tech_tray_count(TechBit tech) const {
        if (tech == TechBit::NONE) return 0;
        return tech_tray[__builtin_ctzll(static_cast<uint64_t>(tech)) - 1];
    }

    void add_to_tech_tray(TechBit tech, uint8_t count = 1) {
        if (tech == TechBit::NONE) return;
        tech_tray[__builtin_ctzll(static_cast<uint64_t>(tech)) - 1] += count;
    }

    bool remove_from_tech_tray(TechBit tech, uint8_t count = 1) {
        if (tech == TechBit::NONE) return false;
        uint8_t& slot = tech_tray[__builtin_ctzll(static_cast<uint64_t>(tech)) - 1];
        if (slot >= count) {
            slot -= count;
            return true;
        }
        return false;
    }
};

inline void to_json(nlohmann::json& j, const State& s) {
    nlohmann::json tray_j = nlohmann::json::object();
    nlohmann::json build_costs_by_player = nlohmann::json::object();
    for (size_t i = 0; i < 40; ++i) {
        if (s.tech_tray[i] > 0) {
            const auto& tech = TECH_TABLE[i];
            nlohmann::json entry = nlohmann::json::object();
            entry["count"] = s.tech_tray[i];
            entry["category"] = tech.category;
            entry["order"] = i;
            entry["base_cost"] = tech.base_cost;
            entry["min_cost"] = tech.min_cost;
            entry["copies"] = tech.copies;
            tray_j[tech.name] = entry;
        }
    }
    for (const ::Player& player : s.players) {
        nlohmann::json costs = nlohmann::json::object();
        costs["Interceptor"] = open_spiel::eclipse::calculate_build_cost(player, open_spiel::eclipse::BuildType::INTERCEPTOR);
        costs["Cruiser"] = open_spiel::eclipse::calculate_build_cost(player, open_spiel::eclipse::BuildType::CRUISER);
        costs["Dreadnought"] = open_spiel::eclipse::calculate_build_cost(player, open_spiel::eclipse::BuildType::DREADNOUGHT);
        costs["Starbase"] = open_spiel::eclipse::calculate_build_cost(player, open_spiel::eclipse::BuildType::STARBASE);
        costs["Orbital"] = open_spiel::eclipse::calculate_build_cost(player, open_spiel::eclipse::BuildType::ORBITAL);
        costs["Monolith"] = open_spiel::eclipse::calculate_build_cost(player, open_spiel::eclipse::BuildType::MONOLITH);
        build_costs_by_player[std::to_string(player.id)] = costs;
    }
    nlohmann::json scores = nlohmann::json::object();
    for (const ::Player& player : s.players) {
        scores[std::to_string(player.id)] = open_spiel::eclipse::compute_player_score(s, player.id);
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
        {"pass_order", s.pass_order},
        {"explore_state", s.explore_state},
        {"research_state", s.research_state},
        {"build_state", s.build_state},
        {"build_costs_by_player", build_costs_by_player},
        {"scores", scores},
        {"influence_state", s.influence_state},
        {"upgrade_state", s.upgrade_state},
        {"move_state", s.move_state},
        {"upkeep_state", s.upkeep_state},
        {"combat_state", s.combat_state},
        {"diplomacy_state", s.diplomacy_state},
        {"next_arrival_order", s.next_arrival_order}
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
            uint8_t count = 0;
            if (el.value().is_number_integer()) {
                count = el.value().get<uint8_t>();
            } else if (el.value().is_object()) {
                count = el.value().at("count").get<uint8_t>();
            }
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

    if (j.contains("explore_state")) {
        j.at("explore_state").get_to(s.explore_state);
    } else {
        s.explore_state = ExploreState{};
    }

    j.at("research_state").get_to(s.research_state);

    if (j.contains("build_state")) {
        j.at("build_state").get_to(s.build_state);
    } else {
        s.build_state = BuildState{};
    }

    if (j.contains("influence_state")) {
        j.at("influence_state").get_to(s.influence_state);
    } else {
        s.influence_state = InfluenceState{};
    }
    if (j.contains("upkeep_state")) {
        j.at("upkeep_state").get_to(s.upkeep_state);
    } else {
        s.upkeep_state = UpkeepState{};
    }

    if (j.contains("upgrade_state")) {
        j.at("upgrade_state").get_to(s.upgrade_state);
    } else {
        s.upgrade_state = UpgradeState{};
    }

    if (j.contains("move_state")) {
        j.at("move_state").get_to(s.move_state);
    } else {
        s.move_state = MoveState{};
    }

    if (j.contains("combat_state")) {
        j.at("combat_state").get_to(s.combat_state);
    } else {
        s.combat_state = CombatState{};
    }

    if (j.contains("diplomacy_state")) {
        j.at("diplomacy_state").get_to(s.diplomacy_state);
    } else {
        s.diplomacy_state = DiplomacyState{};
    }

    if (j.contains("next_arrival_order")) {
        j.at("next_arrival_order").get_to(s.next_arrival_order);
    } else {
        s.next_arrival_order = 0;
    }
}

// Helper: Check if a sector anchors an Influence connection for a player: Control
// or any Ship present. Influence counts pinned ships too (rulebook: "a Ship or
// Control"); Explore needs an Unpinned ship instead — see is_explore_anchor().
inline bool is_sector_anchor(const State& state, uint8_t player_id, const Sector& sector) {
    if (sector.sector_id == 0) return false;
    if (sector.owner_id == player_id) return true;
    for (const Unit& unit : state.unit_registry) {
        if (unit.player_id == player_id && unit.sector_id == sector.sector_id) {
            return true;
        }
    }
    return false;
}

inline bool player_has_warp_portal_anchor(const State& state, uint8_t player_id) {
    for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
        for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
            if (!in_galaxy_bounds(q, r)) continue;
            const Sector& sector = state.galaxy.at(q, r);
            if (sector.sector_id == 0) continue;
            
            bool has_portal = sector.player_warp_portal_vp > 0;
            if (!has_portal) {
                const SectorDefinition* def = get_sector_definition(sector.sector_id);
                if (def && def->has_warp_portal) {
                    has_portal = true;
                }
            }
            if (!has_portal) continue;

            if (is_sector_anchor(state, player_id, sector)) {
                return true;
            }
        }
    }
    return false;
}

#endif //ECLIPSE_STATE_H
