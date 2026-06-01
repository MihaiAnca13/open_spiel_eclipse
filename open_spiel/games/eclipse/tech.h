//
// Created by Mihai on 25/05/2026.
//

#ifndef ECLIPSE_TECH_H
#define ECLIPSE_TECH_H
#include <cstdint>
#include <nlohmann/json.hpp>
#include "dice.h"
#include "discovery_tiles.h"

enum class TechCategory { MILITARY, GRID, NANO, RARE };

// 0 - 7 military, 8 - Y grid, Y - Z nano, Z - W rare
enum class TechBit : uint64_t {
    NONE               = 0,
    // military
    NEUTRON_BOMBS      = 1ULL << 1,
    STARBASE           = 1ULL << 2,
    PLASMA_CANNON      = 1ULL << 3,
    PHASE_SHIELD       = 1ULL << 4,
    ADVANCED_MINING    = 1ULL << 5,
    TACHYON_SOURCE     = 1ULL << 6,
    GLUON_COMPUTER     = 1ULL << 7,
    PLASMA_MISSILE     = 1ULL << 8,
    // grid
    GAUSS_SHIELD       = 1ULL << 9,
    FUSION_SOURCE      = 1ULL << 10,
    IMPROVED_HULL      = 1ULL << 11,
    POSITRON_COMPUTER  = 1ULL << 12,
    ADVANCED_ECONOMY   = 1ULL << 13,
    TACHYON_DRIVE      = 1ULL << 14,
    ANTIMATTER_CANNON  = 1ULL << 15,
    QUANTUM_GRID       = 1ULL << 16,
    // nano
    NANOROBOTS         = 1ULL << 17,
    FUSION_DRIVE       = 1ULL << 18,
    ORBITAL            = 1ULL << 19,
    ADVANCED_ROBOTICS  = 1ULL << 20,
    ADVANCED_LABS      = 1ULL << 21,
    MONOLITH           = 1ULL << 22,
    WORMHOLE_GENERATOR = 1ULL << 23,
    ARTIFACT_KEY       = 1ULL << 24,
    //rare
    ABSORPTION_SHIELD   = 1ULL << 25,
    ANCIENT_LABS        = 1ULL << 26,
    ANTIMATTER_SPLITTER = 1ULL << 27,
    CLOAKING_DEVICE     = 1ULL << 28,
    CONIFOLD_FIELD      = 1ULL << 29,
    FLUX_MISSILE        = 1ULL << 30,
    IMPROVED_LOGISTICS  = 1ULL << 31,
    METASYNTHESIS       = 1ULL << 32,
    NEUTRON_ABSORBER    = 1ULL << 33,
    PICO_MODULATOR      = 1ULL << 34,
    SENTIENT_HULL       = 1ULL << 35,
    SOLITON_CANNON      = 1ULL << 36,
    TRANSITION_DRIVE    = 1ULL << 37,
    WARP_PORTAL         = 1ULL << 38,
    ZERO_POINT_SOURCE   = 1ULL << 39,
    RIFT_CANNON         = 1ULL << 40
};

struct TechDefinition {
    TechBit bit;
    const char* name;
    TechCategory category;
    uint8_t base_cost;
    uint8_t min_cost;
    uint8_t copies;
};

static const TechDefinition TECH_TABLE[] = {
    // MILITARY TECHNOLOGIES
    { TechBit::NEUTRON_BOMBS,   "Neutron Bombs",   TechCategory::MILITARY, 2, 2, 5 },
    { TechBit::STARBASE,        "Starbase",        TechCategory::MILITARY, 4, 3, 5 },
    { TechBit::PLASMA_CANNON,   "Plasma Cannon",   TechCategory::MILITARY, 6, 4, 5 },
    { TechBit::PHASE_SHIELD,    "Phase Shield",    TechCategory::MILITARY, 8, 5, 5 },
    { TechBit::ADVANCED_MINING, "Advanced Mining", TechCategory::MILITARY, 10, 6, 4 },
    { TechBit::TACHYON_SOURCE,  "Tachyon Source",  TechCategory::MILITARY, 12, 6, 3 },
    { TechBit::GLUON_COMPUTER,  "Gluon Computer",  TechCategory::MILITARY, 14, 7, 3 },
    { TechBit::PLASMA_MISSILE,  "Plasma Missile",  TechCategory::MILITARY, 16, 8, 3 },

    // GRID TECHNOLOGIES
    { TechBit::GAUSS_SHIELD,      "Gauss Shield",      TechCategory::GRID, 2, 2, 5 },
    { TechBit::FUSION_SOURCE,     "Fusion Source",     TechCategory::GRID, 4, 3, 5 },
    { TechBit::IMPROVED_HULL,     "Improved Hull",     TechCategory::GRID, 6, 4, 5 },
    { TechBit::POSITRON_COMPUTER, "Positron Computer", TechCategory::GRID, 8, 5, 5 },
    { TechBit::ADVANCED_ECONOMY,  "Advanced Economy",  TechCategory::GRID, 10, 6, 4 },
    { TechBit::TACHYON_DRIVE,     "Tachyon Drive",     TechCategory::GRID, 12, 6, 3 },
    { TechBit::ANTIMATTER_CANNON, "Antimatter Cannon", TechCategory::GRID, 14, 7, 3 },
    { TechBit::QUANTUM_GRID,      "Quantum Grid",      TechCategory::GRID, 16, 8, 3 },

    // NANO TECHNOLOGIES
    { TechBit::NANOROBOTS,         "Nanorobots",         TechCategory::NANO, 2, 2, 5 },
    { TechBit::FUSION_DRIVE,       "Fusion Drive",       TechCategory::NANO, 4, 3, 5 },
    { TechBit::ORBITAL,            "Orbital",            TechCategory::NANO, 6, 4, 5 },
    { TechBit::ADVANCED_ROBOTICS,  "Advanced Robotics",  TechCategory::NANO, 8, 5, 5 },
    { TechBit::ADVANCED_LABS,      "Advanced Labs",      TechCategory::NANO, 10, 6, 4 },
    { TechBit::MONOLITH,           "Monolith",           TechCategory::NANO, 12, 6, 3 },
    { TechBit::WORMHOLE_GENERATOR, "Wormhole Generator", TechCategory::NANO, 14, 7, 3 },
    { TechBit::ARTIFACT_KEY,       "Artifact Key",       TechCategory::NANO, 16, 8, 3 },

    // RARE TECHNOLOGIES
    { TechBit::ABSORPTION_SHIELD,   "Absorption Shield",   TechCategory::RARE, 7, 6, 1 },
    { TechBit::ANCIENT_LABS,        "Ancient Labs",        TechCategory::RARE, 13, 9, 1 },
    { TechBit::ANTIMATTER_SPLITTER, "Antimatter Splitter", TechCategory::RARE, 5, 5, 1 },
    { TechBit::CLOAKING_DEVICE,     "Cloaking Device",     TechCategory::RARE, 7, 6, 1 },
    { TechBit::CONIFOLD_FIELD,      "Conifold Field",      TechCategory::RARE, 5, 5, 1 },
    { TechBit::FLUX_MISSILE,        "Flux Missile",        TechCategory::RARE, 11, 8, 1 },
    { TechBit::IMPROVED_LOGISTICS,  "Improved Logistics",  TechCategory::RARE, 7, 6, 1 },
    { TechBit::METASYNTHESIS,       "Metasynthesis",       TechCategory::RARE, 17, 11, 1 },
    { TechBit::NEUTRON_ABSORBER,    "Neutron Absorber",    TechCategory::RARE, 5, 5, 1 },
    { TechBit::PICO_MODULATOR,      "Pico Modulator",      TechCategory::RARE, 11, 8, 1 },
    { TechBit::SENTIENT_HULL,       "Sentient Hull",       TechCategory::RARE, 7, 6, 1 },
    { TechBit::SOLITON_CANNON,      "Soliton Cannon",      TechCategory::RARE, 9, 7, 1 },
    { TechBit::TRANSITION_DRIVE,    "Transition Drive",    TechCategory::RARE, 9, 7, 1 },
    { TechBit::WARP_PORTAL,         "Warp Portal",         TechCategory::RARE, 9, 7, 1 },
    { TechBit::ZERO_POINT_SOURCE,   "Zero Point Source",   TechCategory::RARE, 15, 10, 1 },
    { TechBit::RIFT_CANNON,         "Rift Cannon",         TechCategory::RARE, 9, 7, 1 }
};

enum class ShipPartId : uint8_t {
    NONE = 0,
    ION_CANNON,
    NUCLEAR_SOURCE,
    NUCLEAR_DRIVE,
    HULL,
    ELECTRON_COMPUTER,
    PLASMA_CANNON,
    PHASE_SHIELD,
    TACHYON_SOURCE,
    GLUON_COMPUTER,
    PLASMA_MISSILE,
    FUSION_SOURCE,
    IMPROVED_HULL,
    POSITRON_COMPUTER,
    GAUSS_SHIELD,
    TACHYON_DRIVE,
    ANTIMATTER_CANNON,
    FUSION_DRIVE,
    ABSORPTION_SHIELD,
    CONIFOLD_FIELD,
    FLUX_MISSILE,
    SENTIENT_HULL,
    SOLITON_CANNON,
    TRANSITION_DRIVE,
    ZERO_POINT_SOURCE,
    RIFT_CANNON,
    MUON_SOURCE,
    RIFT_CONDUCTOR,
    ANTIMATTER_MISSILE,
    AXION_COMPUTER,
    CONFORMAL_DRIVE,
    FLUX_SHIELD,
    HYPERGRID_SOURCE,
    INVERSION_SHIELD,
    ION_DISRUPTOR,
    ION_MISSILE,
    ION_TURRET,
    JUMP_DRIVE,
    MORPH_SHIELD,
    NONLINEAR_DRIVE,
    PLASMA_TURRET,
    SHARD_HULL,
    SOLITON_CHARGER,
    SOLITON_MISSILE
};

struct ShipPart {
    ShipPartId id;
    const char* name;
    bool is_discovery; // true if required_tech is a DiscoveryBit, false if TechBit
    bool is_missile;   // true if this is a missile weapon
    bool external;     // true if placed outside blueprint (doesn't occupy slot)
    union {
        TechBit tech;
        DiscoveryBit discovery;
    } required_tech;
    DieColor die_color;      // YELLOW, ORANGE, BLUE, RED, PURPLE, or NONE
    uint8_t die_amount;      // Number of dice (for cannons/missiles)
    int8_t added_computer;  // To-hit bonus
    int8_t added_shield;    // Evasion penalty for opponents
    int8_t added_hull;      // HP/Absorb value
    int8_t net_energy;    // Production or consumption
    int8_t net_initiative;// Initiative bonus or impairment
    int8_t added_movement;  // Distance for MOVE actions
};

static const ShipPart SHIP_PART_TABLE[] = {
    // Basic Parts (No Tech Required)
    { ShipPartId::ION_CANNON,        "Ion Cannon",     false, false, false, { .tech = TechBit::NONE }, DieColor::YELLOW, 1, 0, 0, 0, -1, 0, 0 },
    { ShipPartId::NUCLEAR_SOURCE,    "Nuclear Source", false, false, false, { .tech = TechBit::NONE }, DieColor::NONE, 0, 0, 0, 0, 3,  0, 0 },
    { ShipPartId::NUCLEAR_DRIVE,     "Nuclear Drive",  false, false, false, { .tech = TechBit::NONE }, DieColor::NONE, 0, 0, 0, 0, -1, 1, 1 },
    { ShipPartId::HULL,              "Hull",           false, false, false, { .tech = TechBit::NONE }, DieColor::NONE, 0, 0, 0, 1, 0,  0, 0 },
    { ShipPartId::ELECTRON_COMPUTER, "Electron Computer", false, false, false, { .tech = TechBit::NONE }, DieColor::NONE, 0, 1, 0, 0, 0,  0, 0 },

    // Military Tech Parts
    { ShipPartId::PLASMA_CANNON,     "Plasma Cannon",  false, false, false, { .tech = TechBit::PLASMA_CANNON }, DieColor::ORANGE, 1, 0, 0, 0, -2, 0, 0 },
    { ShipPartId::PHASE_SHIELD,      "Phase Shield",   false, false, false, { .tech = TechBit::PHASE_SHIELD },  DieColor::NONE, 0, 0, 2, 0, -1,  0, 0 },
    { ShipPartId::TACHYON_SOURCE,    "Tachyon Source", false, false, false, { .tech = TechBit::TACHYON_SOURCE }, DieColor::NONE, 0, 0, 0, 0, 9,  0, 0 },
    { ShipPartId::GLUON_COMPUTER,    "Gluon Computer", false, false, false, { .tech = TechBit::GLUON_COMPUTER }, DieColor::NONE, 0, 2, 0, 0, -1,  0, 0 },
    { ShipPartId::PLASMA_MISSILE,    "Plasma Missile", false, true,  false, { .tech = TechBit::PLASMA_MISSILE }, DieColor::ORANGE, 2, 0, 0, 0, -1,  0, 0 },

    // Grid Tech Parts
    { ShipPartId::FUSION_SOURCE,     "Fusion Source",  false, false, false, { .tech = TechBit::FUSION_SOURCE }, DieColor::NONE, 0, 0, 0, 0, 6,  0, 0 },
    { ShipPartId::IMPROVED_HULL,     "Improved Hull",  false, false, false, { .tech = TechBit::IMPROVED_HULL }, DieColor::NONE, 0, 0, 0, 2, 0,  -1, 0 },
    { ShipPartId::POSITRON_COMPUTER, "Positron Computer", false, false, false, { .tech = TechBit::POSITRON_COMPUTER }, DieColor::NONE, 0, 2, 0, 0, -1,  0, 0 },
    { ShipPartId::GAUSS_SHIELD,      "Gauss Shield",   false, false, false, { .tech = TechBit::GAUSS_SHIELD },   DieColor::NONE, 0, 0, 1, 0, 0,  0, 0 },
    { ShipPartId::TACHYON_DRIVE,     "Tachyon Drive",  false, false, false, { .tech = TechBit::TACHYON_DRIVE },  DieColor::NONE, 0, 0, 0, 0, -3,  3, 3 },
    { ShipPartId::ANTIMATTER_CANNON, "Antimatter Cannon", false, false, false, { .tech = TechBit::ANTIMATTER_CANNON }, DieColor::RED, 1, 0, 0, 0, -4,  0, 0 },

    // Nano Tech Parts
    { ShipPartId::FUSION_DRIVE,      "Fusion Drive",   false, false, false, { .tech = TechBit::FUSION_DRIVE },   DieColor::NONE, 0, 0, 0, 0, -2,  2, 2 },

    // Rare Tech Parts
    { ShipPartId::ABSORPTION_SHIELD, "Absorption Shield", false, false, false, { .tech = TechBit::ABSORPTION_SHIELD }, DieColor::NONE, 0, 0, 1, 0, 4,  0, 0 },
    { ShipPartId::CONIFOLD_FIELD,    "Conifold Field", false, false, false, { .tech = TechBit::CONIFOLD_FIELD }, DieColor::NONE, 0, 0, 0, 3, -2, 0, 0 },
    { ShipPartId::FLUX_MISSILE,      "Flux Missile",   false, true,  false, { .tech = TechBit::FLUX_MISSILE },   DieColor::YELLOW, 2, 0, 0, 0, 0,  1, 0 },
    { ShipPartId::SENTIENT_HULL,     "Sentient Hull",  false, false, false, { .tech = TechBit::SENTIENT_HULL },  DieColor::NONE, 0, 1, 0, 1, 0,  0, 0 },
    { ShipPartId::SOLITON_CANNON,    "Soliton Cannon", false, false, false, { .tech = TechBit::SOLITON_CANNON }, DieColor::BLUE,   1, 1, 0, 0, -3, 0, 0 },
    { ShipPartId::TRANSITION_DRIVE,  "Transition Drive", false, false, false, { .tech = TechBit::TRANSITION_DRIVE }, DieColor::NONE, 0, 0, 0, 0, 0,  0, 3 },
    { ShipPartId::ZERO_POINT_SOURCE, "Zero Point Source", false, false, false, { .tech = TechBit::ZERO_POINT_SOURCE }, DieColor::NONE, 0, 0, 0, 0, 12,  0, 0 },
    { ShipPartId::RIFT_CANNON,       "Rift Cannon",    false, false, false, { .tech = TechBit::RIFT_CANNON }, DieColor::PURPLE, 1, 0, 0, 0, -2,  0, 0 },

    // Discovery (Ancient) Parts
    // Note: is_discovery = true for parts found on discovery tiles
    { ShipPartId::MUON_SOURCE,       "Muon Source",    true,  false, true,  { .discovery = DiscoveryBit::MUON_SOURCE }, DieColor::NONE, 0, 0, 0, 0, 2,  0, 0 },
    { ShipPartId::RIFT_CONDUCTOR,    "Rift Conductor", true,  false, false, { .discovery = DiscoveryBit::PART_RIFT_CONDUCTOR }, DieColor::PURPLE, 1, 0, 0, 1, -1,  0, 0 },
    { ShipPartId::ANTIMATTER_MISSILE, "Antimatter Missile", true, true,  false, { .discovery = DiscoveryBit::PART_ANTIMATTER_MISSILE }, DieColor::RED, 1, 0, 0, 0, 0,  0, 0 },
    { ShipPartId::AXION_COMPUTER,    "Axion Computer", true,  false, false, { .discovery = DiscoveryBit::PART_AXION_COMPUTER }, DieColor::NONE, 0, 2, 0, 0, 0,  0, 0 },
    { ShipPartId::CONFORMAL_DRIVE,   "Conformal Drive", true,  false, false, { .discovery = DiscoveryBit::PART_CONFORMAL_DRIVE }, DieColor::NONE, 0, 0, 0, 0, -2,  2, 4 },
    { ShipPartId::FLUX_SHIELD,       "Flux Shield",    true,  false, false, { .discovery = DiscoveryBit::PART_FLUX_SHIELD }, DieColor::NONE, 0, 0, 3, 0, -2,  1, 0 },
    { ShipPartId::HYPERGRID_SOURCE,  "Hypergrid Source", true,  false, false, { .discovery = DiscoveryBit::PART_HYPERGRID_SOURCE }, DieColor::NONE, 0, 0, 0, 0, 11,  0, 0 },
    { ShipPartId::INVERSION_SHIELD,  "Inversion Shield", true,  false, false, { .discovery = DiscoveryBit::PART_INVERSION_SHIELD }, DieColor::NONE, 0, 0, 2, 0, 2,  0, 0 },
    { ShipPartId::ION_DISRUPTOR,     "Ion Disruptor",  true,  false, false, { .discovery = DiscoveryBit::PART_ION_DISRUPTOR }, DieColor::YELLOW, 1, 0, 0, 0, 0,  3, 0 },
    { ShipPartId::ION_MISSILE,       "Ion Missile",    true,  true,  false, { .discovery = DiscoveryBit::PART_ION_MISSILE }, DieColor::YELLOW, 3, 0, 0, 0, 0,  0, 0 },
    { ShipPartId::ION_TURRET,        "Ion Turret",     true,  false, false, { .discovery = DiscoveryBit::PART_ION_TURRET }, DieColor::YELLOW, 2, 0, 0, 0, 0,  0, 0 },
    { ShipPartId::JUMP_DRIVE,        "Jump Drive",     true,  false, false, { .discovery = DiscoveryBit::PART_JUMP_DRIVE }, DieColor::NONE, 0, 0, 0, 0, -2,  0, 0 }, //tp to adjacent sector regardless of wormhole
    { ShipPartId::MORPH_SHIELD,      "Morph Shield",   true,  false, false, { .discovery = DiscoveryBit::PART_MORPH_SHIELD }, DieColor::NONE, 0, 0, 1, 0, 0,  1, 0 }, //heals 1 per combat
    { ShipPartId::NONLINEAR_DRIVE,   "Nonlinear Drive", true,  false, false, { .discovery = DiscoveryBit::PART_NONLINEAR_DRIVE }, DieColor::NONE, 0, 0, 0, 0, 2,  0, 2 },
    { ShipPartId::PLASMA_TURRET,     "Plasma Turret",  true,  false, false, { .discovery = DiscoveryBit::PART_PLASMA_TURRET }, DieColor::ORANGE, 2, 0, 0, 0, -3,  0, 0 },
    { ShipPartId::SHARD_HULL,        "Shard Hull",     true,  false, false, { .discovery = DiscoveryBit::PART_SHARD_HULL }, DieColor::NONE, 0, 0, 0, 3, 0,  0, 0 },
    { ShipPartId::SOLITON_CHARGER,   "Soliton Charger", true,  false, false, { .discovery = DiscoveryBit::PART_SOLITON_CHARGER }, DieColor::BLUE, 1, 0, 0, 0, -1,  0, 0 },
    { ShipPartId::SOLITON_MISSILE,   "Soliton Missile", true,  true,  false, { .discovery = DiscoveryBit::PART_SOLITON_MISSILE }, DieColor::BLUE, 1, 0, 0, 0, 0,  0, 0 }
};

struct ShipStats {
    int8_t initiative = 0;
    int8_t computer = 0;
    int8_t shield = 0;
    int16_t energy_net = 0;
    uint8_t hull = 0;
    uint8_t movement = 0;
    // Dice: {Yellow, Orange, Blue, Red, Purple}
    uint8_t cannons[5] = {0, 0, 0, 0, 0};
    uint8_t missiles[5] = {0, 0, 0, 0, 0};
};

struct Blueprint {
    // Array of Module IDs (ShipPartId::NONE = empty)
    ShipPartId slots[8] = {
        ShipPartId::NONE, ShipPartId::NONE, ShipPartId::NONE, ShipPartId::NONE,
        ShipPartId::NONE, ShipPartId::NONE, ShipPartId::NONE, ShipPartId::NONE
    };
    uint8_t capacity = 0; // Number of slots in the grid (e.g. 4 for Interceptor, 1 less for Planta)

    // Pre-printed stats outside the grid (varies by species and ship type)
    ShipStats base_stats;

    // Cached stats for O(1) combat math
    ShipStats total_stats;

    void recompute() {
        total_stats = base_stats;
        for (uint8_t i = 0; i < capacity; ++i) {
            ShipPartId part_id = slots[i];
            if (part_id == ShipPartId::NONE) continue;
            
            size_t idx = static_cast<size_t>(part_id) - 1;
            // Ensure index is valid
            if (idx >= sizeof(SHIP_PART_TABLE) / sizeof(SHIP_PART_TABLE[0])) continue;
            
            const ShipPart& part = SHIP_PART_TABLE[idx];
            
            // Double check that the table maps IDs correctly
            if (part.id != part_id) continue;

            total_stats.initiative += part.net_initiative;
            total_stats.computer += part.added_computer;
            total_stats.shield += part.added_shield;
            total_stats.energy_net += part.net_energy;
            total_stats.hull += part.added_hull;
            total_stats.movement += part.added_movement;
            
            if (part.die_color != DieColor::NONE) {
                uint8_t color_idx = static_cast<uint8_t>(part.die_color);
                if (color_idx < 5) {
                    if (part.is_missile) {
                        total_stats.missiles[color_idx] += part.die_amount;
                    } else {
                        total_stats.cannons[color_idx] += part.die_amount;
                    }
                }
            }
        }
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ShipStats, initiative, computer, shield, energy_net, hull, movement, cannons, missiles);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Blueprint, slots, capacity, base_stats, total_stats);

inline void to_json(nlohmann::json& j, const TechBit& t) {
    j = static_cast<uint64_t>(t);
}
inline void from_json(const nlohmann::json& j, TechBit& t) {
    t = static_cast<TechBit>(j.get<uint64_t>());
}

#endif //ECLIPSE_TECH_H
