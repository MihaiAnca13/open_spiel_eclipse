//
// Created by Mihai on 26/05/2026.
//

#ifndef ECLIPSE_DISCOVERY_TILES_H
#define ECLIPSE_DISCOVERY_TILES_H
#include <cstdint>

enum class DiscoveryBit : uint64_t {
    NONE = 0,
    // ancient
    ANCIENT_MONOLITH = 1ULL << 1,
    ANCIENT_ORBITAL = 1ULL << 2,
    ANCIENT_TECH = 1ULL << 3,
    ANCIENT_CRUISER = 1ULL << 4,
    // muon
    MUON_SOURCE = 1ULL << 5,
    // parts
    PART_ANTIMATTER_MISSILE = 1ULL << 6,
    PART_AXION_COMPUTER = 1ULL << 7,
    PART_CONFORMAL_DRIVE = 1ULL << 8,
    PART_FLUX_SHIELD = 1ULL << 9,
    PART_HYPERGRID_SOURCE = 1ULL << 10,
    PART_INVERSION_SHIELD = 1ULL << 11,
    PART_ION_DISRUPTOR = 1ULL << 12,
    PART_ION_MISSILE = 1ULL << 13,
    PART_ION_TURRET = 1ULL << 14,
    PART_JUMP_DRIVE = 1ULL << 15,
    PART_MORPH_SHIELD = 1ULL << 16,
    PART_NONLINEAR_DRIVE = 1ULL << 17,
    PART_PLASMA_TURRET = 1ULL << 18,
    PART_SHARD_HULL = 1ULL << 19,
    PART_SOLITON_CHARGER = 1ULL << 20,
    PART_SOLITON_MISSILE = 1ULL << 21,
    PART_RIFT_CONDUCTOR = 1ULL << 22,
    // resources
    RESOURCE_SCIENCE_3_MONEY_3 = 1ULL << 23,
    RESOURCES_2MAT_2S_3MONEY = 1ULL << 24,
    RESOURCES_6_MATERIALS = 1ULL << 25,
    RESOURCES_5_SCIENCE = 1ULL << 26,
    RESOURCES_8_MONEY = 1ULL << 27,
    // variable vp
    VP_PER_3REP = 1ULL << 28,
    VP_PER_ARTIFACT = 1ULL << 29,
    // warp
    WARP_PORTAL = 1ULL << 30
};

struct DiscoveryTileDefinition {
    DiscoveryBit discovery;
    const char* name;
    uint8_t copies;
};

static const DiscoveryTileDefinition DISCOVERY_TILE_TABLE[] = {
    // ancient
    { DiscoveryBit::ANCIENT_MONOLITH, "Ancient Monolith", 1 },
    { DiscoveryBit::ANCIENT_ORBITAL, "Ancient Orbital", 2 },
    { DiscoveryBit::ANCIENT_TECH, "Ancient Tech", 3 },
    { DiscoveryBit::ANCIENT_CRUISER, "Ancient Cruiser", 3 },
    // muon
    { DiscoveryBit::MUON_SOURCE, "Muon Source", 1 },
    // parts
    { DiscoveryBit::PART_ANTIMATTER_MISSILE, "Antimatter Missile Part", 1 },
    { DiscoveryBit::PART_AXION_COMPUTER, "Axion Computer Part", 1 },
    { DiscoveryBit::PART_CONFORMAL_DRIVE, "Conformal Drive Part", 1 },
    { DiscoveryBit::PART_FLUX_SHIELD, "Flux Shield Part", 1 },
    { DiscoveryBit::PART_HYPERGRID_SOURCE, "Hypergrid Source Part", 1 },
    { DiscoveryBit::PART_INVERSION_SHIELD, "Inversion Shield Part", 1 },
    { DiscoveryBit::PART_ION_DISRUPTOR, "Ion Disruptor Part", 1 },
    { DiscoveryBit::PART_ION_MISSILE, "Ion Missile Part", 1 },
    { DiscoveryBit::PART_ION_TURRET, "Ion Turret Part", 1 },
    { DiscoveryBit::PART_JUMP_DRIVE, "Jump Drive Part", 1 },
    { DiscoveryBit::PART_MORPH_SHIELD, "Morph Shield Part", 1 },
    { DiscoveryBit::PART_NONLINEAR_DRIVE, "Nonlinear Drive Part", 1 },
    { DiscoveryBit::PART_PLASMA_TURRET, "Plasma Turret Part", 1 },
    { DiscoveryBit::PART_SHARD_HULL, "Shard Hull Part", 1 },
    { DiscoveryBit::PART_SOLITON_CHARGER, "Soliton Charger Part", 1 },
    { DiscoveryBit::PART_SOLITON_MISSILE, "Soliton Missile Part", 1 },
    { DiscoveryBit::PART_RIFT_CONDUCTOR, "Rift Conductor Part", 1 },
    // resources
    { DiscoveryBit::RESOURCE_SCIENCE_3_MONEY_3, "Science 3 Money 3", 2 },
    { DiscoveryBit::RESOURCES_2MAT_2S_3MONEY, "Resources 2Mat 2S 3Mon", 2 },
    { DiscoveryBit::RESOURCES_6_MATERIALS, "Resources 3 Materials", 3 },
    { DiscoveryBit::RESOURCES_5_SCIENCE, "Resources 5 Science", 3 },
    { DiscoveryBit::RESOURCES_8_MONEY, "Resources 8 Money", 3 },
    // variable vp
    { DiscoveryBit::VP_PER_3REP, "1 VP per 3 VP from Reputation tiles", 1 },
    { DiscoveryBit::VP_PER_ARTIFACT, "1 VP per Artifact", 1 },
    // warp
    { DiscoveryBit::WARP_PORTAL, "Warp Portal", 1 }
};

#endif //ECLIPSE_DISCOVERY_TILES_H
