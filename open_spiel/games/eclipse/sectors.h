//
// Created by Mihai on 26/05/2026.
//

#ifndef ECLIPSE_SECTORS_H
#define ECLIPSE_SECTORS_H

#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>

#include "discovery_tiles.h"

struct HexCoord {
    int8_t q, r;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(HexCoord, q, r);

struct Sector {
    uint16_t sector_id;
    uint8_t owner_id = 255;  // 255 = unowned
    HexCoord coords;
    uint8_t rotation;
    uint8_t points;
    // Population: Bits or counters for which slots are occupied
    // Since slots are fixed per tile, just track which index is filled
    uint16_t occupied_slots_mask; // Bit i is 1 if SECTOR_TABLE[id].slots[i] has a cube

    bool discovery_tile_present; // Flips to false once claimed
    DiscoveryBit discovery_tile = DiscoveryBit::NONE;
    bool orbital_built;
    bool monolith_built;
    // VP value of a player-placed Warp Portal tile on this sector.
    // 0 = none, 1 = Rare Tech portal, 2 = Ancient Warp Portal discovery tile.
    uint8_t player_warp_portal_vp = 0;
};

inline void to_json(nlohmann::json& j, const Sector& s) {
    j = nlohmann::json{
        {"sector_id", s.sector_id},
        {"owner_id", s.owner_id},
        {"coords", s.coords},
        {"rotation", s.rotation},
        {"points", s.points},
        {"occupied_slots_mask", s.occupied_slots_mask},
        {"discovery_tile_present", s.discovery_tile_present},
        {"discovery_tile", s.discovery_tile},
        {"orbital_built", s.orbital_built},
        {"monolith_built", s.monolith_built},
        {"player_warp_portal_vp", s.player_warp_portal_vp},
    };
}

inline void from_json(const nlohmann::json& j, Sector& s) {
    j.at("sector_id").get_to(s.sector_id);
    j.at("owner_id").get_to(s.owner_id);
    j.at("coords").get_to(s.coords);
    j.at("rotation").get_to(s.rotation);
    j.at("points").get_to(s.points);
    j.at("occupied_slots_mask").get_to(s.occupied_slots_mask);
    j.at("discovery_tile_present").get_to(s.discovery_tile_present);
    if (j.contains("discovery_tile")) {
        j.at("discovery_tile").get_to(s.discovery_tile);
    } else {
        s.discovery_tile = DiscoveryBit::NONE;
    }
    j.at("orbital_built").get_to(s.orbital_built);
    j.at("monolith_built").get_to(s.monolith_built);
    if (j.contains("player_warp_portal_vp")) {
        j.at("player_warp_portal_vp").get_to(s.player_warp_portal_vp);
    } else if (j.contains("has_player_warp_portal")) {
        bool has_player_warp_portal = false;
        j.at("has_player_warp_portal").get_to(has_player_warp_portal);
        s.player_warp_portal_vp = has_player_warp_portal ? 2 : 0;
    } else {
        s.player_warp_portal_vp = 0;
    }
}


enum class PlanetType : uint8_t {
    MONEY, SCIENCE, MATERIALS,
    ADV_MONEY, ADV_SCIENCE, ADV_MATERIALS,
    ANY, ADV_ANY
};

struct PlanetSlot {
    PlanetType type;
};

enum class SectorType { OUTER, INNER, MIDDLE, STARTING, CENTER, GUARDIAN, WARP };

NLOHMANN_JSON_SERIALIZE_ENUM(SectorType, {
    {SectorType::OUTER, "Outer"},
    {SectorType::INNER, "Inner"},
    {SectorType::MIDDLE, "Middle"},
    {SectorType::STARTING, "Starting"},
    {SectorType::CENTER, "Center"},
    {SectorType::GUARDIAN, "Guardian"},
    {SectorType::WARP, "Warp"}
});

struct SectorDefinition {
    const char* name;
    uint16_t sector_id;
    SectorType type;
    uint8_t points;            // VP value
    uint8_t wormholes_mask;    // Bitmask for the 6 edges
    bool has_artifact;
    bool start_with_discovery;
    std::vector<PlanetSlot> slots; // What is printed on the tile
    uint8_t starting_ancients;
    bool has_guardian;         // True for sectors 271-274
    bool is_gcds;              // True only for sector 001
    bool has_warp_portal;      // True if the sector contains an actual warp portal
};

// Index this by sector_id for O(1) lookup
static const SectorDefinition SECTOR_TABLE[] = {
    { "Galactic Center", 1, SectorType::CENTER, 4, 0b111111, true, true, {{PlanetType::ADV_MATERIALS}, {PlanetType::MATERIALS}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}, {PlanetType::ADV_MONEY}, {PlanetType::MONEY}}, 0, false, true, false },
    { "CASTOR", 101, SectorType::INNER, 2, 0b011111, false, true, {{PlanetType::ADV_MATERIALS}, {PlanetType::MATERIALS}, {PlanetType::MONEY}}, 1, false, false, false },
    { "Pollux", 102, SectorType::INNER, 2, 0b101101, false, false, {{PlanetType::SCIENCE}, {PlanetType::SCIENCE}}, 0, false, false, false },
    { "Beta Leonis", 103, SectorType::INNER, 2, 0b111011, false, false, {{PlanetType::SCIENCE}, {PlanetType::MONEY}}, 0, false, false, false },
    { "ARCTURUS", 104, SectorType::INNER, 2, 0b110110, false, true, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}}, 2, false, false, false },
    { "ZETA HERCULIS", 105, SectorType::INNER, 2, 0b110111, false, true, {{PlanetType::MONEY}, {PlanetType::SCIENCE}, {PlanetType::ADV_MATERIALS}}, 1, false, false, false },
    { "Capella", 106, SectorType::INNER, 2, 0b111100, false, false, {{PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Aldebaran", 107, SectorType::INNER, 3, 0b111101, true, false, {{PlanetType::ADV_SCIENCE}, {PlanetType::MONEY}, {PlanetType::ADV_MATERIALS}}, 0, false, false, false },
    { "Mu Cassiopeiae", 108, SectorType::INNER, 2, 0b110110, false, true, {{PlanetType::ADV_MONEY}, {PlanetType::SCIENCE}, {PlanetType::ANY}}, 1, false, false, false },
    { "Alpha Lacertae", 109, SectorType::INNER, 4, 0b011111, true, true, {{PlanetType::MATERIALS}, {PlanetType::MONEY}}, 2, false, false, false },
    { "Iota Bootis", 110, SectorType::INNER, 2, 0b101110, false, true, {{PlanetType::ADV_MONEY}, {PlanetType::ADV_ANY}}, 0, false, false, false },
    { "ALPHA CENTAURI", 201, SectorType::MIDDLE, 1, 0b010101, false, false, {{PlanetType::MATERIALS}, {PlanetType::MONEY}}, 0, false, false, false },
    { "Fomalhaut", 202, SectorType::MIDDLE, 2, 0b010101, true, false, {{PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}}, 0, false, false, false },
    { "Chi Draconis", 203, SectorType::MIDDLE, 2, 0b110101, false, true, {{PlanetType::MATERIALS}, {PlanetType::MONEY}, {PlanetType::SCIENCE}}, 2, false, false, false },
    { "Vega", 204, SectorType::MIDDLE, 2, 0b110101, true, true, {{PlanetType::ADV_MONEY}, {PlanetType::ADV_MATERIALS}, {PlanetType::ANY}}, 1, false, false, false },
    { "MU HERCULIS", 205, SectorType::MIDDLE, 1, 0b001110, true, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}}, 0, false, false, false },
    { "Epsilon Indi", 206, SectorType::MIDDLE, 1, 0b011101, false, true, {{PlanetType::MATERIALS}}, 0, false, false, false },
    { "ZETA RETICULI", 207, SectorType::MIDDLE, 2, 0b110100, false, true, {}, 0, false, false, false },
    { "Iota Persei", 208, SectorType::MIDDLE, 2, 0b101101, false, true, {}, 0, false, false, false },
    { "Delta Eridani", 209, SectorType::MIDDLE, 1, 0b110101, false, false, {{PlanetType::MONEY}, {PlanetType::SCIENCE}}, 0, false, false, false },
    { "Psi Capricorni", 210, SectorType::MIDDLE, 1, 0b100101, false, false, {{PlanetType::MONEY}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Beta Aquilae", 211, SectorType::MIDDLE, 1, 0b111100, false, true, {{PlanetType::MONEY}, {PlanetType::MATERIALS}, {PlanetType::ANY}}, 1, false, false, false },
    { "Beta Monocerotis", 214, SectorType::MIDDLE, 2, 0b111011, false, true, {{PlanetType::ADV_ANY}, {PlanetType::ADV_MATERIALS}, {PlanetType::SCIENCE}}, 1, false, false, false },
    { "Procyon", 221, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Epsilon Eridani", 222, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}}, 0, false, false, false },
    { "Altair", 223, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Beta Hydri", 224, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::ADV_MATERIALS}}, 0, false, false, false },
    { "Eta Cassiopeiae", 225, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "61 Cygni", 226, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Sirius", 227, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Sigma Draconis", 228, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::MONEY}, {PlanetType::SCIENCE}, {PlanetType::ADV_MATERIALS}}, 0, false, false, false },
    { "Tau Ceti", 229, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Lambda Aurigae", 230, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::SCIENCE}, {PlanetType::ADV_MATERIALS}}, 0, false, false, false },
    { "Delta Pavonis", 231, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Rigel", 232, SectorType::STARTING, 3, 0b110110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::SCIENCE}, {PlanetType::ADV_MATERIALS}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Omega Fornacis", 271, SectorType::GUARDIAN, 2, 0b110110, true, true, {{PlanetType::ADV_MATERIALS}, {PlanetType::MONEY}, {PlanetType::SCIENCE}}, 0, true, false, false },
    { "Sigma Hydrae", 272, SectorType::GUARDIAN, 2, 0b0110110, true, true, {{PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}, {PlanetType::MATERIALS}}, 0, true, false, false },
    { "Theta Ophiuchi", 273, SectorType::GUARDIAN, 2, 0b110110, true, true, {{PlanetType::ADV_MATERIALS}, {PlanetType::MATERIALS}, {PlanetType::MONEY}}, 0, true, false, false },
    { "Alpha Lyncis", 274, SectorType::GUARDIAN, 2, 0b110110, true, true, {{PlanetType::ADV_SCIENCE}, {PlanetType::SCIENCE}, {PlanetType::MONEY}}, 0, true, false, false },
    { "Delta Corvi", 281, SectorType::MIDDLE, 2, 0b101101, false, true, {{PlanetType::SCIENCE}, {PlanetType::MONEY}}, 0, false, false, true },
    { "ZETA DRACONIS", 301, SectorType::OUTER, 2, 0b101100, true, true, {{PlanetType::SCIENCE}, {PlanetType::MONEY}, {PlanetType::ADV_MATERIALS}}, 2, false, false, false },
    { "Gamma Serpentis", 302, SectorType::OUTER, 2, 0b100110, true, true, {{PlanetType::ADV_SCIENCE}, {PlanetType::ADV_MONEY}, {PlanetType::MATERIALS}}, 1, false, false, false },
    { "ETA CEPHEI", 303, SectorType::OUTER, 2, 0b000101, true, true, {{PlanetType::ADV_SCIENCE}, {PlanetType::ADV_MONEY}, {PlanetType::ANY}}, 1, false, false, false },
    { "THETA PEGASI", 304, SectorType::OUTER, 2, 0b100100, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "Lambda Serpentis", 305, SectorType::OUTER, 1, 0b110100, false, true, {{PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 1, false, false, false },
    { "BETA CENTAURI", 306, SectorType::OUTER, 1, 0b010100, false, false, {{PlanetType::MONEY}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "SIGMA SAGITTARII", 307, SectorType::OUTER, 2, 0b101100, false, false, {{PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}}, 0, false, false, false },
    { "Kappa Scorpii", 308, SectorType::OUTER, 2, 0b001101, false, false, {{PlanetType::SCIENCE}, {PlanetType::ADV_MATERIALS}}, 0, false, false, false },
    { "Phi Piscium", 309, SectorType::OUTER, 2, 0b100101, false, false, {{PlanetType::MONEY}, {PlanetType::ADV_SCIENCE}}, 0, false, false, false },
    { "Nu Phoenicis", 310, SectorType::OUTER, 1, 0b100100, false, false, {{PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, false },
    { "CANOPUS", 311, SectorType::OUTER, 1, 0b101100, false, true, {{PlanetType::MATERIALS}}, 0, false, false, false },
    { "Antares", 312, SectorType::OUTER, 1, 0b110100, true, true, {{PlanetType::MATERIALS}}, 0, false, false, false },
    { "Alpha Ursae Minoris", 313, SectorType::OUTER, 1, 0b100100, false, true, {{PlanetType::ANY}}, 0, false, false, false },
    { "Spica", 314, SectorType::OUTER, 1, 0b001110, false, true, {{PlanetType::ANY}}, 0, false, false, false },
    { "EPSILON AURIGAE", 315, SectorType::OUTER, 1, 0b100101, false, true, {}, 0, false, false, false },
    { "IOTA CARINAE", 316, SectorType::OUTER, 1, 0b110100, false, true, {}, 0, false, false, false },
    { "Beta Crucis", 317, SectorType::OUTER, 2, 0b000110, false, false, {{PlanetType::ADV_MONEY}, {PlanetType::MONEY}}, 0, false, false, false },
    { "GAMMA VELORUM", 318, SectorType::OUTER, 1, 0b001100, false, false, {{PlanetType::ADV_MATERIALS}, {PlanetType::ANY}}, 0, false, false, false },
    { "Beta Sextantis", 381, SectorType::OUTER, 1, 0b100110, false, true, {{PlanetType::MATERIALS}, {PlanetType::MONEY}}, 0, false, false, true },
    { "Zeta Chamaeleontis", 382, SectorType::OUTER, 1, 0b101100, false, true, {{PlanetType::SCIENCE}, {PlanetType::MATERIALS}}, 0, false, false, true },
    // TODO: provides extra actions. needs special rules to handle. maybe together with other special sectors
    // { "Geminga", 393, SectorType::OUTER, 1, 0b110100, false, false, {{PlanetType::SCIENCE}}, 0, false, false },
    // { "Simeis 147", 394, SectorType::OUTER, 1, 0b100101, false, false, {{PlanetType::MATERIALS}}, 0, false, false },
};

struct SectorLookupTable {
    uint8_t mapping[395];
};

inline const SectorDefinition* get_sector_definition(uint16_t sector_id) {
    static const SectorLookupTable lookup = []() {
        SectorLookupTable t{};
        for (int i = 0; i < 395; ++i) {
            t.mapping[i] = 255;
        }
        for (uint8_t idx = 0; idx < sizeof(SECTOR_TABLE) / sizeof(SECTOR_TABLE[0]); ++idx) {
            uint16_t id = SECTOR_TABLE[idx].sector_id;
            if (id < 395) {
                t.mapping[id] = idx;
            }
        }
        return t;
    }();

    if (sector_id == 0 || sector_id >= 395) return nullptr;
    uint8_t index = lookup.mapping[sector_id];
    if (index == 255) return nullptr;
    return &SECTOR_TABLE[index];
}

#endif //ECLIPSE_SECTORS_H
