//
// Created by Mihai on 24/05/2026.
//
// Axial coordinate system for hex grid galaxy representation.
// Coordinates are packed into uint32_t keys for efficient storage and lookup.
//
// Example usage:
//   galaxy[pack_coords(0, 0)] = Sector{...};
//
//   auto neighbors = get_neighbors(0, 0);
//   for (uint32_t neighbor_key : neighbors) {
//       if (galaxy.contains(neighbor_key)) {
//           // Access neighbor sector
//       }
//   }
//
//   int dist = hex_distance(pack_coords(0, 0), pack_coords(2, -1));
//

#ifndef ECLIPSE_GALAXY_H
#define ECLIPSE_GALAXY_H

#include <cstdint>
#include <array>
#include <cmath>
#include <utility>
#include "sectors.h"

constexpr int GALAXY_RADIUS = 7;
constexpr int MAP_SIZE = (GALAXY_RADIUS * 2) + 1; // 15
constexpr int OFFSET = GALAXY_RADIUS; // 7

struct Galaxy {
    Sector grid[MAP_SIZE][MAP_SIZE] = {};

    // Helper to access using axial coords
    Sector& at(int q, int r) {
        return grid[q + OFFSET][r + OFFSET];
    }

    const Sector& at(int q, int r) const {
        return grid[q + OFFSET][r + OFFSET];
    }
};


// Axial coordinate packing/unpacking for hex grid
inline uint32_t pack_coords(int16_t q, int16_t r) {
    return (static_cast<uint32_t>(q) << 16) | static_cast<uint32_t>(r);
}

inline int16_t unpack_q(uint32_t key) {
    return static_cast<int16_t>(key >> 16);
}

inline int16_t unpack_r(uint32_t key) {
    return static_cast<int16_t>(key & 0xFFFF);
}

// Stable bijection between an axial hex coordinate and a dense cell index in
// [0, MAP_SIZE*MAP_SIZE). Used to give every galaxy hex a fixed action id.
constexpr int GALAXY_CELL_COUNT = MAP_SIZE * MAP_SIZE;

inline int hex_to_index(int q, int r) {
    return (q + OFFSET) * MAP_SIZE + (r + OFFSET);
}

inline HexCoord index_to_hex(int index) {
    return HexCoord{static_cast<int8_t>(index / MAP_SIZE - OFFSET),
                    static_cast<int8_t>(index % MAP_SIZE - OFFSET)};
}

// Hex grid neighbor directions: {dq, dr}
inline constexpr std::array<std::pair<int16_t, int16_t>, 6> HEX_DIRECTIONS = {{
    {1, 0},    // East
    {1, -1},   // Northeast
    {0, -1},   // Northwest
    {-1, 0},   // West
    {-1, 1},   // Southwest
    {0, 1}     // Southeast
}};

// Get all 6 neighbor coordinates for a hex
inline std::array<uint32_t, 6> get_neighbors(int16_t q, int16_t r) {
    return {
        pack_coords(q + 1, r),
        pack_coords(q + 1, r - 1),
        pack_coords(q, r - 1),
        pack_coords(q - 1, r),
        pack_coords(q - 1, r + 1),
        pack_coords(q, r + 1)
    };
}

// Circularly rotate a 6-edge bitmask (e.g. a sector's wormhole edges) by
// `rotation` steps, clockwise in the HEX_DIRECTIONS ordering (E,NE,NW,W,SW,SE).
inline uint8_t rotate_edge_mask(uint8_t mask, uint8_t rotation) {
    rotation %= 6;
    uint8_t lo = static_cast<uint8_t>((mask << rotation) & 0x3F);
    uint8_t hi = static_cast<uint8_t>(mask >> (6 - rotation));
    return static_cast<uint8_t>((lo | hi) & 0x3F);
}

// Hex distance formula using axial coordinates
inline int hex_distance(int16_t q1, int16_t r1, int16_t q2, int16_t r2) {
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) / 2;
}

inline int hex_distance(uint32_t key1, uint32_t key2) {
    return hex_distance(unpack_q(key1), unpack_r(key1), unpack_q(key2), unpack_r(key2));
}

#include <nlohmann/json.hpp>

inline void to_json(nlohmann::json& j, const Galaxy& g) {
    j = nlohmann::json::array();
    for (int i = 0; i < MAP_SIZE; ++i) {
        nlohmann::json row = nlohmann::json::array();
        for (int j_idx = 0; j_idx < MAP_SIZE; ++j_idx) {
            row.push_back(g.grid[i][j_idx]);
        }
        j.push_back(row);
    }
}

inline void from_json(const nlohmann::json& j, Galaxy& g) {
    for (size_t i = 0; i < MAP_SIZE; ++i) {
        for (size_t j_idx = 0; j_idx < MAP_SIZE; ++j_idx) {
            g.grid[i][j_idx] = j.at(i).at(j_idx).get<Sector>();
        }
    }
}

#endif //ECLIPSE_GALAXY_H
