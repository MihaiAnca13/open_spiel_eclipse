#ifndef ECLIPSE_WARPED_UNIVERSE_ADJACENCY_H
#define ECLIPSE_WARPED_UNIVERSE_ADJACENCY_H

#include "../state.h"
#include <cassert>

struct AdjacencyResult {
    HexCoord to;
    int opposite_edge;
};

inline AdjacencyResult GetAdjacency(const State& state, const HexCoord& from, int direction) {
    assert(in_galaxy_bounds(from.q, from.r));
    direction = (direction + 6) % 6;

    if (state.warped_universe) {
        uint8_t cell_from = static_cast<uint8_t>(hex_to_index(from.q, from.r));
        uint16_t idx = static_cast<uint16_t>(cell_from) * 6 + direction;
        if (state.warp_link_dest_cell[idx] != 255) {
            uint8_t dest_cell = state.warp_link_dest_cell[idx];
            uint8_t dest_dir = state.warp_link_dest_dir[idx];
            return AdjacencyResult{index_to_hex(dest_cell), static_cast<int>(dest_dir)};
        }
    }

    // Standard euclidean adjacency
    return AdjacencyResult{
        HexCoord{static_cast<int8_t>(from.q + HEX_DIRECTIONS[direction].first),
                 static_cast<int8_t>(from.r + HEX_DIRECTIONS[direction].second)},
        (direction + 3) % 6
    };
}

inline bool IsExplorableSlot(const State& state, int q, int r) {
    if (!in_galaxy_bounds(q, r)) return false;
    if (state.galaxy.at(q, r).sector_id != 0) return false; // Must be empty
    if (!state.warped_universe) return true; // Standard game: any empty bounds cell is explorable!
    uint8_t kind_val = state.layout_kinds[hex_to_index(q, r)];
    SectorType kind = static_cast<SectorType>(kind_val);
    return kind == SectorType::INNER || kind == SectorType::MIDDLE || kind == SectorType::OUTER;
}

#endif // ECLIPSE_WARPED_UNIVERSE_ADJACENCY_H
