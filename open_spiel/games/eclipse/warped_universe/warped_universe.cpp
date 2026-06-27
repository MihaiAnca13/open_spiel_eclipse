#include "warped_universe.h"
#include "../state.h"
#include "layouts.h"

namespace open_spiel::eclipse {

void RebuildWarpLinks(::State& state, int players_count_override) {
    // 1. Initialize layout kinds from BASE_LAYOUT_CELLS
    for (const auto& cell : BASE_LAYOUT_CELLS) {
        state.layout_kinds[hex_to_index(cell.q, cell.r)] = static_cast<uint8_t>(cell.kind);
    }

    // 2. Reset warp links to 255 (meaning no link)
    state.warp_link_dest_cell.fill(255);
    state.warp_link_dest_dir.fill(255);

    if (!state.warped_universe) {
        return;
    }

    uint8_t players_count = players_count_override > 0 ? players_count_override : state.players.size();
    if (players_count < 3 || players_count > 5) {
        state.warped_universe = false;
        return;
    }

    // Helper lambda to check if a player position is active
    auto is_pos_active = [](uint8_t p_count, uint8_t pos) -> bool {
        if (p_count == 6) return true;
        if (p_count == 5) return pos < 5;
        if (p_count == 4) return pos == 0 || pos == 1 || pos == 3 || pos == 4;
        if (p_count == 3) return pos == 0 || pos == 2 || pos == 4;
        return pos == 0 || pos == 3;
    };

    // Helper to rotate CW
    auto rotate_cw_helper = [](int8_t q, int8_t r, uint8_t steps) -> HexCoord {
        steps %= 6;
        int8_t nq = q;
        int8_t nr = r;
        for (uint8_t i = 0; i < steps; ++i) {
            int8_t next_q = nq + nr;
            int8_t next_r = -nq;
            nq = next_q;
            nr = next_r;
        }
        return HexCoord{nq, nr};
    };

    for (uint8_t pos = 0; pos < 6; ++pos) {
        if (!is_pos_active(players_count, pos)) {
            // Player is missing, so place a warp region!
            uint8_t steps = (pos + 5) % 6;

            // Mark warp cells on layout kinds
            for (const auto& canonical_coord : CANONICAL_WARP_CELLS) {
                HexCoord rotated = rotate_cw_helper(canonical_coord.q, canonical_coord.r, steps);
                state.layout_kinds[hex_to_index(rotated.q, rotated.r)] = static_cast<uint8_t>(SectorType::WARP);
            }

            // Populate active warp portal linkages
            for (const auto& pair : CANONICAL_PORTAL_PAIRINGS) {
                // Rotate warp cell A and exit edge A
                HexCoord rotated_warp_A = rotate_cw_helper(pair.qA, pair.rA, steps);
                uint8_t edgeA = (pair.edgeA + steps) % 6;

                // Sector A coordinate (neighbor of warp cell A in direction edgeA)
                int nqA = rotated_warp_A.q + HEX_DIRECTIONS[edgeA].first;
                int nrA = rotated_warp_A.r + HEX_DIRECTIONS[edgeA].second;
                uint8_t edgeA_opposite = (edgeA + 3) % 6;

                // Rotate warp cell B and exit edge B
                HexCoord rotated_warp_B = rotate_cw_helper(pair.qB, pair.rB, steps);
                uint8_t edgeB = (pair.edgeB + steps) % 6;

                // Sector B coordinate (neighbor of warp cell B in direction edgeB)
                int nqB = rotated_warp_B.q + HEX_DIRECTIONS[edgeB].first;
                int nrB = rotated_warp_B.r + HEX_DIRECTIONS[edgeB].second;
                uint8_t edgeB_opposite = (edgeB + 3) % 6;

                if (in_galaxy_bounds(nqA, nrA) && in_galaxy_bounds(nqB, nrB)) {
                    uint8_t cellA = static_cast<uint8_t>(hex_to_index(nqA, nrA));
                    uint8_t cellB = static_cast<uint8_t>(hex_to_index(nqB, nrB));

                    uint16_t idxA = static_cast<uint16_t>(cellA) * 6 + edgeA_opposite;
                    uint16_t idxB = static_cast<uint16_t>(cellB) * 6 + edgeB_opposite;

                    state.warp_link_dest_cell[idxA] = cellB;
                    state.warp_link_dest_dir[idxA] = edgeB_opposite;

                    state.warp_link_dest_cell[idxB] = cellA;
                    state.warp_link_dest_dir[idxB] = edgeA_opposite;
                }
            }
        }
    }
}

}  // namespace open_spiel::eclipse
