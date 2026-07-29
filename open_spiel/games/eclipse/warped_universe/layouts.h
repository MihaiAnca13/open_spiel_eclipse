//
// Created dynamically by generate_cpp_layouts.py
//

#ifndef OPEN_SPIEL_GAMES_ECLIPSE_WARPED_UNIVERSE_LAYOUTS_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_WARPED_UNIVERSE_LAYOUTS_H_

#include <cstdint>
#include <array>
#include "../sectors.h"

namespace open_spiel::eclipse {

struct LayoutCell {
    int8_t q;
    int8_t r;
    SectorType kind;
};

// Base 6-player layout (127 cells, completely unwarped)
inline constexpr std::array<LayoutCell, 127> BASE_LAYOUT_CELLS = {{
    {-6,  0, SectorType::OUTER},
    {-6,  1, SectorType::OUTER},
    {-6,  2, SectorType::OUTER},
    {-6,  3, SectorType::OUTER},
    {-6,  4, SectorType::OUTER},
    {-6,  5, SectorType::OUTER},
    {-6,  6, SectorType::OUTER},
    {-5, -1, SectorType::OUTER},
    {-5,  0, SectorType::OUTER},
    {-5,  1, SectorType::OUTER},
    {-5,  2, SectorType::OUTER},
    {-5,  3, SectorType::OUTER},
    {-5,  4, SectorType::OUTER},
    {-5,  5, SectorType::OUTER},
    {-5,  6, SectorType::OUTER},
    {-4, -2, SectorType::OUTER},
    {-4, -1, SectorType::OUTER},
    {-4,  0, SectorType::OUTER},
    {-4,  1, SectorType::OUTER},
    {-4,  2, SectorType::OUTER},
    {-4,  3, SectorType::OUTER},
    {-4,  4, SectorType::OUTER},
    {-4,  5, SectorType::OUTER},
    {-4,  6, SectorType::OUTER},
    {-3, -3, SectorType::OUTER},
    {-3, -2, SectorType::OUTER},
    {-3, -1, SectorType::OUTER},
    {-3,  0, SectorType::OUTER},
    {-3,  1, SectorType::OUTER},
    {-3,  2, SectorType::OUTER},
    {-3,  3, SectorType::OUTER},
    {-3,  4, SectorType::OUTER},
    {-3,  5, SectorType::OUTER},
    {-3,  6, SectorType::OUTER},
    {-2, -4, SectorType::OUTER},
    {-2, -3, SectorType::OUTER},
    {-2, -2, SectorType::OUTER},
    {-2, -1, SectorType::OUTER},
    {-2,  0, SectorType::STARTING},
    {-2,  1, SectorType::MIDDLE},
    {-2,  2, SectorType::STARTING},
    {-2,  3, SectorType::OUTER},
    {-2,  4, SectorType::OUTER},
    {-2,  5, SectorType::OUTER},
    {-2,  6, SectorType::OUTER},
    {-1, -5, SectorType::OUTER},
    {-1, -4, SectorType::OUTER},
    {-1, -3, SectorType::OUTER},
    {-1, -2, SectorType::OUTER},
    {-1, -1, SectorType::MIDDLE},
    {-1,  0, SectorType::INNER},
    {-1,  1, SectorType::INNER},
    {-1,  2, SectorType::MIDDLE},
    {-1,  3, SectorType::OUTER},
    {-1,  4, SectorType::OUTER},
    {-1,  5, SectorType::OUTER},
    {-1,  6, SectorType::OUTER},
    { 0, -6, SectorType::OUTER},
    { 0, -5, SectorType::OUTER},
    { 0, -4, SectorType::OUTER},
    { 0, -3, SectorType::OUTER},
    { 0, -2, SectorType::STARTING},
    { 0, -1, SectorType::INNER},
    { 0,  0, SectorType::CENTER},
    { 0,  1, SectorType::INNER},
    { 0,  2, SectorType::STARTING},
    { 0,  3, SectorType::OUTER},
    { 0,  4, SectorType::OUTER},
    { 0,  5, SectorType::OUTER},
    { 0,  6, SectorType::OUTER},
    { 1, -6, SectorType::OUTER},
    { 1, -5, SectorType::OUTER},
    { 1, -4, SectorType::OUTER},
    { 1, -3, SectorType::OUTER},
    { 1, -2, SectorType::MIDDLE},
    { 1, -1, SectorType::INNER},
    { 1,  0, SectorType::WARP},
    { 1,  1, SectorType::MIDDLE},
    { 1,  2, SectorType::OUTER},
    { 1,  3, SectorType::OUTER},
    { 1,  4, SectorType::OUTER},
    { 1,  5, SectorType::OUTER},
    { 2, -6, SectorType::OUTER},
    { 2, -5, SectorType::OUTER},
    { 2, -4, SectorType::OUTER},
    { 2, -3, SectorType::OUTER},
    { 2, -2, SectorType::STARTING},
    { 2, -1, SectorType::WARP},
    { 2,  0, SectorType::WARP},
    { 2,  1, SectorType::WARP},
    { 2,  2, SectorType::OUTER},
    { 2,  3, SectorType::OUTER},
    { 2,  4, SectorType::OUTER},
    { 3, -6, SectorType::OUTER},
    { 3, -5, SectorType::OUTER},
    { 3, -4, SectorType::OUTER},
    { 3, -3, SectorType::OUTER},
    { 3, -2, SectorType::OUTER},
    { 3, -1, SectorType::WARP},
    { 3,  0, SectorType::WARP},
    { 3,  1, SectorType::WARP},
    { 3,  2, SectorType::WARP},
    { 3,  3, SectorType::OUTER},
    { 4, -6, SectorType::OUTER},
    { 4, -5, SectorType::OUTER},
    { 4, -4, SectorType::OUTER},
    { 4, -3, SectorType::OUTER},
    { 4, -2, SectorType::WARP},
    { 4, -1, SectorType::WARP},
    { 4,  0, SectorType::WARP},
    { 4,  1, SectorType::WARP},
    { 4,  2, SectorType::WARP},
    { 5, -6, SectorType::OUTER},
    { 5, -5, SectorType::OUTER},
    { 5, -4, SectorType::OUTER},
    { 5, -3, SectorType::OUTER},
    { 5, -2, SectorType::WARP},
    { 5, -1, SectorType::WARP},
    { 5,  0, SectorType::WARP},
    { 5,  1, SectorType::OUTER},
    { 6, -6, SectorType::OUTER},
    { 6, -5, SectorType::OUTER},
    { 6, -4, SectorType::OUTER},
    { 6, -3, SectorType::WARP},
    { 6, -2, SectorType::WARP},
    { 6, -1, SectorType::OUTER},
    { 6,  0, SectorType::OUTER},
}};

// Canonical warp region (18 coordinates, corresponding to Position 1)
inline constexpr std::array<HexCoord, 18> CANONICAL_WARP_CELLS = {{
    {-3, -3},
    {-2, -4},
    {-2, -3},
    {-2, -2},
    {-1, -4},
    {-1, -3},
    {-1, -2},
    {-1, -1},
    { 0, -5},
    { 0, -4},
    { 0, -3},
    { 0, -2},
    { 0, -1},
    { 1, -5},
    { 1, -4},
    { 1, -3},
    { 2, -6},
    { 2, -5},
}};

// Canonical portal pairings within the canonical warp region frame. The warp
// region has three equivalent orientations; these are the orientation selected
// by the original pairing tool, with its gates closest to the galaxy center.
// Edge indices use HEX_DIRECTIONS.
struct CanonicalPortalPairing {
    int8_t qA, rA;
    uint8_t edgeA;
    int8_t qB, rB;
    uint8_t edgeB;
};

inline constexpr std::array<CanonicalPortalPairing, 12> CANONICAL_PORTAL_PAIRINGS = {{
    { 0, -1, 0,   0, -1, 4},
    { 0, -1, 1,  -1, -1, 5},
    { 0, -2, 0,  -1, -1, 4},
    { 1, -3, 5,  -1, -1, 3},
    { 1, -3, 0,  -1, -2, 4},
    { 1, -3, 1,  -2, -2, 5},
    { 1, -4, 0,  -2, -2, 4},
    { 2, -5, 5,  -2, -2, 3},
    { 2, -5, 0,  -2, -3, 4},
    { 2, -5, 1,  -3, -3, 5},
    { 2, -6, 0,  -3, -3, 4},
    { 2, -6, 1,  -3, -3, 3},
}};

}  // namespace open_spiel::eclipse

#endif  // OPEN_SPIEL_GAMES_ECLIPSE_WARPED_UNIVERSE_LAYOUTS_H_
