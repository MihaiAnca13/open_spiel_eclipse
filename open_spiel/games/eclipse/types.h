//
// Shared type definitions for Eclipse game state
//

#ifndef ECLIPSE_TYPES_H
#define ECLIPSE_TYPES_H

#include <cstdint>
#include <nlohmann/json.hpp>

namespace open_spiel::eclipse {

enum class ShipType { INTERCEPTOR, CRUISER, DREADNOUGHT, STARBASE, ANCIENT, GUARDIAN, GCDS };

NLOHMANN_JSON_SERIALIZE_ENUM(ShipType, {
    {ShipType::INTERCEPTOR, "Interceptor"},
    {ShipType::CRUISER, "Cruiser"},
    {ShipType::DREADNOUGHT, "Dreadnought"},
    {ShipType::STARBASE, "Starbase"},
    {ShipType::ANCIENT, "Ancient"},
    {ShipType::GUARDIAN, "Guardian"},
    {ShipType::GCDS, "GCDS"}
});

enum ReputationTiles { ONE, TWO, THREE, FOUR };

NLOHMANN_JSON_SERIALIZE_ENUM(ReputationTiles, {
    {ReputationTiles::ONE, "One"},
    {ReputationTiles::TWO, "Two"},
    {ReputationTiles::THREE, "Three"},
    {ReputationTiles::FOUR, "Four"}
});

} // namespace open_spiel::eclipse

#endif // ECLIPSE_TYPES_H
