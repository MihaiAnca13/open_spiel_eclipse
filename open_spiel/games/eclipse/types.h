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

enum ReputationTiles { ONE, TWO, THREE, FOUR, NONE };

NLOHMANN_JSON_SERIALIZE_ENUM(ReputationTiles, {
    {ReputationTiles::ONE, "One"},
    {ReputationTiles::TWO, "Two"},
    {ReputationTiles::THREE, "Three"},
    {ReputationTiles::FOUR, "Four"},
    {ReputationTiles::NONE, "None"}
});

// Reputation Track slot class (rulebook p.14):
//   - 2 slots: "May hold either an Ambassador or a Reputation Tile"
//   - 1 slot:  "May hold only an Ambassador Tile"
//   - 2 slots: "May hold only a Reputation Tile"
enum class ReputationSlotKind : uint8_t {
    AMBASSADOR_OR_REP = 0,
    AMBASSADOR_ONLY   = 1,
    REP_ONLY          = 2,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ReputationSlotKind, {
    {ReputationSlotKind::AMBASSADOR_OR_REP, "ambassador_or_rep"},
    {ReputationSlotKind::AMBASSADOR_ONLY,   "ambassador_only"},
    {ReputationSlotKind::REP_ONLY,          "rep_only"},
});

// One physical slot on the Reputation Track (rulebook p.14).
// `kind` is the fixed slot class; `holds_ambassador` selects which payload
// fields are valid. `pending_track_choice` is set after a Diplomatic-Relation
// break while the player's choice of Pop Track for the returned cube is
// still outstanding.
struct ReputationSlot {
    ReputationSlotKind kind = ReputationSlotKind::AMBASSADOR_OR_REP;
    bool holds_ambassador = false;
    ReputationTiles rep_value = ReputationTiles::NONE;
    uint8_t ambassador_from = 255;       // player_id of the giver; valid only if holds_ambassador
    bool pending_track_choice = false;   // Pop Cube returned from a break awaiting track choice
};

inline void to_json(nlohmann::json& j, const ReputationSlot& s) {
    j = nlohmann::json{
        {"kind", s.kind},
        {"holds_ambassador", s.holds_ambassador},
        {"rep_value", s.rep_value},
        {"ambassador_from", s.ambassador_from},
        {"pending_track_choice", s.pending_track_choice},
    };
}

inline void from_json(const nlohmann::json& j, ReputationSlot& s) {
    s = ReputationSlot{};
    if (j.contains("kind")) j.at("kind").get_to(s.kind);
    if (j.contains("holds_ambassador")) j.at("holds_ambassador").get_to(s.holds_ambassador);
    if (j.contains("rep_value")) j.at("rep_value").get_to(s.rep_value);
    if (j.contains("ambassador_from")) j.at("ambassador_from").get_to(s.ambassador_from);
    if (j.contains("pending_track_choice")) j.at("pending_track_choice").get_to(s.pending_track_choice);
}

} // namespace open_spiel::eclipse

#endif // ECLIPSE_TYPES_H
