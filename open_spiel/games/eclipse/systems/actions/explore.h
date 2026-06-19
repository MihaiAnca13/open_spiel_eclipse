//
// Created by Mihai on 28/05/2026.
//

#ifndef ECLIPSE_EXPLORE_H
#define ECLIPSE_EXPLORE_H

#include <array>
#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>

#include "../../sectors.h"  // HexCoord, Sector, SectorType

// Sub-state machine driving the multi-step Explore action. While an Explore
// action is in flight, explore_state.phase is non-inactive and the regular turn
// does not advance until all activations resolve. Defined at global scope (like
// the rest of the raw game state) so State can embed it; see state.h.
enum class ExplorePhase : uint8_t {
    inactive = 0,
    choose_zone,         // player picks an adjacent unexplored hex (zone), or stops
    draw_tile,           // chance node: flip a tile from the zone's ring bag
    draw_again_decision, // Draco may stop after the first tile or draw a second
    select_drawn_tile,   // Descendants of Draco only: keep one of two drawn tiles
    place_or_discard,    // player keeps or discards the (selected) tile
    choose_rotation,     // player picks a rotation that forms a wormhole connection
    claim_control,       // player decides whether to drop an influence disc
    discovery_reward,    // player picks discovery reward vs 2 VP
};

NLOHMANN_JSON_SERIALIZE_ENUM(ExplorePhase, {
    {ExplorePhase::inactive, "inactive"},
    {ExplorePhase::choose_zone, "choose_zone"},
    {ExplorePhase::draw_tile, "draw_tile"},
    {ExplorePhase::draw_again_decision, "draw_again_decision"},
    {ExplorePhase::select_drawn_tile, "select_drawn_tile"},
    {ExplorePhase::place_or_discard, "place_or_discard"},
    {ExplorePhase::choose_rotation, "choose_rotation"},
    {ExplorePhase::claim_control, "claim_control"},
    {ExplorePhase::discovery_reward, "discovery_reward"},
});

struct ExploreState {
    ExplorePhase phase = ExplorePhase::inactive;
    uint8_t player_id = 255;                            // exploring player
    uint8_t activations_remaining = 0;                  // includes the one in flight
    int8_t zone_q = 0, zone_r = 0;                      // chosen empty hex
    SectorType ring = SectorType::INNER;                // which ring bag to draw from
    std::array<uint16_t, 2> drawn_sector_ids = {0, 0};  // Draco may hold two
    uint8_t drawn_count = 0;
    uint16_t selected_sector_id = 0;                    // tile being placed
    uint8_t chosen_rotation = 0;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ExploreState, phase, player_id, activations_remaining, zone_q, zone_r, ring, drawn_sector_ids, drawn_count, selected_sector_id, chosen_rotation);

// The raw game state and player structs live in the global namespace (state.h);
// forward-declare them so this header does not depend on state.h (which depends
// on this one).
struct State;
struct Player;

namespace open_spiel::eclipse {

// These helpers mutate State::explore_state and the galaxy; the OpenSpiel
// wrapper (eclipse.cc) routes player/chance decisions into them and drives the
// chance draws. They follow the research.cpp pattern: free functions taking a
// ::State& and returning bool/containers, with no OpenSpiel dependency.


// True if `sector` is one the player Controls or holds at least one Unpinned Ship
// in, so it can anchor an Explore zone / wormhole connection (rulebook p.13).
bool is_explore_anchor(const ::State& state, uint8_t player_id, const ::Sector& sector);

// The current bitmask bag for a ring (INNER / MIDDLE / OUTER).
uint32_t ring_bag_value(const ::State& state, SectorType ring);

// Sector id of the bit-th tile of a ring, in SECTOR_TABLE order (matches how
// setup.cpp fills the bags). Returns 0 if out of range.
uint16_t ring_bit_to_sector_id(SectorType ring, uint8_t bit);

// Distinct empty hexes adjacent to an anchor sector, in a deterministic order
// (capped by the caller to the available zone action ids).
std::vector<::HexCoord> legal_explore_zones(const ::State& state, uint8_t player_id);

// Early-exit check: does the player have at least one legal explore zone? Used
// to gate the EXPLORE action without materializing the full zone list.
bool has_explore_zone(const ::State& state, uint8_t player_id);

// Rotations (0..5) of the selected tile that form a legal wormhole connection.
std::vector<uint8_t> legal_explore_rotations(const ::State& state, uint8_t player_id);

// Begin an Explore action: set activations from the species explore icon and
// move to choose_zone. Returns false (and stays inactive) if no legal zone.
bool begin_explore(::State& state, uint8_t player_id);

// True if the empty hex (q, r) is a legal explore zone for the player: in
// bounds, unexplored, and adjacent to a sector they Control or have a ship in.
bool is_legal_explore_zone(const ::State& state, uint8_t player_id, int q, int r);

// Fix the chosen zone (and its ring) and move to draw_tile. The zone is given as
// an absolute hex; rejected if it is not currently a legal zone.
bool choose_explore_zone(::State& state, uint8_t player_id, ::HexCoord zone);

// End the Explore action now without using remaining activations (the rules let
// Planta/Draco decide after the first sector whether to keep exploring).
void stop_exploring(::State& state);

// Resolve a single chance draw: record the drawn tile, remove it from the bag,
// and advance the phase (Draco may decide whether to draw a second tile).
void apply_explore_draw(::State& state, uint8_t ring_bit);

// Descendants of Draco, after the first tile: flip a second tile (draw_again)
// or proceed with just the one (skip_second_draw).
bool draw_again(::State& state, uint8_t player_id);
bool skip_second_draw(::State& state, uint8_t player_id);

// Descendants of Draco: keep drawn tile `tile_index`, discard the other.
bool select_drawn_tile(::State& state, uint8_t player_id, uint8_t tile_index);

// Keep the (selected) tile -> choose_rotation. Discard ends the activation.
bool place_drawn_tile(::State& state, uint8_t player_id);
bool discard_drawn_tile(::State& state, uint8_t player_id);

// Place the tile into the galaxy at the chosen zone with `rotation`, spawn
// ancients, then move to claim_control.
bool apply_explore_rotation(::State& state, uint8_t player_id, uint8_t rotation);

// Optionally take Control of the placed sector (drop an influence disc), then
// route to discovery_reward or end the activation.
bool claim_explore_control(::State& state, uint8_t player_id, bool take_control);

// Resolve the discovery tile: 2 VP, or the variable reward (TODO: stub).
bool resolve_explore_discovery(::State& state, uint8_t player_id, bool take_reward);

} // namespace open_spiel::eclipse

#endif //ECLIPSE_EXPLORE_H
