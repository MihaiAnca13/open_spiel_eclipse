//
// Created by Mihai on 28/05/2026.
//

#include "explore.h"

#include <algorithm>
#include <bitset>

#include "../../galaxy.h"
#include "../../species.h"
#include "../../state.h"
#include "../../tech.h"

namespace open_spiel::eclipse {

namespace {

// Precomputed sector ids per ring, in SECTOR_TABLE order. Bit b of a ring's bag
// (see setup.cpp) refers to the b-th entry here, so this is an O(1) lookup that
// avoids re-scanning SECTOR_TABLE on every chance outcome.
struct RingTables {
    std::vector<uint16_t> inner, middle, outer;
    RingTables() {
        for (const SectorDefinition& def : SECTOR_TABLE) {
            switch (def.type) {
                case SectorType::INNER: inner.push_back(def.sector_id); break;
                case SectorType::MIDDLE: middle.push_back(def.sector_id); break;
                case SectorType::OUTER: outer.push_back(def.sector_id); break;
                default: break;
            }
        }
    }
};

const RingTables& ring_tables() {
    static const RingTables tables;
    return tables;
}

const std::vector<uint16_t>& ring_sector_ids(SectorType ring) {
    const RingTables& t = ring_tables();
    switch (ring) {
        case SectorType::INNER: return t.inner;
        case SectorType::MIDDLE: return t.middle;
        default: return t.outer;
    }
}

bool in_galaxy_bounds(int q, int r) {
    return q >= -GALAXY_RADIUS && q <= GALAXY_RADIUS &&
           r >= -GALAXY_RADIUS && r <= GALAXY_RADIUS;
}

bool is_draco(const State& state, uint8_t player_id) {
    return state.players[player_id].species_id == Species::DESCENDANTS_OF_DRACO;
}

// The ring a zone belongs to, by hex distance from the Galactic Center.
SectorType zone_ring(int distance) {
    if (distance <= 1) return SectorType::INNER;
    if (distance == 2) return SectorType::MIDDLE;
    return SectorType::OUTER;
}

// A zone is only explorable if its ring still has tiles left in the bag. There
// is no discard-pile reshuffle: once a ring's bag is empty, that ring can no
// longer be explored.
bool zone_ring_has_tiles(const State& state, int q, int r) {
    return ring_bag_value(state, zone_ring(hex_distance(0, 0, q, r))) != 0;
}

void clear_ring_bag_bit(State& state, SectorType ring, uint8_t bit) {
    switch (ring) {
        case SectorType::INNER:
            state.sector_bag_inner &= ~static_cast<uint16_t>(1u << bit);
            break;
        case SectorType::MIDDLE:
            state.sector_bag_middle &= ~static_cast<uint16_t>(1u << bit);
            break;
        default:
            state.sector_bag_outer &= ~(1u << bit);
            break;
    }
}

// True if an anchor neighbour of the (empty) zone presents a Wormhole on the
// edge facing it — i.e. the zone is reachable by some sector tile, so a draw
// there can actually be placed. Wormhole Generator relaxes this (a Wormhole on
// either side connects), so any anchor neighbour qualifies. This gates zone
// selection so the player is never offered an explore that is guaranteed to
// force a discard for lack of a connection.
bool zone_has_wormhole_access(const State& state, uint8_t player_id, int q, int r) {
    const bool wormhole_generator =
        state.players[player_id].has_tech(TechBit::WORMHOLE_GENERATOR);
    for (uint8_t d = 0; d < 6; ++d) {
        int nq = q + HEX_DIRECTIONS[d].first;
        int nr = r + HEX_DIRECTIONS[d].second;
        if (!in_galaxy_bounds(nq, nr)) continue;
        const Sector& nb = state.galaxy.at(nq, nr);
        if (nb.sector_id == 0 || !is_explore_anchor(state, player_id, nb)) continue;
        if (wormhole_generator) return true;
        const SectorDefinition* ndef = get_sector_definition(nb.sector_id);
        if (ndef == nullptr) continue;
        uint8_t mask = rotate_edge_mask(ndef->wormholes_mask, nb.rotation);
        // The neighbour's edge facing the zone is the opposite direction.
        if ((mask >> ((d + 3) % 6)) & 1u) return true;
    }
    return false;
}

// Enumerate empty hexes adjacent to one of the player's anchor sectors (a sector
// they Control or have a ship in). The galaxy is a dense coordinate grid with no
// owned-sector index and ships are keyed by sector_id (not coords), so we make a
// single grid pass; the ship set above keeps each cell's anchor test O(1) rather
// than re-scanning the unit registry per cell. If `first_only`, returns as soon
// as one zone is found (used by has_explore_zone to gate the action cheaply).
// Sector ids index a dense table < 395; size the ship presence set to cover it.
constexpr int kMaxSectorId = 512;

void collect_explore_zones(const State& state, uint8_t player_id, bool first_only,
                           std::vector<HexCoord>& out) {
    // O(U): mark the sector_ids the player has a ship in, so the per-cell anchor
    // test below is O(1) instead of a std::find over the registry.
    std::bitset<kMaxSectorId> ship_sector;
    for (const Unit& unit : state.unit_registry) {
        if (unit.player_id == player_id && unit.sector_id < kMaxSectorId) {
            ship_sector.set(unit.sector_id);
        }
    }
    const bool wormhole_generator =
        state.players[player_id].has_tech(TechBit::WORMHOLE_GENERATOR);
    // O(1) dedup keyed by the cell's stable index, replacing a linear rescan of
    // the growing output list per candidate.
    std::bitset<GALAXY_CELL_COUNT> listed;

    for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
        for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
            const Sector& sector = state.galaxy.at(q, r);
            if (sector.sector_id == 0) continue;  // empty cell cannot be an anchor
            const bool anchor =
                sector.owner_id == player_id ||
                (sector.sector_id < kMaxSectorId && ship_sector.test(sector.sector_id));
            if (!anchor) continue;
            // The anchor's Wormhole edges, rotated into place once per anchor. Edge
            // d faces the neighbour in direction d, so the gate below is a single
            // bit test rather than a per-zone neighbour rescan.
            const SectorDefinition* def = get_sector_definition(sector.sector_id);
            const uint8_t anchor_mask =
                def ? rotate_edge_mask(def->wormholes_mask, sector.rotation) : 0;
            for (uint8_t d = 0; d < 6; ++d) {
                const int nq = q + HEX_DIRECTIONS[d].first;
                const int nr = r + HEX_DIRECTIONS[d].second;
                if (!in_galaxy_bounds(nq, nr)) continue;
                if (state.galaxy.at(nq, nr).sector_id != 0) continue;  // not empty
                // Gate: the anchor must present a Wormhole toward the zone so a draw
                // there can connect (Wormhole Generator connects either side).
                if (!wormhole_generator && !((anchor_mask >> d) & 1u)) continue;
                if (!zone_ring_has_tiles(state, nq, nr)) continue;  // ring exhausted
                const int cell = hex_to_index(nq, nr);
                if (listed.test(cell)) continue;
                listed.set(cell);
                out.push_back(HexCoord{static_cast<int8_t>(nq), static_cast<int8_t>(nr)});
                if (first_only) return;
            }
        }
    }
}

// Finish the current activation: reset per-activation context, decrement the
// counter, and either start the next activation (if a legal zone exists) or end
// the Explore action.
void end_explore_activation(State& state) {
    ExploreState& es = state.explore_state;
    uint8_t player_id = es.player_id;
    uint8_t remaining = es.activations_remaining > 0 ? es.activations_remaining - 1 : 0;

    es = ExploreState{};
    es.player_id = player_id;
    es.activations_remaining = remaining;

    if (remaining > 0 && has_explore_zone(state, player_id)) {
        es.phase = ExplorePhase::choose_zone;
    } else {
        es.phase = ExplorePhase::inactive;
        es.player_id = 255;
    }
}

}  // namespace


bool is_explore_anchor(const State& state, uint8_t player_id, const Sector& sector) {
    if (sector.sector_id == 0) return false;
    if (sector.owner_id == player_id) return true;
    // TODO: respect Pinning once movement/combat exists; any own ship anchors.
    for (const Unit& unit : state.unit_registry) {
        if (unit.player_id == player_id && unit.sector_id == sector.sector_id) {
            return true;
        }
    }
    return false;
}

uint32_t ring_bag_value(const State& state, SectorType ring) {
    switch (ring) {
        case SectorType::INNER: return state.sector_bag_inner;
        case SectorType::MIDDLE: return state.sector_bag_middle;
        default: return state.sector_bag_outer;
    }
}

uint16_t ring_bit_to_sector_id(SectorType ring, uint8_t bit) {
    const std::vector<uint16_t>& ids = ring_sector_ids(ring);
    return bit < ids.size() ? ids[bit] : 0;
}

std::vector<HexCoord> legal_explore_zones(const State& state, uint8_t player_id) {
    std::vector<HexCoord> zones;
    collect_explore_zones(state, player_id, /*first_only=*/false, zones);
    return zones;
}

bool has_explore_zone(const State& state, uint8_t player_id) {
    std::vector<HexCoord> zones;
    collect_explore_zones(state, player_id, /*first_only=*/true, zones);
    return !zones.empty();
}

std::vector<uint8_t> legal_explore_rotations(const State& state, uint8_t player_id) {
    std::vector<uint8_t> rotations;
    const ExploreState& es = state.explore_state;
    const SectorDefinition* def = get_sector_definition(es.selected_sector_id);
    if (def == nullptr) return rotations;

    // Hoist the per-neighbour work (anchor check + rotated mask) out of the
    // rotation loop: it does not depend on our own rotation.
    struct Neighbour {
        bool anchor = false;
        uint8_t mask = 0;
    };
    std::array<Neighbour, 6> neighbours;
    for (uint8_t d = 0; d < 6; ++d) {
        int nq = es.zone_q + HEX_DIRECTIONS[d].first;
        int nr = es.zone_r + HEX_DIRECTIONS[d].second;
        if (!in_galaxy_bounds(nq, nr)) continue;
        const Sector& nb = state.galaxy.at(nq, nr);
        if (nb.sector_id == 0 || !is_explore_anchor(state, player_id, nb)) continue;
        const SectorDefinition* ndef = get_sector_definition(nb.sector_id);
        if (ndef == nullptr) continue;
        neighbours[d] = {true, rotate_edge_mask(ndef->wormholes_mask, nb.rotation)};
    }

    const bool wormhole_generator =
        state.players[player_id].has_tech(TechBit::WORMHOLE_GENERATOR);

    for (uint8_t rot = 0; rot < 6; ++rot) {
        uint8_t my_mask = rotate_edge_mask(def->wormholes_mask, rot);
        bool connects = false;
        for (uint8_t d = 0; d < 6 && !connects; ++d) {
            if (!neighbours[d].anchor) continue;
            bool my_edge = (my_mask >> d) & 1u;
            bool their_edge = (neighbours[d].mask >> ((d + 3) % 6)) & 1u;
            connects = wormhole_generator ? (my_edge || their_edge)
                                          : (my_edge && their_edge);
        }
        if (connects) rotations.push_back(rot);
    }
    return rotations;
}

bool begin_explore(State& state, uint8_t player_id) {
    if (player_id >= state.players.size()) return false;
    ExploreState& es = state.explore_state;
    es = ExploreState{};
    es.player_id = player_id;

    uint8_t activations =
        SPECIES_TABLE[static_cast<size_t>(state.players[player_id].species_id)]
            .activations.explore;
    es.activations_remaining = activations > 0 ? activations : 1;

    if (!has_explore_zone(state, player_id)) {
        es.phase = ExplorePhase::inactive;
        es.player_id = 255;
        es.activations_remaining = 0;
        return false;
    }
    es.phase = ExplorePhase::choose_zone;
    return true;
}

bool is_legal_explore_zone(const State& state, uint8_t player_id, int q, int r) {
    if (!in_galaxy_bounds(q, r)) return false;
    if (state.galaxy.at(q, r).sector_id != 0) return false;  // must be unexplored
    if (!zone_ring_has_tiles(state, q, r)) return false;     // ring exhausted
    // Must be adjacent to an anchor that presents a Wormhole toward the zone.
    return zone_has_wormhole_access(state, player_id, q, r);
}

bool choose_explore_zone(State& state, uint8_t player_id, HexCoord zone) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::choose_zone) return false;
    if (!is_legal_explore_zone(state, player_id, zone.q, zone.r)) return false;

    es.zone_q = zone.q;
    es.zone_r = zone.r;
    es.ring = zone_ring(hex_distance(0, 0, zone.q, zone.r));
    es.drawn_count = 0;
    es.drawn_sector_ids = {0, 0};
    es.phase = ExplorePhase::draw_tile;
    return true;
}

void stop_exploring(State& state) {
    state.explore_state = ExploreState{};  // phase = inactive, player_id = 255
}

void apply_explore_draw(State& state, uint8_t ring_bit) {
    ExploreState& es = state.explore_state;

    if (ring_bag_value(state, es.ring) != 0) {
        uint16_t sector_id = ring_bit_to_sector_id(es.ring, ring_bit);
        if (sector_id != 0 && es.drawn_count < 2) {
            es.drawn_sector_ids[es.drawn_count] = sector_id;
            ++es.drawn_count;
        }
        clear_ring_bag_bit(state, es.ring, ring_bit);
    }

    if (es.drawn_count == 0) {
        // Ring bag exhausted, nothing flipped: the activation has no effect.
        end_explore_activation(state);
        return;
    }
    if (es.drawn_count >= 2) {
        // Draco has now flipped both tiles; choose which one to keep.
        es.selected_sector_id = 0;
        es.phase = ExplorePhase::select_drawn_tile;
        return;
    }
    // Exactly one tile flipped. Draco may draw a second if the bag still has one.
    if (is_draco(state, es.player_id) && ring_bag_value(state, es.ring) != 0) {
        es.phase = ExplorePhase::draw_again_decision;
    } else {
        es.selected_sector_id = es.drawn_sector_ids[0];
        es.phase = ExplorePhase::place_or_discard;
    }
}

bool draw_again(State& state, uint8_t player_id) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::draw_again_decision) return false;
    es.phase = ExplorePhase::draw_tile;
    return true;
}

bool skip_second_draw(State& state, uint8_t player_id) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::draw_again_decision) return false;
    es.selected_sector_id = es.drawn_sector_ids[0];
    es.phase = ExplorePhase::place_or_discard;
    return true;
}

bool select_drawn_tile(State& state, uint8_t player_id, uint8_t tile_index) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::select_drawn_tile) return false;
    if (tile_index >= es.drawn_count) return false;
    es.selected_sector_id = es.drawn_sector_ids[tile_index];
    es.phase = ExplorePhase::place_or_discard;
    return true;
}

bool place_drawn_tile(State& state, uint8_t player_id) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::place_or_discard) return false;
    if (legal_explore_rotations(state, player_id).empty()) return false;
    es.phase = ExplorePhase::choose_rotation;
    return true;
}

bool discard_drawn_tile(State& state, uint8_t player_id) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::place_or_discard) return false;
    // Discarded tiles are gone for good (no discard pile / reshuffle): a ring's
    // bag only depletes, and an exhausted ring becomes unexplorable.
    end_explore_activation(state);
    return true;
}

bool apply_explore_rotation(State& state, uint8_t player_id, uint8_t rotation) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::choose_rotation) return false;
    const SectorDefinition* def = get_sector_definition(es.selected_sector_id);
    if (def == nullptr) return false;

    Sector& cell = state.galaxy.at(es.zone_q, es.zone_r);
    cell = Sector{};
    cell.sector_id = es.selected_sector_id;
    cell.owner_id = 255;
    cell.coords = HexCoord{es.zone_q, es.zone_r};
    cell.rotation = rotation;
    cell.points = def->points;
    cell.occupied_slots_mask = 0;
    cell.discovery_tile_present = def->start_with_discovery;
    cell.orbital_built = false;
    cell.monolith_built = false;

    for (uint8_t i = 0; i < def->starting_ancients; ++i) {
        state.unit_registry.push_back(Unit{
            .player_id = NPC_PLAYER_ID,
            .type = ShipType::ANCIENT,
            .sector_id = es.selected_sector_id,
            .damage = 0,
        });
    }

    es.chosen_rotation = rotation;
    es.phase = ExplorePhase::claim_control;
    return true;
}

bool claim_explore_control(State& state, uint8_t player_id, bool take_control) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::claim_control) return false;

    const SectorDefinition* def = get_sector_definition(es.selected_sector_id);
    const bool has_ancients = def != nullptr && def->starting_ancients > 0;
    Sector& cell = state.galaxy.at(es.zone_q, es.zone_r);
    Player& player = state.players[player_id];

    // Cannot take Control of an Ancient sector, except Descendants of Draco.
    bool may_control = !has_ancients || is_draco(state, player_id);
    if (take_control && may_control && player.available_influence_discs() > 0) {
        cell.owner_id = player_id;
        ++player.disks_on_sectors;
    }

    // Discovery is only awarded immediately from an undefended sector; Ancient
    // sectors defer the tile to the Combat Phase (not implemented yet).
    if (cell.discovery_tile_present && !has_ancients) {
        es.phase = ExplorePhase::discovery_reward;
    } else {
        end_explore_activation(state);
    }
    return true;
}

bool resolve_explore_discovery(State& state, uint8_t player_id, bool take_reward) {
    ExploreState& es = state.explore_state;
    if (es.phase != ExplorePhase::discovery_reward) return false;

    Sector& cell = state.galaxy.at(es.zone_q, es.zone_r);
    if (take_reward) {
        // TODO: apply the discovery tile's variable reward. There is no
        // discovery-tile bag yet, so the reward side is currently a no-op.
    } else {
        state.players[player_id].score += 2;
    }
    cell.discovery_tile_present = false;
    end_explore_activation(state);
    return true;
}

} // namespace open_spiel::eclipse
