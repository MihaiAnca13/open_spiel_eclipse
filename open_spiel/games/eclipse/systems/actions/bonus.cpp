//
// Bonus actions: Trade and Colony Ship placement.
//

#include "open_spiel/games/eclipse/systems/actions/bonus.h"

#include "open_spiel/games/eclipse/galaxy.h"
#include "open_spiel/games/eclipse/sectors.h"
#include "open_spiel/games/eclipse/tech.h"

// ── Population Track helpers ──────────────────────────────────────────────────

// Cubes remaining on a given track.
static uint8_t track_cubes(const Player& player, PopTrack track) {
    switch (track) {
        case PopTrack::MONEY:     return player.resources.gold_prod;
        case PopTrack::SCIENCE:   return player.resources.science_prod;
        case PopTrack::MATERIALS: return player.resources.materials_prod;
    }
    return 0;
}

// Mutable reference to a given track's cubes-on-track counter.
// Colonizing decrements this (cube leaves track → onto a planet). When the
// loss/graveyard path is built, reuse this in reverse: losing a colonized
// planet INCREMENTS the chosen track (cube returns from planet → onto a track),
// where the player likewise picks which track receives the cube.
static uint8_t& track_ref(Player& player, PopTrack track) {
    switch (track) {
        case PopTrack::SCIENCE:   return player.resources.science_prod;
        case PopTrack::MATERIALS: return player.resources.materials_prod;
        case PopTrack::MONEY:
        default:                  return player.resources.gold_prod;
    }
}

// The track a fixed-type slot must draw from, or 255 for ANY/ADV_ANY (free choice).
static uint8_t required_track(PlanetType slot_type) {
    switch (slot_type) {
        case PlanetType::MONEY:
        case PlanetType::ADV_MONEY:
            return static_cast<uint8_t>(PopTrack::MONEY);
        case PlanetType::SCIENCE:
        case PlanetType::ADV_SCIENCE:
            return static_cast<uint8_t>(PopTrack::SCIENCE);
        case PlanetType::MATERIALS:
        case PlanetType::ADV_MATERIALS:
            return static_cast<uint8_t>(PopTrack::MATERIALS);
        case PlanetType::ANY:
        case PlanetType::ADV_ANY:
        default:
            return 255; // any track
    }
}

// ── Advanced tech requirements per slot type ──────────────────────────────────

static bool has_required_tech(const Player& player, PlanetType slot_type) {
    switch (slot_type) {
        case PlanetType::ADV_MATERIALS:
            return player.has_tech(TechBit::ADVANCED_MINING);
        case PlanetType::ADV_SCIENCE:
            return player.has_tech(TechBit::ADVANCED_LABS);
        case PlanetType::ADV_MONEY:
            return player.has_tech(TechBit::ADVANCED_ECONOMY);
        case PlanetType::ADV_ANY:
            return player.has_tech(TechBit::METASYNTHESIS);
        default:
            return true; // non-advanced slots require no tech
    }
}

// ── Trade ─────────────────────────────────────────────────────────────────────

bool can_trade(const Player& player, TradeConversion conv) {
    uint8_t rate = player.trade_rate;
    switch (conv) {
        case TradeConversion::GOLD_TO_SCIENCE:
        case TradeConversion::GOLD_TO_MATERIALS:
            return player.resources.gold >= rate;
        case TradeConversion::SCIENCE_TO_GOLD:
        case TradeConversion::SCIENCE_TO_MATERIALS:
            return player.resources.science >= rate;
        case TradeConversion::MATERIALS_TO_GOLD:
        case TradeConversion::MATERIALS_TO_SCIENCE:
            return player.resources.materials >= rate;
    }
    return false;
}

bool execute_trade(State& state, uint8_t player_id, TradeConversion conv) {
    if (player_id >= state.players.size()) return false;
    Player& player = state.players[player_id];
    if (!can_trade(player, conv)) return false;

    uint8_t rate = player.trade_rate;
    switch (conv) {
        case TradeConversion::GOLD_TO_SCIENCE:
            player.resources.gold     -= rate;
            player.resources.science  += 1;
            break;
        case TradeConversion::GOLD_TO_MATERIALS:
            player.resources.gold      -= rate;
            player.resources.materials += 1;
            break;
        case TradeConversion::SCIENCE_TO_GOLD:
            player.resources.science -= rate;
            player.resources.gold    += 1;
            break;
        case TradeConversion::SCIENCE_TO_MATERIALS:
            player.resources.science   -= rate;
            player.resources.materials += 1;
            break;
        case TradeConversion::MATERIALS_TO_GOLD:
            player.resources.materials -= rate;
            player.resources.gold      += 1;
            break;
        case TradeConversion::MATERIALS_TO_SCIENCE:
            player.resources.materials -= rate;
            player.resources.science   += 1;
            break;
    }
    return true;
}

// ── Colony Ships ──────────────────────────────────────────────────────────────

bool can_use_colony_ship(const State& state, uint8_t player_id,
                         uint8_t galaxy_cell_idx, uint8_t slot_idx,
                         PopTrack track) {
    if (player_id >= state.players.size()) return false;
    const Player& player = state.players[player_id];

    if (player.colony_ships_available == 0) return false;

    HexCoord coord = index_to_hex(galaxy_cell_idx);
    const Sector& sector = state.galaxy.at(coord.q, coord.r);

    if (sector.owner_id != player_id) return false;

    const SectorDefinition* def = get_sector_definition(sector.sector_id);
    if (!def || slot_idx >= def->slots.size()) return false;

    // Slot must be empty.
    if ((sector.occupied_slots_mask >> slot_idx) & 1) return false;

    PlanetType slot_type = def->slots[slot_idx].type;

    // Advanced slots require the corresponding tech.
    if (!has_required_tech(player, slot_type)) return false;

    // Fixed-type slots accept only their matching track; ANY/ADV_ANY any track.
    uint8_t req = required_track(slot_type);
    if (req != 255 && req != static_cast<uint8_t>(track)) return false;

    // Chosen track must have a cube to place.
    if (track_cubes(player, track) == 0) return false;

    return true;
}

bool use_colony_ship(State& state, uint8_t player_id,
                     uint8_t galaxy_cell_idx, uint8_t slot_idx,
                     PopTrack track) {
    if (!can_use_colony_ship(state, player_id, galaxy_cell_idx, slot_idx, track))
        return false;

    Player& player = state.players[player_id];
    HexCoord coord = index_to_hex(galaxy_cell_idx);
    Sector& sector = state.galaxy.at(coord.q, coord.r);

    // Flip colony ship facedown.
    --player.colony_ships_available;

    // Mark slot occupied.
    sector.occupied_slots_mask |= static_cast<uint16_t>(1u << slot_idx);

    // Remove cube from the chosen track (decrement cubes-on-track counter).
    --track_ref(player, track);

    return true;
}

void refresh_colony_ships(Player& player, uint8_t count) {
    uint8_t facedown = player.colony_ships_total - player.colony_ships_available;
    uint8_t to_flip  = std::min(count, facedown);
    player.colony_ships_available += to_flip;
}

std::vector<ColonyPlacement> legal_colony_ship_placements(
    const State& state, uint8_t player_id) {
    std::vector<ColonyPlacement> result;
    if (player_id >= state.players.size()) return result;
    const Player& player = state.players[player_id];
    if (player.colony_ships_available == 0) return result;

    for (int cell = 0; cell < GALAXY_CELL_COUNT; ++cell) {
        HexCoord coord = index_to_hex(cell);
        const Sector& sector = state.galaxy.at(coord.q, coord.r);
        if (sector.sector_id == 0 || sector.owner_id != player_id) continue;

        const SectorDefinition* def = get_sector_definition(sector.sector_id);
        if (!def) continue;

        for (uint8_t s = 0; s < static_cast<uint8_t>(def->slots.size()); ++s) {
            for (uint8_t t = 0; t < POP_TRACK_COUNT; ++t) {
                PopTrack track = static_cast<PopTrack>(t);
                if (can_use_colony_ship(state, player_id,
                                        static_cast<uint8_t>(cell), s, track)) {
                    result.push_back({static_cast<uint8_t>(cell), s, track});
                }
            }
        }
    }
    return result;
}

bool can_place_warp_portal(const State& state, uint8_t player_id, uint8_t galaxy_cell_idx) {
    // TODO: Implement Warp Portal placement logic later.
    // 1. Verify player is eligible (either by claiming the Warp Portal Discovery Tile
    //    or having researched the Warp Portal Rare Tech).
    // 2. Consume the eligibility flag/tile.
    // 3. Mark the target sector as containing a player warp portal (set has_player_warp_portal = true).
    return false;
}

bool place_warp_portal(State& state, uint8_t player_id, uint8_t galaxy_cell_idx) {
    // TODO: Implement Warp Portal placement logic later.
    return false;
}
