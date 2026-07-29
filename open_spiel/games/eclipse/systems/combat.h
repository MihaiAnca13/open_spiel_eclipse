//
// Combat Phase state machine for step-driven MCTS/RL tracking.
//

#ifndef ECLIPSE_COMBAT_H
#define ECLIPSE_COMBAT_H

#include <cstdint>
#include <cstddef>
#include <random>
#include <array>
#include <vector>

#include "../types.h"
#include "../sectors.h"
#include "../npc.h"
#include "../dice.h"

// Forward declaration - State is in global namespace
struct State;

namespace open_spiel::eclipse {

// Maximum sizes chosen to cover the worst-case galaxy and a generous combat
// queue. The galaxy has 225 cells, so the maximum number of simultaneous
// battles is bounded by that; in practice <5. Fixed capacities keep the
// combat struct cache-friendly for the MCTS/RL hot path.
constexpr uint8_t  kMaxCombatBattles = 8;
constexpr uint8_t  kMaxInitiativeGroups = 16; // <= 4 ship types * 4 participants
constexpr uint8_t  kMaxParticipantsPerBattle = 6;
constexpr uint8_t  kMaxReputationDraw = 5;
constexpr uint8_t  kMaxDestroyedPerBattle = 32;
constexpr uint8_t  kMaxGalaxyCells = 225;
constexpr uint8_t  kMaxPendingDice = 64;
constexpr uint8_t  kMaxRetreatingGroups = 16;

// Sentinel for "no player" in combat bookkeeping. Distinct from
// NPC_PLAYER_ID (255), which is a real combatant: an NPC can legitimately be
// stored as the current defender / firing group, so the empty marker must not
// collide with it.
constexpr uint8_t  kNoPlayer = 254;
// Sentinel sector_id marking a unit as destroyed (awaiting flush from the
// registry at the end of a battle).
constexpr uint16_t kGraveyardSectorId = 999;
// Engagement rounds are capped to break unresolvable stalemates (neither side
// can damage the other); the attacker's ships are destroyed when the cap hits.
constexpr uint8_t  kMaxEngagementRounds = 20;
// The Antimatter Cannon rolls the red die (4 damage); Antimatter Splitter only
// affects this die colour.
constexpr DieColor kAntimatterDie = DieColor::RED;

struct CombatSectorInfo {
    uint16_t sector_id;
    uint8_t  participant_count;
    std::array<uint8_t, kMaxParticipantsPerBattle> participant_ids;
    std::array<uint32_t, kMaxParticipantsPerBattle> latest_arrival;
    uint8_t  defender_idx; // index into participant_ids
};

struct InitiativeGroup {
    uint8_t   player_id;
    ShipType  type;
    int8_t    initiative;
    bool      is_npc;
    bool      destroyed;
    bool      retreating;
    uint8_t   alive_count;   // ships of this group still on the board
    uint8_t   destroyed_count; // ships of this group destroyed this battle
};

struct DestroyedShipRecord {
    uint8_t  player_id;     // destroyed ship owner
    ShipType type;
    uint8_t  count;
    uint8_t  destroyed_by;
};

inline void to_json(nlohmann::json& j, const CombatSectorInfo& s) {
    j = nlohmann::json{
        {"sector_id", s.sector_id},
        {"participant_count", s.participant_count},
        {"participant_ids", s.participant_ids},
        {"latest_arrival", s.latest_arrival},
        {"defender_idx", s.defender_idx},
    };
}

inline void from_json(const nlohmann::json& j, CombatSectorInfo& s) {
    j.at("sector_id").get_to(s.sector_id);
    j.at("participant_count").get_to(s.participant_count);
    j.at("participant_ids").get_to(s.participant_ids);
    j.at("latest_arrival").get_to(s.latest_arrival);
    j.at("defender_idx").get_to(s.defender_idx);
}

inline void to_json(nlohmann::json& j, const InitiativeGroup& g) {
    j = nlohmann::json{
        {"player_id", g.player_id},
        {"type", g.type},
        {"initiative", g.initiative},
        {"is_npc", g.is_npc},
        {"destroyed", g.destroyed},
        {"retreating", g.retreating},
        {"alive_count", g.alive_count},
        {"destroyed_count", g.destroyed_count},
    };
}

inline void from_json(const nlohmann::json& j, InitiativeGroup& g) {
    j.at("player_id").get_to(g.player_id);
    j.at("type").get_to(g.type);
    j.at("initiative").get_to(g.initiative);
    j.at("is_npc").get_to(g.is_npc);
    j.at("destroyed").get_to(g.destroyed);
    j.at("retreating").get_to(g.retreating);
    j.at("alive_count").get_to(g.alive_count);
    j.at("destroyed_count").get_to(g.destroyed_count);
}

inline void to_json(nlohmann::json& j, const DestroyedShipRecord& r) {
    j = nlohmann::json{
        {"player_id", r.player_id},
        {"type", r.type},
        {"count", r.count},
        {"destroyed_by", r.destroyed_by},
    };
}

inline void from_json(const nlohmann::json& j, DestroyedShipRecord& r) {
    j.at("player_id").get_to(r.player_id);
    j.at("type").get_to(r.type);
    j.at("count").get_to(r.count);
    if (j.contains("destroyed_by")) {
        j.at("destroyed_by").get_to(r.destroyed_by);
    } else {
        r.destroyed_by = 255;
    }
}

struct CombatState {
    enum class Phase : uint8_t {
        inactive                  = 0,
        determine_battles         = 1,
        missile_phase             = 2,
        choose_engagement_action  = 3, // player decision: attack or retreat
        engagement_firing         = 4, // cannons + dice rolls
        select_ship_order         = 5, // player chooses firing order among tied initiative groups
        select_reputation_tile    = 6, // player decision
        attack_population         = 7, // auto: remaining ships attack population
        influence_sectors         = 8, // remove control, then player disc decisions
        discovery_award           = 9, // player decision: reward vs VP
        repair                    = 10, // auto: zero damage, reset state
    };

    Phase phase = Phase::inactive;
    uint16_t active_sector_id = 0;

    // Battle queue (descending sector_id order).
    std::array<CombatSectorInfo, kMaxCombatBattles> battle_queue;
    uint8_t battle_queue_size = 0;
    uint8_t current_battle_idx = 0;

    // Per-battle bookkeeping. After each battle's reputation step the
    // destroyed_ships buffer is drained to the players' graveyards / vp
    // totals and then reset for the next battle.
    std::array<DestroyedShipRecord, kMaxDestroyedPerBattle> destroyed_ships;
    uint8_t destroyed_ships_size = 0;

    // Engagement loop. initiative_timeline holds every (participant, type)
    // pair currently in the battle. initiative_idx points at the activating
    // group; the loop runs once per engagement_round. The timeline is
    // rebuilt at the start of each engagement round so destroyed / fully
    // retreated groups are removed.
    std::array<InitiativeGroup, kMaxInitiativeGroups> initiative_timeline;
    uint8_t initiative_size = 0;
    uint8_t initiative_idx  = 0;
    uint8_t engagement_round = 0;
    uint8_t current_attacker_id = kNoPlayer;
    uint8_t current_defender_id = kNoPlayer;

    // Player whose decision is currently pending (engagement choice, rep tile
    // selection, influence disc placement, discovery reward).
    uint8_t pending_player = kNoPlayer;

    // Tied-initiative ship ordering: when a player has multiple ship types with
    // the same initiative in the same pair, they choose the firing order once
    // per round. ship_order_queue holds the ShipType values of tied groups for
    // the current pending_player; empty means no decision is required.
    // The queue is populated at choose_engagement_action time, then the player
    // picks the first ship to fire; remaining ships fire in queue order.
    std::array<ShipType, kMaxInitiativeGroups> ship_order_queue{};
    uint8_t ship_order_size = 0;
    uint8_t ship_order_idx = 0;

    // Active ship type for the current group (used by dice-target assignment
    // and the engagement-action phase).
    ShipType active_ship_type = ShipType::INTERCEPTOR;

    // Legal retreat destination that the current player may choose from. Filled
    // by compute_legal_retreat_destinations at the start of the engagement
    // action phase for the activating player.
    std::array<uint16_t, 6> retreat_destinations;
    uint8_t retreat_destinations_size = 0;

    // Reputation draw pool (at most kMaxReputationDraw tiles drawn from the
    // bag for the current player). The selected tile is moved into
    // players[].reputation_tiles; the rest are returned to the bag and the
    // bag is reshuffled if exhausted.
    std::array<ReputationTiles, kMaxReputationDraw> drawn_tiles;
    uint8_t drawn_tiles_size = 0;
    uint8_t tile_select_player = kNoPlayer;
    // Number of tiles this participant must draw (set when a draw begins; tiles
    // arrive one per chance node until drawn_tiles_size reaches this target).
    uint8_t rep_draw_target = 0;
    // Per-participant draw bookkeeping (1 = already drew / processed).
    // Indexed by CombatSectorInfo::participant_ids position. We allocate
    // kMaxParticipantsPerBattle entries even though only one battle at a
    // time uses it; this keeps the struct fixed-size.
    std::array<uint8_t, kMaxParticipantsPerBattle> reputation_drawn_mask{};
    // How many rep this participant earned (1 participation + destroyed
    // enemy ship values). Used for cap-at-5 enforcement when the player
    // selects a tile.
    uint8_t reputation_earned = 0;

    // Dice target assignment. A volley queues one die per (ship, weapon-die):
    // pending_die_colors[] is filled at setup with values left 0 (unrolled).
    // Each die is rolled by a chance node (PendingRandomEvent::combat_roll),
    // then either auto-resolved or, when the player has a real target choice,
    // assigned by the firing player. pending_target_group_player == kNoPlayer
    // means no volley is in flight.
    uint8_t  pending_target_group_player = kNoPlayer;
    ShipType pending_target_group_type = ShipType::INTERCEPTOR;
    std::array<uint8_t, 128> pending_target_indices;
    uint8_t  pending_target_count = 0;
    std::array<uint8_t, kMaxPendingDice> pending_die_values{};
    std::array<uint8_t, kMaxPendingDice> pending_die_colors{};
    uint8_t pending_die_count = 0;
    uint8_t pending_die_index = 0;
    bool pending_dice_are_missiles = false;
    // When true the active volley is a population bombardment: rolled hits
    // accumulate into pop_attack_damage_remaining instead of hitting ships.
    bool pending_dice_pop_attack = false;

    // Groups that have started retreating. Their ships remain in the battle
    // sector and can be targeted until their next activation.
    std::array<uint8_t, kMaxRetreatingGroups> retreating_players{};
    std::array<ShipType, kMaxRetreatingGroups> retreating_types{};
    std::array<uint16_t, kMaxRetreatingGroups> retreating_destinations{};
    std::array<uint8_t, kMaxRetreatingGroups> retreating_rounds{};
    uint8_t retreating_group_count = 0;
    std::array<uint8_t, kMaxParticipantsPerBattle> reputation_retreat_penalty_mask{};

    // Population attack: sector + remaining damage after non-missile rolls.
    uint16_t pop_attack_sector_id = 0;
    uint8_t  pop_attack_player = kNoPlayer;
    uint8_t  pop_attack_owner = kNoPlayer;
    uint8_t  pop_attack_damage_remaining = 0;
    uint8_t  pop_attack_unit_index = 0;

    // Influence: sectors with no population after bombardment lose control.
    // We then ask each player (in turn order) to place one disc in an
    // uncontrolled sector where they have ships. Sector ids are stored
    // here; the per-player choice uses the to_cell action ids.
    std::array<uint16_t, kMaxGalaxyCells> influence_uncontrolled_sectors;
    uint8_t influence_uncontrolled_size = 0;
    uint8_t influence_scan_index = 0;
    uint8_t influence_turn_order_index = 0;
    uint8_t influence_decision_player = kNoPlayer;
    uint16_t influence_decision_sector = 0;

    // Discovery: sectors with ships and a discovery tile remaining. After
    // combat each such player may take the discovery reward or 2 VP.
    uint16_t discovery_decision_sector = 0;
    uint8_t  discovery_decision_player = kNoPlayer;

    void Reset() {
        phase = Phase::inactive;
        active_sector_id = 0;
        battle_queue_size = 0;
        current_battle_idx = 0;
        destroyed_ships_size = 0;
        initiative_size = 0;
        initiative_idx = 0;
        engagement_round = 0;
        current_attacker_id = kNoPlayer;
        current_defender_id = kNoPlayer;
        pending_player = kNoPlayer;
        active_ship_type = ShipType::INTERCEPTOR;
        retreat_destinations_size = 0;
        drawn_tiles_size = 0;
        tile_select_player = kNoPlayer;
        rep_draw_target = 0;
        reputation_drawn_mask.fill(0);
        reputation_earned = 0;
        ship_order_size = 0;
        ship_order_idx = 0;
        pending_target_group_player = kNoPlayer;
        pending_target_group_type = ShipType::INTERCEPTOR;
        pending_target_count = 0;
        pending_die_values.fill(0);
        pending_die_colors.fill(0);
        pending_die_count = 0;
        pending_die_index = 0;
        pending_dice_are_missiles = false;
        pending_dice_pop_attack = false;
        retreating_players.fill(255);
        retreating_types.fill(ShipType::INTERCEPTOR);
        retreating_destinations.fill(0);
        retreating_rounds.fill(0);
        retreating_group_count = 0;
        reputation_retreat_penalty_mask.fill(0);
        pop_attack_sector_id = 0;
        pop_attack_player = kNoPlayer;
        pop_attack_owner = kNoPlayer;
        pop_attack_damage_remaining = 0;
        pop_attack_unit_index = 0;
        influence_uncontrolled_size = 0;
        influence_scan_index = 0;
        influence_turn_order_index = 0;
        influence_decision_player = kNoPlayer;
        influence_decision_sector = 0;
        discovery_decision_sector = 0;
        discovery_decision_player = kNoPlayer;
    }
};

NLOHMANN_JSON_SERIALIZE_ENUM(CombatState::Phase, {
    {CombatState::Phase::inactive, "inactive"},
    {CombatState::Phase::determine_battles, "determine_battles"},
    {CombatState::Phase::missile_phase, "missile_phase"},
    {CombatState::Phase::choose_engagement_action, "choose_engagement_action"},
    {CombatState::Phase::engagement_firing, "engagement_firing"},
    {CombatState::Phase::select_ship_order, "select_ship_order"},
    {CombatState::Phase::select_reputation_tile, "select_reputation_tile"},
    {CombatState::Phase::attack_population, "attack_population"},
    {CombatState::Phase::influence_sectors, "influence_sectors"},
    {CombatState::Phase::discovery_award, "discovery_award"},
    {CombatState::Phase::repair, "repair"}
});

// JSON for CombatState. We serialize the in-flight substate as a flat object
// so existing fixtures that omit combat_state still load.
inline void to_json(nlohmann::json& j, const CombatState& s) {
    j = nlohmann::json::object();
    j["phase"] = s.phase;
    j["active_sector_id"] = s.active_sector_id;
    j["battle_queue"] = s.battle_queue;
    j["battle_queue_size"] = s.battle_queue_size;
    j["current_battle_idx"] = s.current_battle_idx;
    j["destroyed_ships"] = s.destroyed_ships;
    j["destroyed_ships_size"] = s.destroyed_ships_size;
    j["initiative_timeline"] = s.initiative_timeline;
    j["initiative_size"] = s.initiative_size;
    j["initiative_idx"] = s.initiative_idx;
    j["engagement_round"] = s.engagement_round;
    j["current_attacker_id"] = s.current_attacker_id;
    j["ship_order_queue"] = s.ship_order_queue;
    j["ship_order_size"] = s.ship_order_size;
    j["ship_order_idx"] = s.ship_order_idx;
    j["current_defender_id"] = s.current_defender_id;
    j["pending_player"] = s.pending_player;
    j["active_ship_type"] = s.active_ship_type;
    j["retreat_destinations"] = s.retreat_destinations;
    j["retreat_destinations_size"] = s.retreat_destinations_size;
    j["drawn_tiles"] = s.drawn_tiles;
    j["drawn_tiles_size"] = s.drawn_tiles_size;
    j["tile_select_player"] = s.tile_select_player;
    j["rep_draw_target"] = s.rep_draw_target;
    j["reputation_drawn_mask"] = s.reputation_drawn_mask;
    j["reputation_earned"] = s.reputation_earned;
    j["pending_target_group_player"] = s.pending_target_group_player;
    j["pending_target_group_type"] = s.pending_target_group_type;
    j["pending_target_indices"] = s.pending_target_indices;
    j["pending_target_count"] = s.pending_target_count;
    j["pending_die_values"] = s.pending_die_values;
    j["pending_die_colors"] = s.pending_die_colors;
    j["pending_die_count"] = s.pending_die_count;
    j["pending_die_index"] = s.pending_die_index;
    j["pending_dice_are_missiles"] = s.pending_dice_are_missiles;
    j["pending_dice_pop_attack"] = s.pending_dice_pop_attack;
    j["retreating_players"] = s.retreating_players;
    j["retreating_types"] = s.retreating_types;
    j["retreating_destinations"] = s.retreating_destinations;
    j["retreating_rounds"] = s.retreating_rounds;
    j["retreating_group_count"] = s.retreating_group_count;
    j["reputation_retreat_penalty_mask"] = s.reputation_retreat_penalty_mask;
    j["pop_attack_sector_id"] = s.pop_attack_sector_id;
    j["pop_attack_player"] = s.pop_attack_player;
    j["pop_attack_owner"] = s.pop_attack_owner;
    j["pop_attack_damage_remaining"] = s.pop_attack_damage_remaining;
    j["pop_attack_unit_index"] = s.pop_attack_unit_index;
    j["influence_uncontrolled_sectors"] = s.influence_uncontrolled_sectors;
    j["influence_uncontrolled_size"] = s.influence_uncontrolled_size;
    j["influence_scan_index"] = s.influence_scan_index;
    j["influence_turn_order_index"] = s.influence_turn_order_index;
    j["influence_decision_player"] = s.influence_decision_player;
    j["influence_decision_sector"] = s.influence_decision_sector;
    j["discovery_decision_sector"] = s.discovery_decision_sector;
    j["discovery_decision_player"] = s.discovery_decision_player;
}

inline void from_json(const nlohmann::json& j, CombatState& s) {
    s = CombatState{};
    if (j.contains("phase")) j.at("phase").get_to(s.phase);
    if (j.contains("active_sector_id")) j.at("active_sector_id").get_to(s.active_sector_id);
    if (j.contains("battle_queue")) j.at("battle_queue").get_to(s.battle_queue);
    if (j.contains("battle_queue_size")) j.at("battle_queue_size").get_to(s.battle_queue_size);
    if (j.contains("current_battle_idx")) j.at("current_battle_idx").get_to(s.current_battle_idx);
    if (j.contains("destroyed_ships")) j.at("destroyed_ships").get_to(s.destroyed_ships);
    if (j.contains("destroyed_ships_size")) j.at("destroyed_ships_size").get_to(s.destroyed_ships_size);
    if (j.contains("initiative_timeline")) j.at("initiative_timeline").get_to(s.initiative_timeline);
    if (j.contains("initiative_size")) j.at("initiative_size").get_to(s.initiative_size);
    if (j.contains("initiative_idx")) j.at("initiative_idx").get_to(s.initiative_idx);
    if (j.contains("engagement_round")) j.at("engagement_round").get_to(s.engagement_round);
    if (j.contains("current_attacker_id")) j.at("current_attacker_id").get_to(s.current_attacker_id);
    if (j.contains("current_defender_id")) j.at("current_defender_id").get_to(s.current_defender_id);
    if (j.contains("ship_order_queue")) j.at("ship_order_queue").get_to(s.ship_order_queue);
    if (j.contains("ship_order_size")) j.at("ship_order_size").get_to(s.ship_order_size);
    if (j.contains("ship_order_idx")) j.at("ship_order_idx").get_to(s.ship_order_idx);
    if (j.contains("pending_player")) j.at("pending_player").get_to(s.pending_player);
    if (j.contains("active_ship_type")) j.at("active_ship_type").get_to(s.active_ship_type);
    if (j.contains("retreat_destinations")) j.at("retreat_destinations").get_to(s.retreat_destinations);
    if (j.contains("retreat_destinations_size")) j.at("retreat_destinations_size").get_to(s.retreat_destinations_size);
    if (j.contains("drawn_tiles")) j.at("drawn_tiles").get_to(s.drawn_tiles);
    if (j.contains("drawn_tiles_size")) j.at("drawn_tiles_size").get_to(s.drawn_tiles_size);
    if (j.contains("tile_select_player")) j.at("tile_select_player").get_to(s.tile_select_player);
    if (j.contains("rep_draw_target")) j.at("rep_draw_target").get_to(s.rep_draw_target);
    if (j.contains("reputation_drawn_mask")) {
        const auto& m = j.at("reputation_drawn_mask");
        for (size_t i = 0; i < s.reputation_drawn_mask.size() && i < m.size(); ++i) {
            s.reputation_drawn_mask[i] = m[i].get<uint8_t>();
        }
    }
    if (j.contains("reputation_earned")) j.at("reputation_earned").get_to(s.reputation_earned);
    if (j.contains("pending_target_group_player")) j.at("pending_target_group_player").get_to(s.pending_target_group_player);
    if (j.contains("pending_target_group_type")) j.at("pending_target_group_type").get_to(s.pending_target_group_type);
    if (j.contains("pending_target_indices")) j.at("pending_target_indices").get_to(s.pending_target_indices);
    if (j.contains("pending_target_count")) j.at("pending_target_count").get_to(s.pending_target_count);
    if (j.contains("pending_die_values")) j.at("pending_die_values").get_to(s.pending_die_values);
    if (j.contains("pending_die_colors")) j.at("pending_die_colors").get_to(s.pending_die_colors);
    if (j.contains("pending_die_count")) j.at("pending_die_count").get_to(s.pending_die_count);
    if (j.contains("pending_die_index")) j.at("pending_die_index").get_to(s.pending_die_index);
    if (j.contains("pending_dice_are_missiles")) j.at("pending_dice_are_missiles").get_to(s.pending_dice_are_missiles);
    if (j.contains("pending_dice_pop_attack")) j.at("pending_dice_pop_attack").get_to(s.pending_dice_pop_attack);
    if (j.contains("retreating_players")) j.at("retreating_players").get_to(s.retreating_players);
    if (j.contains("retreating_types")) j.at("retreating_types").get_to(s.retreating_types);
    if (j.contains("retreating_destinations")) j.at("retreating_destinations").get_to(s.retreating_destinations);
    if (j.contains("retreating_rounds")) j.at("retreating_rounds").get_to(s.retreating_rounds);
    if (j.contains("retreating_group_count")) j.at("retreating_group_count").get_to(s.retreating_group_count);
    if (j.contains("reputation_retreat_penalty_mask")) {
        j.at("reputation_retreat_penalty_mask").get_to(s.reputation_retreat_penalty_mask);
    }
    if (j.contains("pop_attack_sector_id")) j.at("pop_attack_sector_id").get_to(s.pop_attack_sector_id);
    if (j.contains("pop_attack_player")) j.at("pop_attack_player").get_to(s.pop_attack_player);
    if (j.contains("pop_attack_owner")) j.at("pop_attack_owner").get_to(s.pop_attack_owner);
    if (j.contains("pop_attack_damage_remaining")) j.at("pop_attack_damage_remaining").get_to(s.pop_attack_damage_remaining);
    if (j.contains("pop_attack_unit_index")) j.at("pop_attack_unit_index").get_to(s.pop_attack_unit_index);
    if (j.contains("influence_uncontrolled_sectors")) {
        j.at("influence_uncontrolled_sectors").get_to(s.influence_uncontrolled_sectors);
    }
    if (j.contains("influence_uncontrolled_size")) j.at("influence_uncontrolled_size").get_to(s.influence_uncontrolled_size);
    if (j.contains("influence_scan_index")) j.at("influence_scan_index").get_to(s.influence_scan_index);
    if (j.contains("influence_turn_order_index")) j.at("influence_turn_order_index").get_to(s.influence_turn_order_index);
    if (j.contains("influence_decision_player")) j.at("influence_decision_player").get_to(s.influence_decision_player);
    if (j.contains("influence_decision_sector")) j.at("influence_decision_sector").get_to(s.influence_decision_sector);
    if (j.contains("discovery_decision_sector")) j.at("discovery_decision_sector").get_to(s.discovery_decision_sector);
    if (j.contains("discovery_decision_player")) j.at("discovery_decision_player").get_to(s.discovery_decision_player);
}

// Driver entry points called from eclipse.cc.
void begin_combat_phase(::State& state);

// Helpers exposed for tests and the eclipse.cc integration layer.
bool identify_combat_sectors(const ::State& state,
                             std::array<CombatSectorInfo, kMaxCombatBattles>& out,
                             uint8_t& out_size);

// Whether unit target_idx is a legal recipient of a die from `attacker_id`'s
// current volley (enemy in the active pair, in the active sector).
bool IsLegalDieTarget(const ::State& state, uint8_t attacker_id,
                      size_t target_idx);

// Resolve a single rolled die. `face` is the d6 result (1-6) produced by a
// PendingRandomEvent::combat_roll chance node. Applies self-damage (Rift),
// then either auto-resolves the die (miss, single viable target, or NPC fire)
// or leaves it for the firing player to assign a target.
void ResolveCombatDie(::State& state, uint8_t face);

// Apply the firing player's chosen target for the die currently awaiting
// assignment, then advance the volley.
void ApplyPlayerDieTarget(::State& state, size_t target_idx);

// Draw one reputation tile of the given value into the draw pool. Driven by a
// PendingRandomEvent::reputation_draw chance node.
void DrawOneReputationTile(::State& state, ReputationTiles value);

void AddRetreatingGroup(::State& state, uint8_t player_id, ShipType type,
                        uint16_t destination_sector_id);
void FlushDestroyedShips(::State& state);
void RebuildInitiativeTimeline(::State& state);
// Advance the combat state machine one deterministic step (no randomness;
// dice and tile draws are resolved via chance nodes).
void advance_combat_state(::State& state);
int ReputationValueFor(ShipType t);

} // namespace open_spiel::eclipse

#endif // ECLIPSE_COMBAT_H
