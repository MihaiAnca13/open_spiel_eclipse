//
// Created by Mihai on 01/06/2026.
//

#include "open_spiel/games/eclipse/eclipse.h"

#include "open_spiel/games/eclipse/observation.h"

#include <algorithm>
#include <bitset>
#include <sstream>
#include <stdexcept>

#include "open_spiel/games/eclipse/species.h"
#include "open_spiel/games/eclipse/systems/actions/bonus.h"
#include "open_spiel/games/eclipse/systems/actions/diplomacy.h"
#include "open_spiel/games/eclipse/systems/actions/minor_species_diplomacy.h"
#include "open_spiel/games/eclipse/systems/actions/explore.h"
#include "open_spiel/games/eclipse/systems/actions/research.h"
#include "open_spiel/games/eclipse/systems/actions/build.h"
#include "open_spiel/games/eclipse/systems/actions/influence.h"
#include "open_spiel/games/eclipse/systems/actions/upgrade.h"
#include "open_spiel/games/eclipse/systems/actions/move.h"
#include "open_spiel/games/eclipse/systems/setup.h"
#include "open_spiel/games/eclipse/systems/upkeep.h"
#include "open_spiel/games/eclipse/systems/scoring.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/observer.h"

namespace open_spiel {
namespace eclipse {

namespace {

constexpr Action chance_resolve = 0;

// Action layout:
//   0        = PASS  (also the no-op chance-resolve token)
//   1        = start RESEARCH action
//   2        = stop RESEARCH action (for multi-activation species)
//   3-26     = standard techs (24 indices, 0-23 of TECH_TABLE)
//   27-74    = rare techs (16 rare techs * 3 tracks = 48 indices)
//   75-82    = BUILD (index 0-7)
//   83       = start an EXPLORE action
//   84 / 85  = explore: place / discard the drawn tile
//   86-91    = explore: place with rotation 0-5
//   92 / 93  = explore: take control / decline
//   94 / 95  = explore: discovery reward / 2 VP
//   96 / 97  = explore (Draco): keep drawn tile 0 / 1
//   98 / 99  = explore (Draco): flip a second tile / proceed with one
//   100      = explore: stop (decline remaining activations)
//   101..325 = explore: choose zone == galaxy cell index (one id per hex)
//   326-331  = TRADE (index into TradeConversion enum)
//   332+     = COLONY_SHIP: cell_idx*(SLOTS*TRACKS) + slot_idx*TRACKS + track
//             track: 0=Money, 1=Science, 2=Materials
constexpr Action action_pass = 0;
constexpr Action action_research = 1;
constexpr Action action_research_stop = 2;
constexpr Action action_research_standard_start = 3;
constexpr Action action_research_rare_start = 27;
constexpr Action action_build = 75;
constexpr Action action_explore = 83;
constexpr Action explore_place = 84;
constexpr Action explore_discard = 85;
constexpr Action explore_rotation_start = 86;
constexpr Action explore_claim_yes = 92;
constexpr Action explore_claim_no = 93;
constexpr Action explore_discovery_reward = 94;
constexpr Action explore_discovery_vp = 95;
constexpr Action explore_select_tile_start = 96;
constexpr Action explore_draw_again = 98;
constexpr Action explore_skip_second = 99;
constexpr Action explore_stop = 100;
constexpr Action explore_zone_start = 101;  // + galaxy cell index (0..224)
constexpr Action action_trade_start        = 326;
constexpr Action action_colony_ship_start  = 332;
constexpr int COLONY_SHIP_SLOTS_PER_CELL   = 8;
constexpr int COLONY_SHIP_TRACKS           = POP_TRACK_COUNT;  // 3
constexpr int COLONY_SHIP_CODES_PER_CELL   =
    COLONY_SHIP_SLOTS_PER_CELL * COLONY_SHIP_TRACKS;

constexpr Action action_influence_start = action_colony_ship_start + GALAXY_CELL_COUNT * COLONY_SHIP_CODES_PER_CELL; // 5732
constexpr Action action_influence_stop = action_influence_start + 1; // 5733
constexpr Action action_influence_to_cell_start = action_influence_stop + 1; // 5734
constexpr Action action_reclaim_from_cell_start = action_influence_to_cell_start + GALAXY_CELL_COUNT; // 5959
constexpr Action action_choose_return_track_start = action_reclaim_from_cell_start + GALAXY_CELL_COUNT; // 6184

constexpr Action action_build_stop = action_choose_return_track_start + 3; // 6187
constexpr Action action_build_choice_start = action_build_stop + 1; // 6188
constexpr Action action_build_end = action_build_choice_start + BUILD_TYPE_COUNT * GALAXY_CELL_COUNT; // 6188 + 6*225 = 7538

// Upgrade action IDs (4 ship types * 8 slots * ~30 parts = 960 choices)
constexpr Action action_upgrade = action_build_end + 1; // 7539
constexpr Action action_upgrade_stop = action_upgrade + 1; // 7540
constexpr Action action_upgrade_choice_start = action_upgrade_stop + 1; // 7541
constexpr int UPGRADE_SHIP_COUNT = 4; // INTERCEPTOR, CRUISER, DREADNOUGHT, STARBASE
constexpr int UPGRADE_SLOTS_PER_SHIP = 8;
constexpr int UPGRADE_PART_COUNT = 50; // Increased to support all discovery ship parts as well
constexpr Action action_upgrade_end = action_upgrade_choice_start + UPGRADE_SHIP_COUNT * UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT; // 7541 + 960 = 8501
constexpr int num_distinct_actions_upgrade = action_upgrade_end + 1; // 8501 + 1 = 8502

// Move action IDs (128 units * 7 codes = 896 step codes, 128 warp entry codes, 225 warp destinations)
constexpr Action action_move = num_distinct_actions_upgrade; // 8502
constexpr Action action_move_stop = action_move + 1; // 8503
constexpr Action action_move_choice_start = action_move_stop + 1; // 8504
constexpr int MAX_MOVE_UNITS = 128; // unit_registry capacity
constexpr Action action_move_warp_start = action_move_choice_start + MAX_MOVE_UNITS * MOVE_CODES_PER_UNIT; // 8504 + 896 = 9400
constexpr Action action_move_warp_destination_start = action_move_warp_start + MAX_MOVE_UNITS; // 9400 + 128 = 9528
constexpr Action action_upkeep_colony_done = action_move_warp_destination_start + GALAXY_CELL_COUNT; // 9753
constexpr Action action_upkeep_pay_done = action_upkeep_colony_done + 1; // 9754

// Combat action IDs (appended after upkeep).
constexpr Action action_combat_continue = action_upkeep_pay_done + 1; // 9755
constexpr Action action_combat_attack    = action_combat_continue + 1; // 9756
constexpr Action action_combat_retreat_to_cell_start = action_combat_attack + 1; // 9757 (+ 225 cells)
constexpr Action action_combat_dice_target_start =
    action_combat_retreat_to_cell_start + GALAXY_CELL_COUNT; // 9982 (+ 128 units)
constexpr Action action_combat_rep_select_start =
    action_combat_dice_target_start + 128; // 10110 (+ 5 tiles)
constexpr Action action_combat_rep_skip =
    action_combat_rep_select_start + 5; // 10115
constexpr Action action_combat_pop_target_start =
    action_combat_rep_skip + 1; // 10116 (+ 16 slots)
constexpr Action action_combat_influence_yes =
    action_combat_pop_target_start + 16; // 10132
constexpr Action action_combat_influence_no = action_combat_influence_yes + 1; // 10133
constexpr Action action_combat_discovery_reward = action_combat_influence_no + 1; // 10134
constexpr Action action_combat_discovery_vp = action_combat_discovery_reward + 1; // 10135

// Ship order selection: player picks which of their tied-initiative ship types
// fires first. 4 ShipType values (INTERCEPTOR, CRUISER, DREADNOUGHT, STARBASE).
// This action appears during combat's select_ship_order phase.
constexpr Action action_combat_ship_order_start = action_combat_discovery_vp + 1; // 10136
constexpr Action action_combat_influence_to_cell_start =
    action_combat_ship_order_start + 4; // 10140 (+ 225 cells)

// ── Diplomacy action IDs (rulebook p.14-15) ──────────────────────────────────
// Propose: one per (proposer, partner) pair, 6*6 = 36 ids.
constexpr Action action_diplomacy_propose_start =
    action_combat_influence_to_cell_start + GALAXY_CELL_COUNT; // 10365
constexpr Action action_diplomacy_propose_end =
    action_diplomacy_propose_start + MAX_PLAYERS * MAX_PLAYERS; // 10397

// Each side's Pop Track pick at formation: 3 tracks x 2 sides = 6 ids.
// Encoding: action_diplomacy_pick_track_start + side*3 + track
//   side=0: proposer, side=1: partner
constexpr Action action_diplomacy_pick_track_start = action_diplomacy_propose_end; // 10397
constexpr Action action_diplomacy_pick_track_end =
    action_diplomacy_pick_track_start + 6; // 10403

// Rearrange: return tile to bag. 5 slots per side x 2 sides = 10 ids.
// Encoding: action_diplomacy_return_start + side*5 + slot
constexpr Action action_diplomacy_return_start = action_diplomacy_pick_track_end; // 10403
constexpr Action action_diplomacy_return_end =
    action_diplomacy_return_start + 10; // 10413

// Swap: AMBASSADOR_OR_REP slot <-> free REP_ONLY slot. 5 source slots * 5 dest
// slots - 5 self = 20 swaps per side x 2 sides = 40 ids.
// Encoding: action_diplomacy_swap_start + side*20 + (src*5 + dst) where src != dst
constexpr Action action_diplomacy_swap_start = action_diplomacy_return_end; // 10413
constexpr Action action_diplomacy_swap_end =
    action_diplomacy_swap_start + 40; // 10453

// Accept/decline a diplomacy proposal (Fix #1: partner-accept step).
constexpr Action action_diplomacy_accept = action_diplomacy_swap_end; // 10453
constexpr Action action_diplomacy_decline = action_diplomacy_accept + 1; // 10454

// ── Minor Species Diplomacy action IDs ─────────────────────────────────────
// Free bonus action: form diplomatic relations with a minor species.
// Encoding: action_minor_species_start + ms_idx (0..8)
constexpr Action action_minor_species_start = action_diplomacy_decline + 1; // 10455
constexpr Action action_minor_species_end =
    action_minor_species_start + MINOR_SPECIES_COUNT; // 10464

// PLACE_POP_CUBE track choice: 3 tracks (0=Money, 1=Science, 2=Materials).
constexpr Action action_minor_species_track_start = action_minor_species_end; // 10464
constexpr Action action_minor_species_track_end =
    action_minor_species_track_start + POP_TRACK_COUNT; // 10467

// Artifact Key resource-choice sub-action (3 ids: Materials/Science/Money).
constexpr Action action_artifact_key_track_start = action_minor_species_track_end; // 10467
constexpr Action action_artifact_key_track_end =
    action_artifact_key_track_start + POP_TRACK_COUNT; // 10470

// ── REACTION action IDs ─────────────────────────────────────────────────────
// After passing, players may take 1 Activation of Upgrade, Build, or Move.
constexpr Action action_reaction_upgrade = action_artifact_key_track_end;     // 10470
constexpr Action action_reaction_build   = action_artifact_key_track_end + 1; // 10471
constexpr Action action_reaction_move    = action_artifact_key_track_end + 2; // 10472

constexpr int num_distinct_actions = action_reaction_move + 1; // 10473

const GameType game_type{
    /*short_name=*/"eclipse",
    /*long_name=*/"Eclipse: New Dawn for the Galaxy",
    /*dynamics=*/GameType::Dynamics::kSequential,
    /*chance_mode=*/GameType::ChanceMode::kSampledStochastic,
    /*information=*/GameType::Information::kImperfectInformation,
    /*utility=*/GameType::Utility::kGeneralSum,
    /*reward_model=*/GameType::RewardModel::kTerminal,
    /*max_num_players=*/6,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{
        {"players", GameParameter(4)},
        {"rng_seed", GameParameter(0)},
        {"npc_difficulty", GameParameter(std::string("Easy"))},
        {"warped_universe", GameParameter(false)},
        {"randomize_races", GameParameter(false)},
        {"race_alien_prob", GameParameter(0.8)},
        {"randomize_npc_difficulty", GameParameter(false)},
        {"randomize_warped", GameParameter(false)},
        {"warped_prob", GameParameter(0.5)},
        {"species_p0", GameParameter(std::string("Terran Factions"))},
        {"species_p1", GameParameter(std::string("Terran Factions"))},
        {"species_p2", GameParameter(std::string("Terran Factions"))},
        {"species_p3", GameParameter(std::string("Terran Factions"))},
        {"species_p4", GameParameter(std::string("Terran Factions"))},
        {"species_p5", GameParameter(std::string("Terran Factions"))},
    },
};

std::shared_ptr<const Game> CreateGame(const GameParameters& params) {
  return std::make_shared<EclipseGame>(params);
}

REGISTER_SPIEL_GAME(game_type, CreateGame);
RegisterSingleTensorObserver single_tensor(game_type.short_name);

std::string PendingRandomEventToString(
    EclipseState::PendingRandomEvent pending_event) {
  switch (pending_event) {
    case EclipseState::PendingRandomEvent::none:
      return "none";
    case EclipseState::PendingRandomEvent::initial_setup:
      return "initial_setup";
    case EclipseState::PendingRandomEvent::explore_draw:
      return "explore_draw";
    case EclipseState::PendingRandomEvent::combat_roll:
      return "combat_roll";
    case EclipseState::PendingRandomEvent::reputation_draw:
      return "reputation_draw";
  }
  return "unknown";
}

EclipseState::PendingRandomEvent PendingRandomEventFromInt(int value) {
  if (value < 0 ||
      value >
          static_cast<int>(EclipseState::PendingRandomEvent::reputation_draw)) {
    throw std::invalid_argument("invalid pending random event");
  }
  return static_cast<EclipseState::PendingRandomEvent>(value);
}

bool HasLegalResearchChoice(const ::State& state, uint8_t player_id) {
  if (player_id >= state.players.size()) return false;
  const ::Player& player = state.players[player_id];

  for (size_t i = 0; i < TECH_TABLE_SIZE; ++i) {
    const TechDefinition& def = TECH_TABLE[i];
    if (def.category == TechCategory::RARE) continue;
    if (player.has_tech(def.bit)) continue;
    if (state.get_tech_tray_count(def.bit) == 0) continue;
    if (get_track_tile_count(player, def.category) >= 8) continue;
    if (player.resources.science < calculate_research_cost(player, def, def.category)) continue;
    return true;
  }

  for (size_t rare_idx = 0; rare_idx < TECH_RARE_COUNT; ++rare_idx) {
    const TechDefinition& def = TECH_TABLE[TECH_TABLE_SIZE + rare_idx];
    if (def.category != TechCategory::RARE) continue;
    if (player.has_tech(def.bit)) continue;
    if (state.get_tech_tray_count(def.bit) == 0) continue;

    for (int track = 0; track < 3; ++track) {
      const TechCategory target_track = static_cast<TechCategory>(track);
      if (get_track_tile_count(player, target_track) >= 8) continue;
      if (player.resources.science < calculate_research_cost(player, def, target_track)) continue;
      return true;
    }
  }

  return false;
}

bool HasLegalBuildChoice(const ::State& state, uint8_t player_id) {
  if (player_id >= state.players.size()) return false;
  // Build the unit-supply counts and the player's owned-cell list once, then
  // probe only those cells (most of the 225-hex grid is not ours).
  const PlayerUnitCounts counts = BuildUnitCounts(state, player_id);
  const std::vector<uint8_t> owned = PlayerOwnedBuildCells(state, player_id);
  if (owned.empty()) return false;
  for (int t = 0; t < BUILD_TYPE_COUNT; ++t) {
    const BuildType type = static_cast<BuildType>(t);
    for (uint8_t cell : owned) {
      if (can_build(state, player_id, type, cell, &counts)) {
        return true;
      }
    }
  }
  return false;
}

bool HasLegalInfluenceChoice(const ::State& state, uint8_t player_id) {
  if (player_id >= state.players.size()) return false;
  // Build the ship-presence bitsets once; per-cell checks are then O(1)
  // instead of each re-scanning the unit registry.
  const InfluenceShipMap ship_map = BuildInfluenceShipMap(state, player_id);
  for (int cell = 0; cell < GALAXY_CELL_COUNT; ++cell) {
    if (can_influence_to_sector(state, player_id, static_cast<uint8_t>(cell), &ship_map) ||
        can_reclaim_from_sector(state, player_id, static_cast<uint8_t>(cell))) {
      return true;
    }
  }
  return false;
}

bool HasLegalUpgradeChoice(const ::State& state, uint8_t player_id) {
  if (player_id >= state.players.size()) return false;
  const ::Player& player = state.players[player_id];
  // Only probe parts the player can legally place anywhere (tech owned, or
  // discovery part in inventory) instead of all of SHIP_PART_TABLE.
  const std::vector<ShipPartId> parts = PlaceablePartIds(state, player_id);

  for (int ship = 0; ship < UPGRADE_SHIP_COUNT; ++ship) {
    const Blueprint& bp = player.blueprints[ship];
    for (int slot = 0; slot < UPGRADE_SLOTS_PER_SHIP; ++slot) {
      if (slot >= bp.capacity) continue;
      const ShipType ship_type = static_cast<ShipType>(ship);
      if (can_upgrade(state, player_id, ship_type, static_cast<uint8_t>(slot), ShipPartId::NONE)) {
        return true;
      }
      for (ShipPartId part_id : parts) {
        if (can_upgrade(state, player_id, ship_type, static_cast<uint8_t>(slot), part_id)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool HasLegalMoveChoice(const ::State& state, uint8_t player_id) {
  if (player_id >= state.players.size()) return false;
  // Reuse legal_move_steps to check if any unit can move.
  // A non-empty result means at least one legal move step exists.
  return !open_spiel::eclipse::legal_move_steps(state, player_id).empty();
}

void AppendActionPhaseBonusActions(const ::State& state, uint8_t player_id,
                                   std::vector<Action>* actions) {
  if (player_id >= state.players.size()) return;

  const ::Player& player = state.players[player_id];
  for (int c = 0; c < TRADE_CONVERSION_COUNT; ++c) {
    if (can_trade(player, static_cast<TradeConversion>(c))) {
      actions->push_back(action_trade_start + c);
    }
  }
  for (const auto& placement : legal_colony_ship_placements(state, player_id)) {
    actions->push_back(action_colony_ship_start +
                       placement.cell * COLONY_SHIP_CODES_PER_CELL +
                       placement.slot * COLONY_SHIP_TRACKS +
                       static_cast<int>(placement.track));
  }

  if (state.players.size() >= 4) {
    for (uint8_t partner = 0; partner < state.players.size(); ++partner) {
      if (partner != player_id && can_propose_diplomacy(state, player_id, partner)) {
        actions->push_back(action_diplomacy_propose_start +
                           player_id * MAX_PLAYERS + partner);
      }
    }
  }

  for (uint8_t ms_idx : state.minor_species_pool) {
    if (can_form_minor_species(state, player_id, ms_idx)) {
      actions->push_back(action_minor_species_start + ms_idx);
    }
  }
}

bool IsActionPhaseBonusAction(Action action_id) {
  return (action_id >= action_trade_start &&
          action_id < action_colony_ship_start) ||
         (action_id >= action_colony_ship_start &&
          action_id < action_influence_start) ||
         (action_id >= action_diplomacy_propose_start &&
          action_id < action_diplomacy_propose_end) ||
         (action_id >= action_minor_species_start &&
          action_id < action_minor_species_end);
}

void ApplyActionPhaseBonusAction(::State& state, uint8_t player_id,
                                 Action action_id) {
  SPIEL_CHECK_TRUE(player_id < state.players.size());

  if (action_id >= action_trade_start && action_id < action_colony_ship_start) {
    SPIEL_CHECK_TRUE(execute_trade(
        state, player_id,
        static_cast<TradeConversion>(action_id - action_trade_start)));
    return;
  }
  if (action_id >= action_colony_ship_start && action_id < action_influence_start) {
    const int encoded = static_cast<int>(action_id - action_colony_ship_start);
    const int cell = encoded / COLONY_SHIP_CODES_PER_CELL;
    const int remainder = encoded % COLONY_SHIP_CODES_PER_CELL;
    SPIEL_CHECK_TRUE(use_colony_ship(
        state, player_id, static_cast<uint8_t>(cell),
        static_cast<uint8_t>(remainder / COLONY_SHIP_TRACKS),
        static_cast<PopTrack>(remainder % COLONY_SHIP_TRACKS)));
    return;
  }
  if (action_id >= action_diplomacy_propose_start &&
      action_id < action_diplomacy_propose_end) {
    const int encoded = static_cast<int>(action_id - action_diplomacy_propose_start);
    const int proposer = encoded / MAX_PLAYERS;
    const int partner = encoded % MAX_PLAYERS;
    SPIEL_CHECK_EQ(proposer, player_id);
    SPIEL_CHECK_TRUE(begin_diplomacy(state, player_id,
                                     static_cast<uint8_t>(partner)));
    return;
  }
  SPIEL_CHECK_TRUE(action_id >= action_minor_species_start &&
                   action_id < action_minor_species_end);
  SPIEL_CHECK_TRUE(begin_minor_species_formation(
      state, player_id, action_id - action_minor_species_start));
}

bool SectorHasOpponentShips(const ::State& state, uint8_t player_id,
                            uint8_t galaxy_cell_idx) {
  const HexCoord coord = index_to_hex(galaxy_cell_idx);
  const Sector& sector = state.galaxy.at(coord.q, coord.r);
  if (sector.sector_id == 0) return false;
  for (const Unit& unit : state.unit_registry) {
    if (unit.sector_id == sector.sector_id &&
        unit.player_id != player_id &&
        unit.player_id != NPC_PLAYER_ID) {
      return true;
    }
  }
  return false;
}

void AppendLegalUpkeepColonyShipActions(const ::State& state, uint8_t player_id,
                                        std::vector<Action>& actions) {
  if (player_id >= state.players.size()) return;
  const ::Player& player = state.players[player_id];
  if (player.colony_ships_available == 0) return;

  constexpr int max_sector_id = 512;
  std::bitset<max_sector_id> sectors_with_opponent_player_ships;
  for (const Unit& unit : state.unit_registry) {
    if (unit.sector_id < max_sector_id && unit.player_id != player_id &&
        unit.player_id != NPC_PLAYER_ID) {
      sectors_with_opponent_player_ships.set(unit.sector_id);
    }
  }

  for (int cell = 0; cell < GALAXY_CELL_COUNT; ++cell) {
    const HexCoord coord = index_to_hex(cell);
    const Sector& sector = state.galaxy.at(coord.q, coord.r);
    if (sector.sector_id == 0 || sector.owner_id != player_id ||
        sector.sector_id >= max_sector_id ||
        sectors_with_opponent_player_ships.test(sector.sector_id)) {
      continue;
    }

    const SectorDefinition* def = get_sector_definition(sector.sector_id);
    if (!def) continue;
    for (uint8_t slot = 0; slot < static_cast<uint8_t>(def->slots.size());
         ++slot) {
      for (uint8_t track_id = 0; track_id < POP_TRACK_COUNT; ++track_id) {
        const PopTrack track = static_cast<PopTrack>(track_id);
        if (can_use_colony_ship(state, player_id, static_cast<uint8_t>(cell),
                                slot, track)) {
          actions.push_back(action_colony_ship_start +
                            cell * COLONY_SHIP_CODES_PER_CELL +
                            slot * COLONY_SHIP_TRACKS +
                            static_cast<int>(track));
        }
      }
    }
  }
}

void AppendReclaimActions(const ::State& state, uint8_t player_id,
                          std::vector<Action>& actions) {
  for (int cell = 0; cell < GALAXY_CELL_COUNT; ++cell) {
    if (can_reclaim_from_sector(state, player_id, static_cast<uint8_t>(cell))) {
      actions.push_back(action_reclaim_from_cell_start + cell);
    }
  }
}

bool HasReclaimableSector(const ::State& state, uint8_t player_id) {
  for (int cell = 0; cell < GALAXY_CELL_COUNT; ++cell) {
    if (can_reclaim_from_sector(state, player_id, static_cast<uint8_t>(cell))) {
      return true;
    }
  }
  return false;
}

bool ProcessCurrentPendingReturnsAuto(::State& state) {
  UpkeepState& us = state.upkeep_state;
  if (us.player_id >= state.players.size()) return true;
  ::Player& player = state.players[us.player_id];

  while (!us.pending_returns.empty()) {
    const PendingReturn pending = us.pending_returns.front();
    if (pending_return_requires_choice(player, pending.type,
                                       pending.is_orbital)) {
      us.step = UpkeepState::Step::choose_return_track;
      return false;
    }
    apply_return_to_track(player, get_matching_track(pending.type));
    us.pending_returns.erase(us.pending_returns.begin());
  }
  return true;
}

void RemovePlayerFromBoard(::State& state, uint8_t player_id) {
  for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
    for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
      Sector& sector = state.galaxy.at(q, r);
      if (sector.owner_id == player_id) {
        sector.owner_id = 255;
        sector.occupied_slots_mask = 0;
      }
    }
  }
  FixedVector<Unit, 128> filtered_units;
  for (const Unit& unit : state.unit_registry) {
    if (unit.player_id != player_id) {
      filtered_units.push_back(unit);
    }
  }
  state.unit_registry = filtered_units;
}

void QueueCleanupGraveyardReturns(::State& state, uint8_t player_id) {
  state.upkeep_state.pending_returns.clear();
  if (player_id >= state.players.size()) return;
  ::Player& player = state.players[player_id];
  const size_t total_returns = player.graveyard_counts[0] +
                               player.graveyard_counts[1] +
                               player.graveyard_counts[2];
  state.upkeep_state.pending_returns.resize(total_returns);
  static const PlanetType kTrackTypes[3] = {
      PlanetType::MONEY, PlanetType::SCIENCE, PlanetType::MATERIALS};
  size_t pending_index = 0;
  for (int track = 0; track < 3; ++track) {
    while (player.graveyard_counts[track] > 0) {
      state.upkeep_state.pending_returns[pending_index++] = {
          kTrackTypes[track], false};
      --player.graveyard_counts[track];
    }
  }
}

}  // namespace

EclipseGame::EclipseGame(const GameParameters& params)
    : Game(game_type, params),
      rng_(std::mt19937_64(static_cast<uint64_t>(ParameterValue<int>(
          "rng_seed")))) {}

int EclipseGame::NumDistinctActions() const { return num_distinct_actions; }

int EclipseGame::NumPlayers() const { return ParameterValue<int>("players"); }

int EclipseGame::MaxGameLength() const { return 1000; }

std::vector<int> EclipseGame::ObservationTensorShape() const {
  // Fixed size for max players (6); smaller games zero-pad the trailing seat
  // blocks (each carries an `occupied` bit). See observation.h for the layout.
  return {open_spiel::eclipse::obs::kTotalSize};
}

std::unique_ptr<State> EclipseGame::NewInitialState() const {
  return std::make_unique<EclipseState>(shared_from_this());
}

std::unique_ptr<State> EclipseGame::DeserializeState(
    const std::string& str) const {
  auto state = std::make_unique<EclipseState>(shared_from_this());
  if (str.empty()) {
    return state;
  }

  nlohmann::json value = nlohmann::json::parse(str);
  SetupConfig config = NormalizeSetupConfig(value.at("setup_config").get<SetupConfig>());
  ::State raw_state = value.at("state").get<::State>();
  EclipseState::PendingRandomEvent pending_random_event =
      PendingRandomEventFromInt(value.at("pending_random_event").get<int>());
  state->RestoreFromSnapshot(config, raw_state, pending_random_event);
  if (value.contains("rng_state")) {
    SetRNGState(value.at("rng_state").get<std::string>());
  }
  return state;
}

std::string EclipseGame::GetRNGState() const {
  std::ostringstream stream;
  stream << rng_;
  return stream.str();
}

void EclipseGame::SetRNGState(const std::string& rng_state) const {
  if (rng_state.empty()) {
    return;
  }
  std::istringstream stream(rng_state);
  stream >> rng_;
}

SetupConfig EclipseGame::InitialSetupConfig() const {
  SetupConfig config;
  config.players = static_cast<uint8_t>(GetPlayersParam());
  config.rng_seed = GetRngSeedParam();
  config.npc_difficulty =
      nlohmann::json(ParameterValue<std::string>("npc_difficulty")).get<NPCDifficulty>();
  config.warped_universe = GetWarpedUniverseParam();
  config.staged_players.resize(config.players);
  for (int player = 0; player < config.players; ++player) {
    config.staged_players[player].species =
        nlohmann::json(ParameterValue<std::string>("species_p" + std::to_string(player))).get<Species>();
    config.staged_players[player].is_ai = false;
  }
  return NormalizeSetupConfig(config);
}

EclipseState::EclipseState(std::shared_ptr<const Game> game)
    : State(game),
      eclipse_game_(std::static_pointer_cast<const EclipseGame>(game)),
      eclipse_state_(InitializeDeterministicSetupState(
          eclipse_game_->InitialSetupConfig())),
      setup_config_(eclipse_game_->InitialSetupConfig()) {}

std::unique_ptr<State> EclipseState::Clone() const {
  // Sampled-stochastic games share the same Game (and its RNG) across
  // cloned states. This is OpenSpiel's accepted trade-off: sibling branches
  // see the same RNG state. See tarok.cc and stones_and_gems.cc.
  return std::unique_ptr<State>(new EclipseState(*this));
}

Player EclipseState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (pending_random_event_ != PendingRandomEvent::none) {
    return kChancePlayerId;
  }
  if (eclipse_state_.upkeep_state.step != UpkeepState::Step::inactive) {
    return eclipse_state_.upkeep_state.player_id;
  }
  if (eclipse_state_.current_phase == RoundPhase::COMBAT &&
      eclipse_state_.combat_state.phase != CombatState::Phase::inactive) {
    if (eclipse_state_.combat_state.pending_player != kNoPlayer) {
      return eclipse_state_.combat_state.pending_player;
    }
    if (eclipse_state_.combat_state.pending_target_group_player != kNoPlayer &&
        eclipse_state_.combat_state.pending_die_index <
            eclipse_state_.combat_state.pending_die_count) {
      return eclipse_state_.combat_state.pending_target_group_player;
    }
    if (eclipse_state_.combat_state.tile_select_player != kNoPlayer) {
      return eclipse_state_.combat_state.tile_select_player;
    }
    if (eclipse_state_.combat_state.influence_decision_player != kNoPlayer) {
      return eclipse_state_.combat_state.influence_decision_player;
    }
    if (eclipse_state_.combat_state.discovery_decision_player != kNoPlayer) {
      return eclipse_state_.combat_state.discovery_decision_player;
    }
  }

  // Diplomacy sub-states: return the player who needs to act.
  if (eclipse_state_.diplomacy_state.phase != DiplomacyState::Phase::inactive) {
    const auto& ds = eclipse_state_.diplomacy_state;
    switch (ds.phase) {
      case DiplomacyState::Phase::choose_accept:
        // Partner must accept or decline.
        if (ds.partner_id < eclipse_state_.players.size()) {
          return ds.partner_id;
        }
        break;
      case DiplomacyState::Phase::choose_rearrange:
        if (ds.rearrange_side == 0 && ds.player_id < eclipse_state_.players.size()) {
          return ds.player_id;
        } else if (ds.rearrange_side == 1 && ds.partner_id < eclipse_state_.players.size()) {
          return ds.partner_id;
        }
        break;
      case DiplomacyState::Phase::choose_pop_track:
        if (ds.pop_track_side == 0 && ds.player_id < eclipse_state_.players.size()) {
          return ds.player_id;
        } else if (ds.pop_track_side == 1 && ds.partner_id < eclipse_state_.players.size()) {
          return ds.partner_id;
        }
        break;
      case DiplomacyState::Phase::choose_return_track: {
        uint8_t pending = pending_return_track_player(eclipse_state_);
        if (pending < eclipse_state_.players.size()) {
          return pending;
        }
        break;
      }
      default:
        break;
    }
  }

  return eclipse_state_.current_player;
}

std::vector<Action> EclipseState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }
  if (pending_random_event_ != PendingRandomEvent::none) {
    // Chance node: the legal actions are exactly the chance outcomes.
    std::vector<Action> actions;
    for (const auto& [outcome, prob] : ChanceOutcomes()) {
      actions.push_back(outcome);
    }
    return actions;
  }

  const ::State& s = eclipse_state_;

  // Diplomacy responses and Minor Species track choices must resolve before an
  // interrupted Action or Reaction resumes.
  if (s.diplomacy_state.phase != DiplomacyState::Phase::inactive) {
    return DiplomacyLegalActions();
  }
  if (s.minor_species_pending_track != 255) {
    const ::Player& p = s.players[s.minor_species_pending_track];
    std::vector<Action> track_actions;
    if (p.resources.gold_prod > 0)
        track_actions.push_back(action_minor_species_track_start + 0);
    if (p.resources.science_prod > 0)
        track_actions.push_back(action_minor_species_track_start + 1);
    if (p.resources.materials_prod > 0)
        track_actions.push_back(action_minor_species_track_start + 2);
    return track_actions;
  }

  const auto add_bonuses = [&](uint8_t player_id, std::vector<Action> actions) {
    if (s.current_phase == RoundPhase::ACTION) {
      AppendActionPhaseBonusActions(s, player_id, &actions);
    }
    std::sort(actions.begin(), actions.end());
    return actions;
  };

  if (s.explore_state.phase != ExplorePhase::inactive) {
    return ExploreLegalActions();
  }
  if (s.research_state.phase != ::ResearchState::Phase::inactive) {
    return ResearchLegalActions();
  }
  if (s.influence_state.phase != ::InfluenceState::Phase::inactive) {
    return InfluenceLegalActions();
  }
  if (s.build_state.phase != ::BuildState::Phase::inactive) {
    return add_bonuses(s.build_state.player_id, BuildLegalActions());
  }
  if (s.upgrade_state.phase != ::UpgradeState::Phase::inactive) {
    return add_bonuses(s.upgrade_state.player_id, UpgradeLegalActions());
  }
  if (s.move_state.phase != ::MoveState::Phase::inactive) {
    if (s.move_state.phase == ::MoveState::Phase::choose_move) {
      return add_bonuses(s.move_state.player_id, MoveLegalActions());
    }
    return MoveLegalActions();
  }

  // Artifact Key resource-choice sub-action: pick a resource type per chunk.
  // Intercepts before action phase / upkeep / combat so the player resolves
  // pending chunks before continuing.
  for (const ::Player& p : s.players) {
    if (p.pending_artifact_key_chunks > 0) {
      std::vector<Action> chunk_actions;
      chunk_actions.push_back(action_artifact_key_track_start + 0); // Money
      chunk_actions.push_back(action_artifact_key_track_start + 1); // Science
      chunk_actions.push_back(action_artifact_key_track_start + 2); // Materials
      return chunk_actions;
    }
  }

  if (s.upkeep_state.step != UpkeepState::Step::inactive) {
    return UpkeepLegalActions();
  }

  if (s.current_phase == RoundPhase::COMBAT &&
      s.combat_state.phase != CombatState::Phase::inactive) {
    return CombatLegalActions();
  }

  std::vector<Action> actions;

  if (s.current_phase != RoundPhase::ACTION) {
    return actions;
  }

  actions.push_back(action_pass);

  uint8_t current_player = eclipse_state_.current_player;
  if (current_player >= eclipse_state_.players.size()) {
    SPIEL_CHECK_TRUE(eclipse_state_.players.empty());
    return actions;
  }

  const auto& player = eclipse_state_.players[current_player];
  const bool has_action_disk = player.available_influence_discs() > 0;

  if (player.has_passed) {
    // Passed players may only take Reaction actions (1 Activation each, no tech
    // bonuses). Reactions do not consume an action-track influence disc — they
    // use the Reaction Track instead.
    if (has_action_disk) {
      if (HasLegalUpgradeChoice(s, current_player)) {
        actions.push_back(action_reaction_upgrade);
      }
      if (HasLegalBuildChoice(s, current_player)) {
        actions.push_back(action_reaction_build);
      }
      if (HasLegalMoveChoice(s, current_player)) {
        actions.push_back(action_reaction_move);
      }
    }
  } else {
    if (has_action_disk && player.resources.science >= 2 && HasLegalResearchChoice(s, current_player)) {
      actions.push_back(action_research);
    }
    if (has_action_disk && HasLegalBuildChoice(s, current_player)) {
      actions.push_back(action_build);
    }
    if (has_action_disk && has_explore_zone(s, current_player)) {
      actions.push_back(action_explore);
    }
    if (has_action_disk && HasLegalInfluenceChoice(s, current_player)) {
      actions.push_back(action_influence_start);
    }
    if (has_action_disk && HasLegalUpgradeChoice(s, current_player)) {
      actions.push_back(action_upgrade);
    }
    if (has_action_disk) {
      actions.push_back(action_move);
    }

  }

  AppendActionPhaseBonusActions(s, current_player, &actions);

  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> EclipseState::ExploreLegalActions() const {
  const ::State& s = eclipse_state_;
  const ExploreState& es = s.explore_state;
  std::vector<Action> actions;

  switch (es.phase) {
    case ExplorePhase::choose_zone: {
      // One stable action id per galaxy hex; no truncation regardless of how
      // many legal zones exist. Plus the option to stop exploring.
      for (const HexCoord& zone : legal_explore_zones(s, es.player_id)) {
        actions.push_back(explore_zone_start + hex_to_index(zone.q, zone.r));
      }
      actions.push_back(explore_stop);
      break;
    }
    case ExplorePhase::draw_again_decision: {
      actions.push_back(explore_draw_again);
      actions.push_back(explore_skip_second);
      break;
    }
    case ExplorePhase::select_drawn_tile: {
      for (uint8_t i = 0; i < es.drawn_count && i < 2; ++i) {
        actions.push_back(explore_select_tile_start + i);
      }
      break;
    }
    case ExplorePhase::place_or_discard: {
      if (!legal_explore_rotations(s, es.player_id).empty()) {
        actions.push_back(explore_place);
      }
      actions.push_back(explore_discard);
      break;
    }
    case ExplorePhase::choose_rotation: {
      for (uint8_t rot : legal_explore_rotations(s, es.player_id)) {
        actions.push_back(explore_rotation_start + rot);
      }
      break;
    }
    case ExplorePhase::claim_control: {
      const SectorDefinition* def = get_sector_definition(es.selected_sector_id);
      bool has_ancients = def != nullptr && def->starting_ancients > 0;
      bool draco = s.players[es.player_id].species_id ==
                   Species::DESCENDANTS_OF_DRACO;
      bool may_control = !has_ancients || draco;
      if (may_control &&
          s.players[es.player_id].available_influence_discs() > 0) {
        actions.push_back(explore_claim_yes);
      }
      actions.push_back(explore_claim_no);
      break;
    }
    case ExplorePhase::discovery_reward: {
      actions.push_back(explore_discovery_reward);
      actions.push_back(explore_discovery_vp);
      break;
    }
    case ExplorePhase::discovery_upgrade: {
      const uint8_t player_id = es.player_id;
      const ShipPartId part_id = static_cast<ShipPartId>(es.discovered_part);

      for (int ship = 0; ship < UPGRADE_SHIP_COUNT; ++ship) {
        const ShipType ship_type = static_cast<ShipType>(ship);
        const ::Player& player = s.players[player_id];
        const Blueprint& bp = player.blueprints[ship];
        for (int slot = 0; slot < bp.capacity; ++slot) {
          if (can_upgrade(s, player_id, ship_type, static_cast<uint8_t>(slot), part_id, /*is_free_immediate=*/true)) {
            Action action = action_upgrade_choice_start + ship * UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT + slot * UPGRADE_PART_COUNT + static_cast<int>(part_id);
            actions.push_back(action);
          }
        }
      }
      actions.push_back(action_upgrade_stop);
      break;
    }
    default:
      break;
  }
  // OpenSpiel requires LegalActions sorted ascending; zone ids are emitted in
  // grid-discovery order and explore_stop trails the zone block numerically.
  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> EclipseState::ResearchLegalActions() const {
  const ::State& s = eclipse_state_;
  const ::ResearchState& rs = s.research_state;
  std::vector<Action> actions;

  if (rs.phase != ::ResearchState::Phase::choose_tech) {
    return actions;
  }

  const ::Player& player = s.players[rs.player_id];

  // Standard techs (indices 0 to TECH_TABLE_SIZE-1 of TECH_TABLE)
  for (size_t i = 0; i < TECH_TABLE_SIZE; ++i) {
    const TechDefinition& def = TECH_TABLE[i];
    if (def.category == TechCategory::RARE) continue; // Skip rare techs here

    // Check if already owned
    if (player.has_tech(def.bit)) continue;

    // Check availability in tech tray
    if (s.get_tech_tray_count(def.bit) == 0) continue;

    // Check track capacity
    if (get_track_tile_count(player, def.category) >= 8) continue;

    // Check affordability
    uint8_t cost = calculate_research_cost(player, def, def.category);
    if (player.resources.science < cost) continue;

    actions.push_back(action_research_standard_start + i);
  }

  // Rare techs (TECH_RARE_COUNT rare techs * 3 tracks = 48 indices)
  // Rare techs start at index TECH_TABLE_SIZE in TECH_TABLE
  for (size_t rare_idx = 0; rare_idx < TECH_RARE_COUNT; ++rare_idx) {
    const TechDefinition& def = TECH_TABLE[TECH_TABLE_SIZE + rare_idx];
    if (def.category != TechCategory::RARE) continue;

    // Check if already owned
    if (player.has_tech(def.bit)) continue;

    // Check availability in tech tray
    if (s.get_tech_tray_count(def.bit) == 0) continue;

    // For each track, check if placement is legal
    for (int track = 0; track < 3; ++track) {
      TechCategory target_track = static_cast<TechCategory>(track);

      // Check track capacity
      if (get_track_tile_count(player, target_track) >= 8) continue;

      // Check affordability
      uint8_t cost = calculate_research_cost(player, def, target_track);
      if (player.resources.science < cost) continue;

      actions.push_back(action_research_rare_start + rare_idx * 3 + track);
    }
  }

  // Option to stop early (for multi-activation species)
  actions.push_back(action_research_stop);

  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> EclipseState::InfluenceLegalActions() const {
  const ::State& s = eclipse_state_;
  const ::InfluenceState& is = s.influence_state;
  std::vector<Action> actions;

  if (is.phase == ::InfluenceState::Phase::choose_influence) {
    // 1. Placement of influence disc (influence to cell)
    const InfluenceShipMap ship_map = BuildInfluenceShipMap(s, is.player_id);
    for (int cell = 0; cell < GALAXY_CELL_COUNT; ++cell) {
      if (can_influence_to_sector(s, is.player_id, static_cast<uint8_t>(cell), &ship_map)) {
        actions.push_back(action_influence_to_cell_start + cell);
      }
    }
    // 2. Reclamation of influence disc (reclaim from cell)
    for (int cell = 0; cell < GALAXY_CELL_COUNT; ++cell) {
      if (can_reclaim_from_sector(s, is.player_id, static_cast<uint8_t>(cell))) {
        actions.push_back(action_reclaim_from_cell_start + cell);
      }
    }
    // 3. Stop/Finish
    actions.push_back(action_influence_stop);
  } else if (is.phase == ::InfluenceState::Phase::choose_return_track) {
    // Choice of return track (0=Money, 1=Science, 2=Materials)
    for (uint8_t track : get_legal_return_tracks_for_current_pending(s)) {
      actions.push_back(action_choose_return_track_start + track);
    }
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> EclipseState::BuildLegalActions() const {
  const ::State& s = eclipse_state_;
  const ::BuildState& bs = s.build_state;
  std::vector<Action> actions;

  if (bs.phase == ::BuildState::Phase::choose_build) {
    const PlayerUnitCounts counts = BuildUnitCounts(s, bs.player_id);
    const std::vector<uint8_t> owned = PlayerOwnedBuildCells(s, bs.player_id);
    for (int t = 0; t < BUILD_TYPE_COUNT; ++t) {
      const BuildType type = static_cast<BuildType>(t);
      for (uint8_t cell : owned) {
        if (can_build(s, bs.player_id, type, cell, &counts)) {
          actions.push_back(action_build_choice_start + t * GALAXY_CELL_COUNT + cell);
        }
      }
    }
    actions.push_back(action_build_stop);
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> EclipseState::UpgradeLegalActions() const {
  const ::State& s = eclipse_state_;
  const ::UpgradeState& us = s.upgrade_state;
  std::vector<Action> actions;

  if (us.phase == ::UpgradeState::Phase::choose_upgrade) {
    const ::Player& player = s.players[us.player_id];
    // Only probe parts the player can place anywhere.
    const std::vector<ShipPartId> parts = PlaceablePartIds(s, us.player_id);

    // 4 ship types (INTERCEPTOR=0, CRUISER=1, DREADNOUGHT=2, STARBASE=3)
    for (int ship = 0; ship < UPGRADE_SHIP_COUNT; ++ship) {
      ShipType ship_type = static_cast<ShipType>(ship);
      const Blueprint& bp = player.blueprints[ship];

      for (int slot = 0; slot < UPGRADE_SLOTS_PER_SHIP; ++slot) {
        if (slot >= bp.capacity) continue;

        // Check if removal is legal (energy-positive, retains drive)
        if (can_upgrade(s, us.player_id, ship_type, static_cast<uint8_t>(slot), ShipPartId::NONE)) {
          Action action = action_upgrade_choice_start + ship * UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT + slot * UPGRADE_PART_COUNT;
          if (action == action_upgrade_stop) {
            continue;
          }
          actions.push_back(action);
        }

        // Check all placeable ship parts (skip NONE, already handled)
        for (ShipPartId part_id : parts) {
          if (can_upgrade(s, us.player_id, ship_type, static_cast<uint8_t>(slot), part_id)) {
            Action action = action_upgrade_choice_start + ship * UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT + slot * UPGRADE_PART_COUNT + static_cast<int>(part_id);
            if (action == action_upgrade_stop) {
              continue;
            }
            actions.push_back(action);
          }
        }
      }
    }
    actions.push_back(action_upgrade_stop);
  }

  std::sort(actions.begin(), actions.end());
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
  return actions;
}

std::vector<Action> EclipseState::MoveLegalActions() const {
  const ::State& s = eclipse_state_;
  const ::MoveState& ms = s.move_state;
  std::vector<Action> actions;

  if (ms.phase == ::MoveState::Phase::choose_move) {
    const std::vector<MoveStepOption> steps = legal_move_steps(s, ms.player_id);
    for (const MoveStepOption& opt : steps) {
      actions.push_back(action_move_choice_start + opt.unit_idx * MOVE_CODES_PER_UNIT + opt.direction);
    }
    actions.push_back(action_move_stop);
  } else if (ms.phase == ::MoveState::Phase::choose_warp_destination) {
    const std::vector<uint8_t> cells = legal_warp_destination_cells(s, ms.player_id);
    for (uint8_t cell : cells) {
      actions.push_back(action_move_warp_destination_start + cell);
    }
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

std::vector<Action> EclipseState::DiplomacyLegalActions() const {
  const ::State& s = eclipse_state_;
  const DiplomacyState& ds = s.diplomacy_state;
  std::vector<Action> actions;

  if (ds.phase == DiplomacyState::Phase::choose_accept) {
    // Fix #1: partner may accept or decline (rulebook p.14).
    const uint8_t expected = ds.partner_id;
    if (expected < s.players.size() && !s.players[expected].eliminated) {
      actions.push_back(action_diplomacy_accept);
      actions.push_back(action_diplomacy_decline);
    }
  } else if (ds.phase == DiplomacyState::Phase::choose_pop_track) {
    // Proposer (side=0) or partner (side=1) picks a Pop Track for the cube
    // they're giving. The three legal tracks are gated by cube availability.
    const uint8_t side = ds.pop_track_side;
    const uint8_t expected = (side == 0) ? ds.player_id : ds.partner_id;
    if (expected < s.players.size()) {
      const ::Player& p = s.players[expected];
      if (p.resources.gold_prod > 0) {
        actions.push_back(action_diplomacy_pick_track_start + side * 3 + 0);
      }
      if (p.resources.science_prod > 0) {
        actions.push_back(action_diplomacy_pick_track_start + side * 3 + 1);
      }
      if (p.resources.materials_prod > 0) {
        actions.push_back(action_diplomacy_pick_track_start + side * 3 + 2);
      }
    }
  } else if (ds.phase == DiplomacyState::Phase::choose_rearrange) {
    // Either side may return a Rep tile to the bag to free an Ambassador slot.
    const uint8_t side = ds.rearrange_side;
    const uint8_t expected = (side == 0) ? ds.player_id : ds.partner_id;
    if (expected < s.players.size()) {
      const ::Player& p = s.players[expected];
      for (size_t slot = 0; slot < p.reputation_track.size(); ++slot) {
        if (slot_is_returnable(p.reputation_track[slot])) {
          actions.push_back(action_diplomacy_return_start + side * 5 +
                            static_cast<uint8_t>(slot));
        }
      }
      // Swaps: AMBASSADOR_OR_REP <-> free REP_ONLY.
      for (size_t src = 0; src < p.reputation_track.size(); ++src) {
        if (p.reputation_track[src].kind != ReputationSlotKind::AMBASSADOR_OR_REP) continue;
        if (p.reputation_track[src].holds_ambassador) continue;
        for (size_t dst = 0; dst < p.reputation_track.size(); ++dst) {
          if (src == dst) continue;
          if (p.reputation_track[dst].kind != ReputationSlotKind::REP_ONLY) continue;
          if (p.reputation_track[dst].holds_ambassador) continue;
          actions.push_back(action_diplomacy_swap_start + side * 20 +
                            static_cast<uint8_t>(src) * 5 +
                            static_cast<uint8_t>(dst));
        }
      }
    }
  } else if (ds.phase == DiplomacyState::Phase::choose_return_track) {
    // The player whose return-track choice is pending picks a track with room.
    const uint8_t expected = pending_return_track_player(s);
    if (expected < s.players.size()) {
      const ::Player& p = s.players[expected];
      if (p.resources.gold_prod < 12) actions.push_back(action_choose_return_track_start + 0);
      if (p.resources.science_prod < 12) actions.push_back(action_choose_return_track_start + 1);
      if (p.resources.materials_prod < 12) actions.push_back(action_choose_return_track_start + 2);
    }
  }
  return actions;
}

std::vector<Action> EclipseState::UpkeepLegalActions() const {
  const ::State& s = eclipse_state_;
  const UpkeepState& us = s.upkeep_state;
  std::vector<Action> actions;
  if (us.player_id >= s.players.size()) return actions;

  const ::Player& player = s.players[us.player_id];
  switch (us.step) {
    case UpkeepState::Step::colony_ships: {
      AppendLegalUpkeepColonyShipActions(s, us.player_id, actions);
      actions.push_back(action_upkeep_colony_done);
      break;
    }
    case UpkeepState::Step::bankruptcy: {
      if (IsPlayerSolvent(player)) {
        actions.push_back(action_upkeep_pay_done);
        break;
      }
      for (int conv = 0; conv < TRADE_CONVERSION_COUNT; ++conv) {
        const TradeConversion trade = static_cast<TradeConversion>(conv);
        if ((trade == TradeConversion::SCIENCE_TO_GOLD ||
             trade == TradeConversion::MATERIALS_TO_GOLD) &&
            can_trade(player, trade)) {
          actions.push_back(action_trade_start + conv);
        }
      }
      AppendReclaimActions(s, us.player_id, actions);
      break;
    }
    case UpkeepState::Step::choose_return_track: {
      if (!us.pending_returns.empty()) {
        for (PopTrack track : get_legal_return_tracks(
                 player, us.pending_returns.front().type,
                 us.pending_returns.front().is_orbital)) {
          actions.push_back(action_choose_return_track_start +
                            static_cast<int>(track));
        }
      }
      break;
    }
    case UpkeepState::Step::cleanup_graveyards:
    case UpkeepState::Step::inactive:
      break;
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

std::string EclipseState::ActionToString(Player player, Action action_id) const {
  if (pending_random_event_ == PendingRandomEvent::explore_draw) {
    uint32_t bag = ring_bag_value(eclipse_state_, eclipse_state_.explore_state.ring);
    if (bag == 0) return "EXPLORE_DRAW_EMPTY";
    uint16_t sector_id = ring_bit_to_sector_id(
        eclipse_state_.explore_state.ring, static_cast<uint8_t>(action_id));
    if (sector_id == 0) {
      return "EXPLORE_DRAW_BIT_" + std::to_string(action_id);
    }
    return "EXPLORE_DRAW_SECTOR_" + std::to_string(sector_id);
  }
  if (pending_random_event_ == PendingRandomEvent::combat_roll) {
    return "COMBAT_ROLL_" + std::to_string(action_id);  // die face 1-6
  }
  if (pending_random_event_ == PendingRandomEvent::reputation_draw) {
    return "REP_DRAW_" + std::to_string(action_id);  // tile value enum 0-3
  }
  if (pending_random_event_ != PendingRandomEvent::none) {
    return "RESOLVE_" + PendingRandomEventToString(pending_random_event_);
  }
  if (action_id == action_reaction_upgrade) {
    return "REACTION_UPGRADE";
  }
  if (action_id == action_reaction_build) {
    return "REACTION_BUILD";
  }
  if (action_id == action_reaction_move) {
    return "REACTION_MOVE";
  }
  if (action_id == action_pass) {
    return "PASS";
  }
  if (action_id == action_research) {
    return "RESEARCH";
  }
  if (action_id == action_research_stop) {
    return "RESEARCH_STOP";
  }
  if (action_id >= action_research_standard_start && action_id < action_research_rare_start) {
    size_t tech_idx = action_id - action_research_standard_start;
    if (tech_idx < TECH_TABLE_SIZE) {
      return "RESEARCH_" + std::string(TECH_TABLE[tech_idx].name);
    }
  }
  if (action_id >= action_research_rare_start && action_id < action_build) {
    size_t offset = action_id - action_research_rare_start;
    size_t rare_idx = offset / 3;
    size_t track = offset % 3;
    if (rare_idx < TECH_RARE_COUNT) {
      const TechDefinition& def = TECH_TABLE[TECH_TABLE_SIZE + rare_idx];
      static const char* kTrackNames[3] = {"MILITARY", "GRID", "NANO"};
      return "RESEARCH_" + std::string(def.name) + "_ON_" + kTrackNames[track];
    }
  }
  if (action_id == action_explore) {
    return "EXPLORE";
  }
  if (action_id == action_build) {
    return "BUILD";
  }
  if (action_id == action_build_stop) {
    return "BUILD_STOP";
  }
  if (action_id >= action_build_choice_start && action_id < action_build_end) {
    int encoded = action_id - action_build_choice_start;
    int type_idx = encoded / GALAXY_CELL_COUNT;
    int cell_idx = encoded % GALAXY_CELL_COUNT;
    static const char* kBuildTypeNames[BUILD_TYPE_COUNT] = {
        "INTERCEPTOR", "CRUISER", "DREADNOUGHT", "STARBASE", "ORBITAL", "MONOLITH"
    };
    HexCoord c = index_to_hex(cell_idx);
    return "BUILD_" + std::string(kBuildTypeNames[type_idx]) + "_" + std::to_string(c.q) + "_" + std::to_string(c.r);
  }
  if (action_id == explore_place) return "EXPLORE_PLACE";
  if (action_id == explore_discard) return "EXPLORE_DISCARD";
  if (action_id >= explore_rotation_start &&
      action_id < explore_rotation_start + 6) {
    return "EXPLORE_ROT_" + std::to_string(action_id - explore_rotation_start);
  }
  if (action_id == explore_claim_yes) return "EXPLORE_CLAIM_YES";
  if (action_id == explore_claim_no) return "EXPLORE_CLAIM_NO";
  if (action_id == explore_discovery_reward) return "EXPLORE_DISCOVERY_REWARD";
  if (action_id == explore_discovery_vp) return "EXPLORE_DISCOVERY_VP";
  if (action_id >= explore_select_tile_start &&
      action_id < explore_select_tile_start + 2) {
    return "EXPLORE_SELECT_TILE_" +
           std::to_string(action_id - explore_select_tile_start);
  }
  if (action_id == explore_draw_again) return "EXPLORE_DRAW_AGAIN";
  if (action_id == explore_skip_second) return "EXPLORE_SKIP_SECOND";
  if (action_id == explore_stop) return "EXPLORE_STOP";
  if (action_id >= explore_zone_start &&
      action_id < explore_zone_start + GALAXY_CELL_COUNT) {
    HexCoord zone = index_to_hex(action_id - explore_zone_start);
    return "EXPLORE_ZONE_" + std::to_string(zone.q) + "_" +
           std::to_string(zone.r);
  }
  static const char* kTradeNames[TRADE_CONVERSION_COUNT] = {
      "TRADE_GOLD_TO_SCIENCE", "TRADE_GOLD_TO_MATERIALS",
      "TRADE_SCIENCE_TO_GOLD", "TRADE_SCIENCE_TO_MATERIALS",
      "TRADE_MATERIALS_TO_GOLD", "TRADE_MATERIALS_TO_SCIENCE",
  };
  if (action_id >= action_trade_start &&
      action_id < action_colony_ship_start) {
    return kTradeNames[action_id - action_trade_start];
  }
  if (action_id >= action_colony_ship_start && action_id < action_influence_start) {
    int encoded = static_cast<int>(action_id - action_colony_ship_start);
    int cell = encoded / COLONY_SHIP_CODES_PER_CELL;
    int rem  = encoded % COLONY_SHIP_CODES_PER_CELL;
    int slot = rem / COLONY_SHIP_TRACKS;
    int track = rem % COLONY_SHIP_TRACKS;
    static const char* kTrackNames[COLONY_SHIP_TRACKS] = {"MONEY", "SCIENCE",
                                                          "MATERIALS"};
    HexCoord c = index_to_hex(cell);
    return "COLONY_SHIP_" + std::to_string(c.q) + "_" +
           std::to_string(c.r) + "_SLOT" + std::to_string(slot) + "_" +
           kTrackNames[track];
  }
  if (action_id == action_influence_start) {
    return "INFLUENCE";
  }
  if (action_id == action_influence_stop) {
    return "INFLUENCE_STOP";
  }
  if (action_id >= action_influence_to_cell_start && action_id < action_reclaim_from_cell_start) {
    HexCoord zone = index_to_hex(action_id - action_influence_to_cell_start);
    return "INFLUENCE_TO_" + std::to_string(zone.q) + "_" + std::to_string(zone.r);
  }
  if (action_id >= action_reclaim_from_cell_start && action_id < action_choose_return_track_start) {
    HexCoord zone = index_to_hex(action_id - action_reclaim_from_cell_start);
    return "RECLAIM_FROM_" + std::to_string(zone.q) + "_" + std::to_string(zone.r);
  }
  if (action_id >= action_choose_return_track_start && action_id < action_choose_return_track_start + 3) {
    static const char* kTrackNames[3] = {"MONEY", "SCIENCE", "MATERIALS"};
    return "RETURN_CUBE_TO_" + std::string(kTrackNames[action_id - action_choose_return_track_start]);
  }
  if (action_id == action_upgrade) {
    return "UPGRADE";
  }
  if (action_id == action_upgrade_stop) {
    if (RawState().explore_state.phase == ExplorePhase::discovery_upgrade) {
      return "EXPLORE_DISCOVERY_UPGRADE_STORE";
    }
    return "UPGRADE_STOP";
  }
  if (action_id >= action_upgrade_choice_start && action_id < action_upgrade_choice_start + UPGRADE_SHIP_COUNT * UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT) {
    int encoded = action_id - action_upgrade_choice_start;
    int ship = encoded / (UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT);
    int rem = encoded % (UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT);
    int slot = rem / UPGRADE_PART_COUNT;
    int part_idx = rem % UPGRADE_PART_COUNT;
    static const char* kShipNames[UPGRADE_SHIP_COUNT] = {"INTERCEPTOR", "CRUISER", "DREADNOUGHT", "STARBASE"};
    if (part_idx == 0) {
      return "UPGRADE_" + std::string(kShipNames[ship]) + "_SLOT" + std::to_string(slot) + "_REMOVE";
    }
    if (part_idx > 0 && part_idx - 1 < sizeof(SHIP_PART_TABLE) / sizeof(SHIP_PART_TABLE[0])) {
      const ShipPart& part = SHIP_PART_TABLE[part_idx - 1];
      return "UPGRADE_" + std::string(kShipNames[ship]) + "_SLOT" + std::to_string(slot) + "_" + std::string(part.name);
    }
    return "UPGRADE_" + std::string(kShipNames[ship]) + "_SLOT" + std::to_string(slot) + "_PART" + std::to_string(part_idx);
  }
  if (action_id == action_move) {
    return "MOVE";
  }
  if (action_id == action_move_stop) {
    return "MOVE_STOP";
  }
  if (action_id >= action_move_choice_start && action_id < action_move_warp_start) {
    int encoded = action_id - action_move_choice_start;
    int unit_idx = encoded / MOVE_CODES_PER_UNIT;
    int direction = encoded % MOVE_CODES_PER_UNIT;
    if (direction == MOVE_WARP_DIRECTION) {
      return "MOVE_UNIT_" + std::to_string(unit_idx) + "_WARP";
    }
    static const char* kDirNames[6] = {"E", "NE", "NW", "W", "SW", "SE"};
    return "MOVE_UNIT_" + std::to_string(unit_idx) + "_" + kDirNames[direction];
  }
  if (action_id >= action_move_warp_destination_start &&
      action_id < action_upkeep_colony_done) {
    HexCoord c = index_to_hex(action_id - action_move_warp_destination_start);
    return "MOVE_WARP_TO_" + std::to_string(c.q) + "_" + std::to_string(c.r);
  }
  if (action_id == action_upkeep_colony_done) {
    return "UPKEEP_COLONY_DONE";
  }
  if (action_id == action_upkeep_pay_done) {
    return "UPKEEP_PAY_DONE";
  }
  if (action_id == action_combat_continue) {
    return "COMBAT_CONTINUE";
  }
  if (action_id == action_combat_attack) {
    return "COMBAT_ATTACK";
  }
  if (action_id >= action_combat_retreat_to_cell_start &&
      action_id < action_combat_dice_target_start) {
    HexCoord c = index_to_hex(action_id - action_combat_retreat_to_cell_start);
    return "COMBAT_RETREAT_TO_" + std::to_string(c.q) + "_" +
           std::to_string(c.r);
  }
  if (action_id >= action_combat_dice_target_start &&
      action_id < action_combat_rep_select_start) {
    return "COMBAT_TARGET_UNIT_" +
           std::to_string(action_id - action_combat_dice_target_start);
  }
  if (action_id >= action_combat_rep_select_start &&
      action_id < action_combat_rep_skip) {
    return "COMBAT_REPUTATION_SELECT_" +
           std::to_string(action_id - action_combat_rep_select_start);
  }
  if (action_id == action_combat_rep_skip) {
    return "COMBAT_REPUTATION_SKIP";
  }
  if (action_id >= action_combat_pop_target_start &&
      action_id < action_combat_influence_yes) {
    return "COMBAT_POP_TARGET_" +
           std::to_string(action_id - action_combat_pop_target_start);
  }
  if (action_id == action_combat_influence_yes) {
    return "COMBAT_INFLUENCE_YES";
  }
  if (action_id == action_combat_influence_no) {
    return "COMBAT_INFLUENCE_NO";
  }
  if (action_id == action_combat_discovery_reward) {
    return "COMBAT_DISCOVERY_REWARD";
  }
  if (action_id == action_combat_discovery_vp) {
    return "COMBAT_DISCOVERY_VP";
  }
  if (action_id >= action_combat_ship_order_start &&
      action_id < action_combat_ship_order_start + 4) {
    static const char* kShipTypeNames[4] = {
        "INTERCEPTOR", "CRUISER", "DREADNOUGHT", "STARBASE"};
    return "COMBAT_SHIP_ORDER_" +
           std::string(kShipTypeNames[action_id - action_combat_ship_order_start]);
  }
  if (action_id >= action_combat_influence_to_cell_start &&
      action_id < action_diplomacy_propose_start) {
    HexCoord c = index_to_hex(action_id - action_combat_influence_to_cell_start);
    return "COMBAT_INFLUENCE_TO_" + std::to_string(c.q) + "_" +
           std::to_string(c.r);
  }
  if (action_id >= action_diplomacy_propose_start &&
      action_id < action_diplomacy_propose_end) {
    int encoded = action_id - action_diplomacy_propose_start;
    int p = encoded / MAX_PLAYERS;
    int q = encoded % MAX_PLAYERS;
    return "DIPLOMACY_PROPOSE_" + std::to_string(p) + "_" + std::to_string(q);
  }
  if (action_id >= action_diplomacy_pick_track_start &&
      action_id < action_diplomacy_pick_track_end) {
    int encoded = action_id - action_diplomacy_pick_track_start;
    int side = encoded / 3;
    int track = encoded % 3;
    static const char* kTrackNames[3] = {"MONEY", "SCIENCE", "MATERIALS"};
    return std::string("DIPLOMACY_PICK_TRACK_") + (side == 0 ? "PROPOSER" : "PARTNER") +
           "_" + kTrackNames[track];
  }
  if (action_id >= action_diplomacy_return_start &&
      action_id < action_diplomacy_return_end) {
    int encoded = action_id - action_diplomacy_return_start;
    int side = encoded / 5;
    int slot = encoded % 5;
    return std::string("DIPLOMACY_RETURN_REP_") + (side == 0 ? "PROPOSER" : "PARTNER") +
           "_SLOT" + std::to_string(slot);
  }
  if (action_id >= action_diplomacy_swap_start &&
      action_id < action_diplomacy_swap_end) {
    int encoded = action_id - action_diplomacy_swap_start;
    int side = encoded / 20;
    int rem = encoded % 20;
    int src = rem / 5;
    int dst = rem % 5;
    return std::string("DIPLOMACY_SWAP_REP_") + (side == 0 ? "PROPOSER" : "PARTNER") +
           "_" + std::to_string(src) + "_TO_" + std::to_string(dst);
  }
  if (action_id == action_diplomacy_accept) return "DIPLOMACY_ACCEPT";
  if (action_id == action_diplomacy_decline) return "DIPLOMACY_DECLINE";
  if (action_id >= action_minor_species_start && action_id < action_minor_species_end) {
    uint8_t ms_idx = action_id - action_minor_species_start;
    if (ms_idx < MINOR_SPECIES_COUNT) {
      return std::string("MINOR_SPECIES_FORM_") + MINOR_SPECIES_TABLE[ms_idx].name;
    }
  }
  if (action_id >= action_minor_species_track_start && action_id < action_minor_species_track_end) {
    static const char* kTrackNames[3] = {"MONEY", "SCIENCE", "MATERIALS"};
    return std::string("MINOR_SPECIES_PLACE_POP_") + kTrackNames[action_id - action_minor_species_track_start];
  }
  if (action_id >= action_artifact_key_track_start && action_id < action_artifact_key_track_end) {
    static const char* kTrackNames[3] = {"MONEY", "SCIENCE", "MATERIALS"};
    return std::string("ARTIFACT_KEY_") + kTrackNames[action_id - action_artifact_key_track_start];
  }
  return "UNKNOWN_ACTION(" + std::to_string(action_id) + ")";
}

std::string EclipseState::ToString() const {
  std::stringstream ss;
  if (pending_random_event_ != PendingRandomEvent::none) {
    ss << "Eclipse Pending Random Event: "
       << PendingRandomEventToString(pending_random_event_) << "\n";
    ss << "Configured players: " << static_cast<int>(setup_config_.players)
       << "\n";
    return ss.str();
  }

  ss << "Eclipse Game State:\n";
  ss << "Round: " << static_cast<int>(eclipse_state_.current_round) << "\n";
  ss << "Phase: " << static_cast<int>(eclipse_state_.current_phase) << "\n";
  ss << "Current Player: " << static_cast<int>(eclipse_state_.current_player)
     << "\n";
  ss << "Turn Order:";
  for (int player = 0; player < setup_config_.players; ++player) {
    ss << " " << static_cast<int>(eclipse_state_.turn_order[player]);
  }
  ss << "\nPlayers:\n";
  for (const auto& player : eclipse_state_.players) {
    ss << "  Player " << static_cast<int>(player.id)
       << " [Species: " << nlohmann::json(player.species_id).get<std::string>()
       << ", Score: " << static_cast<int>(player.score)
       << ", Money: " << static_cast<int>(player.resources.gold)
       << ", Science: " << static_cast<int>(player.resources.science)
       << ", Materials: " << static_cast<int>(player.resources.materials)
       << ", Passed: " << (player.has_passed ? "Yes" : "No") << "]\n";
  }
  return ss.str();
}

bool EclipseState::IsTerminal() const {
  return pending_random_event_ == PendingRandomEvent::none &&
         (eclipse_state_.current_round > 8 ||
          MoveNumber() >= game_->MaxGameLength());
}

std::vector<double> EclipseState::Returns() const {
  std::vector<double> returns(NumPlayers(), 0.0);
  if (!IsTerminal()) {
    return returns;
  }
  auto final_returns = open_spiel::eclipse::evaluate_final_returns(eclipse_state_);
  for (size_t i = 0; i < returns.size(); ++i) {
    returns[i] = final_returns[i];
  }
  return returns;
}

ActionsAndProbs EclipseState::ChanceOutcomes() const {
  if (pending_random_event_ == PendingRandomEvent::none) {
    return {};
  }
  if (pending_random_event_ == PendingRandomEvent::explore_draw) {
    // Flip a uniformly random tile from the chosen zone's ring bag. The chance
    // outcome action id is the bit index within that bag.
    uint32_t bag = ring_bag_value(eclipse_state_, eclipse_state_.explore_state.ring);
    if (bag == 0) {
      return {{chance_resolve, 1.0}};  // empty bag: resolved as "drew nothing"
    }
    int count = __builtin_popcount(bag);
    double prob = 1.0 / static_cast<double>(count);
    ActionsAndProbs outcomes;
    for (int bit = 0; bit < 22; ++bit) {
      if (bag & (1u << bit)) {
        outcomes.push_back({static_cast<Action>(bit), prob});
      }
    }
    return outcomes;
  }
  if (pending_random_event_ == PendingRandomEvent::combat_roll) {
    // Uniform d6 for the weapon die awaiting a roll. Action id = face (1-6).
    ActionsAndProbs outcomes;
    for (int face = 1; face <= 6; ++face) {
      outcomes.push_back({static_cast<Action>(face), 1.0 / 6.0});
    }
    return outcomes;
  }
  if (pending_random_event_ == PendingRandomEvent::reputation_draw) {
    // Draw one tile from the reputation bag. Action id = tile-value enum (0-3),
    // probability proportional to that value's multiplicity in the bag.
    const auto& bag = eclipse_state_.reputation_tiles;
    int counts[4] = {0, 0, 0, 0};
    int total = 0;
    for (size_t i = 0; i < bag.size(); ++i) {
      const int v = static_cast<int>(bag[i]);
      if (v >= 0 && v < 4) {
        ++counts[v];
        ++total;
      }
    }
    if (total == 0) return {{chance_resolve, 1.0}};
    ActionsAndProbs outcomes;
    for (int v = 0; v < 4; ++v) {
      if (counts[v] > 0) {
        outcomes.push_back({static_cast<Action>(v),
                            static_cast<double>(counts[v]) / total});
      }
    }
    return outcomes;
  }
  return {{chance_resolve, 1.0}};
}

std::string EclipseState::Serialize() const {
  nlohmann::json value = nlohmann::json::object();
  value["setup_config"] = setup_config_;
  value["state"] = eclipse_state_;
  value["pending_random_event"] = static_cast<int>(pending_random_event_);
  value["rng_state"] = eclipse_game_->GetRNGState();
  return value.dump();
}

std::string EclipseState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  return ObservationString(player);
}

std::string EclipseState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());

  std::stringstream ss;
  ss << "Observation for Player " << player << ":\n";
  if (pending_random_event_ != PendingRandomEvent::none) {
    ss << "Waiting for " << PendingRandomEventToString(pending_random_event_)
       << "\n";
    return ss.str();
  }

  const auto& me = eclipse_state_.players[player];
  ss << "My Score: " << static_cast<int>(me.score)
     << ", Money: " << static_cast<int>(me.resources.gold)
     << ", Science: " << static_cast<int>(me.resources.science)
     << ", Materials: " << static_cast<int>(me.resources.materials) << "\n";
  const UpkeepState& us = eclipse_state_.upkeep_state;
  if (us.step != UpkeepState::Step::inactive) {
    ss << "Upkeep: step=" << nlohmann::json(us.step).get<std::string>()
       << ", player=" << static_cast<int>(us.player_id)
       << ", pending_returns=" << us.pending_returns.size() << "\n";
  }
  ss << "Visible sectors owned by me or empty near me.\n";

  // The Explore sub-state is public once a tile is flipped, so report it to all.
  const ExploreState& es = eclipse_state_.explore_state;
  if (es.phase != ExplorePhase::inactive) {
    ss << "Explore: phase=" << nlohmann::json(es.phase).get<std::string>()
       << ", player=" << static_cast<int>(es.player_id)
       << ", activations_left=" << static_cast<int>(es.activations_remaining)
       << ", zone=(" << static_cast<int>(es.zone_q) << ","
       << static_cast<int>(es.zone_r) << ")"
       << ", drawn=" << static_cast<int>(es.drawn_count);
    for (uint8_t i = 0; i < es.drawn_count && i < 2; ++i) {
      ss << " " << static_cast<int>(es.drawn_sector_ids[i]);
    }
    ss << ", selected=" << static_cast<int>(es.selected_sector_id) << "\n";
  }
  return ss.str();
}

// The observation tensor's layout and encoding live in observation.{h,cpp} --
// one source of truth, mirrored by open_spiel/python/eclipse/obs_layout.py.
void EclipseState::ObservationTensor(Player player, absl::Span<float> values) const {
  // A chance node has no well-defined per-player view; rl_environment resolves
  // chance nodes before handing control to a policy, so this is only reachable
  // by code inspecting a state paused mid-resolution.
  if (pending_random_event_ != PendingRandomEvent::none) {
    std::fill(values.begin(), values.end(), 0.0f);
    return;
  }
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  // No fill here: WriteObservationTensor zero-fills the whole span itself
  // (observation.cpp). Filling in both places cost a redundant 147 KB memset on
  // every observation -- ~3.5 us against a 13 us write, on the hottest path in
  // training.
  open_spiel::eclipse::obs::WriteObservationTensor(eclipse_state_, player,
                                                  NumPlayers(), values);
}

void EclipseState::RestoreFromSnapshot(
    const SetupConfig& config, const ::State& state,
    PendingRandomEvent pending_random_event) {
  setup_config_ = config;
  eclipse_state_ = state;
  pending_random_event_ = pending_random_event;
}

void EclipseState::ResolveChanceEvent(Action action_id) {
  switch (pending_random_event_) {
    case PendingRandomEvent::initial_setup: {
      SPIEL_CHECK_EQ(action_id, chance_resolve);
      // Per-episode setup randomization (opt-in): draw races, NPC difficulty,
      // and the warped-universe flag from the game RNG into a local copy, and
      // record the draw back into setup_config_ so serialization and the
      // information string reflect the actual episode config.
      SetupConfig cfg = setup_config_;
      if (eclipse_game_->GetRandomizeRacesParam() ||
          eclipse_game_->GetRandomizeNpcDifficultyParam() ||
          eclipse_game_->GetRandomizeWarpedParam()) {
        RandomizeSetupForEpisode(
            eclipse_game_->rng(), cfg, eclipse_game_->GetRaceAlienProbParam(),
            eclipse_game_->GetRandomizeNpcDifficultyParam(),
            eclipse_game_->GetRandomizeWarpedParam(),
            eclipse_game_->GetWarpedProbParam());
        setup_config_ = cfg;
      }
      eclipse_state_ = InitializeDeterministicSetupState(cfg);
      ResolveInitialSetupRandomness(eclipse_game_->rng(), cfg,
                                    eclipse_state_);

      std::vector<PlayerConfig> player_choices;
      player_choices.reserve(cfg.players);
      for (const StagedPlayerConfig& staged_player :
           cfg.staged_players) {
        player_choices.push_back(PlayerConfig{
            .species = staged_player.species,
            .is_ai = staged_player.is_ai,
        });
      }
      FinalizeGameSetup(eclipse_state_, player_choices);
      pending_random_event_ = PendingRandomEvent::none;
      return;
    }
    case PendingRandomEvent::explore_draw: {
      // action_id is the drawn bag bit (or chance_resolve if the bag was empty).
      apply_explore_draw(eclipse_state_, static_cast<uint8_t>(action_id));
      // Caller (DoApplyAction) inspects explore_state.phase to re-arm or finish.
      return;
    }
    case PendingRandomEvent::combat_roll: {
      // action_id is the d6 face (1-6) for the die awaiting a roll.
      ResolveCombatDie(eclipse_state_, static_cast<uint8_t>(action_id));
      return;
    }
    case PendingRandomEvent::reputation_draw: {
      // action_id is the drawn tile value enum (0-3), or chance_resolve if the
      // bag was empty.
      CombatState& cs = eclipse_state_.combat_state;
      if (eclipse_state_.reputation_tiles.empty()) {
        // Bag exhausted: stop drawing for this participant so the state machine
        // proceeds to selection (or to the next participant if nothing drawn).
        cs.rep_draw_target = cs.drawn_tiles_size;
        if (cs.drawn_tiles_size == 0) cs.tile_select_player = kNoPlayer;
        return;
      }
      DrawOneReputationTile(eclipse_state_,
                            static_cast<ReputationTiles>(action_id));
      return;
    }
    case PendingRandomEvent::none:
      SpielFatalError("no pending random event to resolve");
  }
}

void EclipseState::ApplyExploreSubAction(Action action_id) {
  ::State& s = eclipse_state_;
  const uint8_t player = s.explore_state.player_id;
  switch (s.explore_state.phase) {
    case ExplorePhase::choose_zone:
      if (action_id == explore_stop) {
        stop_exploring(s);
      } else {
        choose_explore_zone(s, player,
                            index_to_hex(action_id - explore_zone_start));
      }
      break;
    case ExplorePhase::draw_again_decision:
      if (action_id == explore_draw_again) {
        draw_again(s, player);
      } else {
        skip_second_draw(s, player);
      }
      break;
    case ExplorePhase::select_drawn_tile:
      select_drawn_tile(
          s, player,
          static_cast<uint8_t>(action_id - explore_select_tile_start));
      break;
    case ExplorePhase::place_or_discard:
      if (action_id == explore_place) {
        place_drawn_tile(s, player);
      } else {
        discard_drawn_tile(s, player);
      }
      break;
    case ExplorePhase::choose_rotation:
      apply_explore_rotation(
          s, player, static_cast<uint8_t>(action_id - explore_rotation_start));
      break;
    case ExplorePhase::claim_control:
      claim_explore_control(s, player, action_id == explore_claim_yes);
      break;
    case ExplorePhase::discovery_reward:
      resolve_explore_discovery(s, player,
                                action_id == explore_discovery_reward);
      break;
    case ExplorePhase::discovery_upgrade: {
      ExploreState& es = s.explore_state;
      if (action_id == action_upgrade_stop) {
        // Store for later
        s.players[es.player_id].parts_inventory.push_back(static_cast<ShipPartId>(es.discovered_part));
        end_explore_activation(s);
      } else if (action_id >= action_upgrade_choice_start && action_id < action_upgrade_choice_start + UPGRADE_SHIP_COUNT * UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT) {
        int encoded = action_id - action_upgrade_choice_start;
        int ship = encoded / (UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT);
        int rem = encoded % (UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT);
        int slot = rem / UPGRADE_PART_COUNT;
        int part_idx = rem % UPGRADE_PART_COUNT;

        ShipPartId part_id = static_cast<ShipPartId>(part_idx);
        execute_upgrade(s, es.player_id, static_cast<ShipType>(ship), static_cast<uint8_t>(slot), part_id, /*is_free_immediate=*/true);
        end_explore_activation(s);
      }
      break;
    }
    default:
      break;
  }
}

void EclipseState::ApplyResearchSubAction(Action action_id) {
  ::State& s = eclipse_state_;
  const uint8_t player = s.research_state.player_id;

  if (action_id == action_research_stop) {
    // Stop early, don't use remaining activations
    s.research_state.phase = ::ResearchState::Phase::inactive;
    s.research_state.player_id = 255;
    s.research_state.activations_remaining = 0;
    return;
  }

  const TechDefinition* tech_def = nullptr;
  TechCategory target_track;

  if (action_id >= action_research_standard_start && action_id < action_research_rare_start) {
    // Standard tech
    size_t tech_idx = action_id - action_research_standard_start;
    if (tech_idx >= TECH_TABLE_SIZE) return; // Invalid index
    tech_def = &TECH_TABLE[tech_idx];
    target_track = tech_def->category;
  } else if (action_id >= action_research_rare_start && action_id < action_build) {
    // Rare tech
    size_t offset = action_id - action_research_rare_start;
    size_t rare_idx = offset / 3;
    size_t track = offset % 3;
    if (rare_idx >= TECH_RARE_COUNT) return; // Invalid index
    tech_def = &TECH_TABLE[TECH_TABLE_SIZE + rare_idx];
    target_track = static_cast<TechCategory>(track);
  } else {
    return; // Invalid action
  }

  // Attempt to research the tech
  bool success = research_tech(s, player, *tech_def, target_track);
  if (!success) {
    // If research failed (shouldn't happen if LegalActions is correct), just decrement and continue
  }

  // Handle immediate tech benefits
  if (success) {
    if (tech_def->bit == TechBit::ADVANCED_ROBOTICS) {
      // Advanced Robotics: gain 1 influence disc
      s.players[player].extra_influence_discs += 1;
    } else if (tech_def->bit == TechBit::QUANTUM_GRID) {
      // Quantum Grid: gain 2 influence discs
      s.players[player].extra_influence_discs += 2;
    }
  }

  // Decrement activations remaining
  s.research_state.activations_remaining--;
  if (s.research_state.activations_remaining == 0) {
    s.research_state.phase = ::ResearchState::Phase::inactive;
    s.research_state.player_id = 255;
  }
}

void EclipseState::ApplyInfluenceSubAction(Action action_id) {
  ::State& s = eclipse_state_;
  const uint8_t player = s.influence_state.player_id;

  if (action_id == action_influence_stop) {
    s.influence_state.phase = ::InfluenceState::Phase::inactive;
    s.influence_state.player_id = 255;
    s.influence_state.activations_remaining = 0;
    s.influence_state.pending_returns.clear();
    return;
  }

  if (action_id >= action_influence_to_cell_start && action_id < action_reclaim_from_cell_start) {
    uint8_t cell_idx = static_cast<uint8_t>(action_id - action_influence_to_cell_start);
    SPIEL_CHECK_TRUE(execute_influence_to_sector(s, player, cell_idx));
  } else if (action_id >= action_reclaim_from_cell_start && action_id < action_choose_return_track_start) {
    uint8_t cell_idx = static_cast<uint8_t>(action_id - action_reclaim_from_cell_start);
    SPIEL_CHECK_TRUE(execute_reclaim_from_sector(s, player, cell_idx));
  } else if (action_id >= action_choose_return_track_start && action_id < action_choose_return_track_start + 3) {
    uint8_t track = static_cast<uint8_t>(action_id - action_choose_return_track_start);
    SPIEL_CHECK_TRUE(execute_choose_return_track(s, player, track));
  }
}

void EclipseState::ApplyBuildSubAction(Action action_id) {
  ::State& s = eclipse_state_;
  const uint8_t player = s.build_state.player_id;

  if (action_id == action_build_stop) {
    s.build_state.phase = ::BuildState::Phase::inactive;
    s.build_state.player_id = 255;
    s.build_state.activations_remaining = 0;
    return;
  }

  if (action_id >= action_build_choice_start && action_id < action_build_end) {
    int encoded = action_id - action_build_choice_start;
    BuildType type = static_cast<BuildType>(encoded / GALAXY_CELL_COUNT);
    uint8_t cell_idx = static_cast<uint8_t>(encoded % GALAXY_CELL_COUNT);
    if (!execute_build(s, player, type, cell_idx)) {
      // Build failed (e.g. stale legal action). Stay in build phase so the
      // player can try again without losing the influence disc.
    }
  }
}

void EclipseState::ApplyUpgradeSubAction(Action action_id) {
  ::State& s = eclipse_state_;
  const uint8_t player = s.upgrade_state.player_id;

  if (action_id == action_upgrade_stop) {
    s.upgrade_state.phase = ::UpgradeState::Phase::inactive;
    s.upgrade_state.player_id = 255;
    s.upgrade_state.activations_remaining = 0;
    return;
  }

  if (action_id >= action_upgrade_choice_start && action_id < action_upgrade_choice_start + UPGRADE_SHIP_COUNT * UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT) {
    int encoded = action_id - action_upgrade_choice_start;
    int ship = encoded / (UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT);
    int rem = encoded % (UPGRADE_SLOTS_PER_SHIP * UPGRADE_PART_COUNT);
    int slot = rem / UPGRADE_PART_COUNT;
    int part_idx = rem % UPGRADE_PART_COUNT;
    ShipType ship_type = static_cast<ShipType>(ship);
    ShipPartId part_id = static_cast<ShipPartId>(part_idx);
    SPIEL_CHECK_TRUE(execute_upgrade(s, player, ship_type, static_cast<uint8_t>(slot), part_id));
  }
}

void EclipseState::ApplyMoveSubAction(Action action_id) {
  ::State& s = eclipse_state_;
  const uint8_t player = s.move_state.player_id;

  if (action_id == action_move_stop) {
    s.move_state.phase = ::MoveState::Phase::inactive;
    s.move_state.player_id = 255;
    s.move_state.activations_remaining = 0;
    return;
  }

  if (action_id >= action_move_choice_start && action_id < action_move_warp_start) {
    int encoded = action_id - action_move_choice_start;
    uint8_t unit_idx = static_cast<uint8_t>(encoded / MOVE_CODES_PER_UNIT);
    uint8_t direction = static_cast<uint8_t>(encoded % MOVE_CODES_PER_UNIT);
    if (direction == MOVE_WARP_DIRECTION) {
      SPIEL_CHECK_TRUE(begin_warp_move(s, player, unit_idx));
    } else {
      SPIEL_CHECK_TRUE(execute_move_step(s, player, unit_idx, direction));
    }
  } else if (action_id >= action_move_warp_destination_start &&
             action_id < action_upkeep_colony_done) {
    uint8_t cell_idx = static_cast<uint8_t>(action_id - action_move_warp_destination_start);
    SPIEL_CHECK_TRUE(execute_warp_move(s, player, cell_idx));
  }
}

void EclipseState::ApplyDiplomacySubAction(Action action_id) {
  ::State& s = eclipse_state_;
  DiplomacyState& ds = s.diplomacy_state;

  if (ds.phase == DiplomacyState::Phase::inactive) return;

  if (ds.phase == DiplomacyState::Phase::choose_accept) {
    // Fix #1: partner accept/decline (rulebook p.14).
    if (action_id == action_diplomacy_accept) {
      execute_diplomacy_accept(s);
    } else if (action_id == action_diplomacy_decline) {
      execute_diplomacy_decline(s);
    }
    return;
  }

  if (ds.phase == DiplomacyState::Phase::choose_pop_track) {
    if (action_id >= action_diplomacy_pick_track_start &&
        action_id < action_diplomacy_pick_track_end) {
      int encoded = action_id - action_diplomacy_pick_track_start;
      int side = encoded / 3;
      int track = encoded % 3;
      const uint8_t expected = (side == 0) ? ds.player_id : ds.partner_id;
      execute_diplomacy_pick_track(s, expected, static_cast<PopTrack>(track));
    }
    return;
  }

  if (ds.phase == DiplomacyState::Phase::choose_rearrange) {
    if (action_id >= action_diplomacy_return_start &&
        action_id < action_diplomacy_return_end) {
      int encoded = action_id - action_diplomacy_return_start;
      int side = encoded / 5;
      int slot = encoded % 5;
      const uint8_t expected = (side == 0) ? ds.player_id : ds.partner_id;
      execute_return_rep_to_bag(s, expected, static_cast<uint8_t>(slot));
    } else if (action_id >= action_diplomacy_swap_start &&
               action_id < action_diplomacy_swap_end) {
      int encoded = action_id - action_diplomacy_swap_start;
      int side = encoded / 20;
      int rem = encoded % 20;
      int src = rem / 5;
      int dst = rem % 5;
      const uint8_t expected = (side == 0) ? ds.player_id : ds.partner_id;
      execute_swap_rep_slots(s, expected, static_cast<uint8_t>(src),
                             static_cast<uint8_t>(dst));
    }
    return;
  }

  if (ds.phase == DiplomacyState::Phase::choose_return_track) {
    if (action_id >= action_choose_return_track_start &&
        action_id < action_choose_return_track_start + 3) {
      int track = action_id - action_choose_return_track_start;
      const uint8_t expected = pending_return_track_player(s);
      if (expected < s.players.size()) {
        execute_choose_return_track(s, expected, static_cast<PopTrack>(track));
      }
    }
    return;
  }
}

void EclipseState::BeginUpkeep() {
  eclipse_state_.current_phase = RoundPhase::UPKEEP;
  eclipse_state_.upkeep_state = UpkeepState{};
  eclipse_state_.upkeep_state.player_id =
      FirstActivePlayerInTurnOrder(eclipse_state_, NumPlayers());
  if (eclipse_state_.upkeep_state.player_id != 255) {
    eclipse_state_.upkeep_state.step = UpkeepState::Step::colony_ships;
    AdvanceUpkeepState();
    return;
  }
  BeginCleanup();
}

void EclipseState::BeginCleanup() {
  eclipse_state_.current_phase = RoundPhase::CLEANUP;
  eclipse_state_.upkeep_state = UpkeepState{};
  for (auto& player : eclipse_state_.players) {
    player.disks_on_actions = 0;
    player.disks_on_reactions = 0;
  }

  DrawCleanupTechTiles(eclipse_state_, setup_config_.players);

  eclipse_state_.upkeep_state.player_id =
      FirstActivePlayerInTurnOrder(eclipse_state_, NumPlayers());
  if (eclipse_state_.upkeep_state.player_id != 255) {
    eclipse_state_.upkeep_state.step = UpkeepState::Step::cleanup_graveyards;
    QueueCleanupGraveyardReturns(eclipse_state_,
                                 eclipse_state_.upkeep_state.player_id);
    AdvanceCleanupState();
    return;
  }
  FinishCleanup();
}

void EclipseState::FinishCleanup() {
  for (auto& player : eclipse_state_.players) {
    if (player.eliminated) continue;
    player.has_passed = false;
    player.colony_ships_available = player.colony_ships_total;
  }

  uint8_t next_start = 255;
  for (uint8_t player_id : eclipse_state_.pass_order) {
    if (player_id < eclipse_state_.players.size() &&
        !eclipse_state_.players[player_id].eliminated) {
      next_start = player_id;
      break;
    }
  }
  if (next_start != 255) {
    std::vector<uint8_t> reordered;
    reordered.reserve(NumPlayers());
    int start_index = 0;
    for (int i = 0; i < NumPlayers(); ++i) {
      if (eclipse_state_.turn_order[i] == next_start) {
        start_index = i;
        break;
      }
    }
    for (int step = 0; step < NumPlayers(); ++step) {
      reordered.push_back(
          eclipse_state_.turn_order[(start_index + step) % NumPlayers()]);
    }
    for (int i = 0; i < MAX_PLAYERS; ++i) {
      eclipse_state_.turn_order[i] =
          i < reordered.size() ? reordered[i] : static_cast<uint8_t>(255);
    }
  }

  eclipse_state_.pass_order.clear();
  eclipse_state_.upkeep_state = UpkeepState{};
  if (eclipse_state_.current_round >= 8) {
    eclipse_state_.current_round = 9;
    eclipse_state_.current_player = 255;
    return;
  }

  eclipse_state_.current_phase = RoundPhase::ACTION;
  eclipse_state_.current_round += 1;
  eclipse_state_.current_player = eclipse_state_.turn_order[0];
}

void EclipseState::AdvanceUpkeepState() {
  while (eclipse_state_.upkeep_state.player_id < eclipse_state_.players.size()) {
    const uint8_t player_id = eclipse_state_.upkeep_state.player_id;
    ::Player& player = eclipse_state_.players[player_id];
    if (player.eliminated) {
      const uint8_t next_player =
          NextActivePlayerInTurnOrder(eclipse_state_, player_id, NumPlayers());
      if (next_player == 255) {
        BeginCleanup();
        return;
      }
      eclipse_state_.upkeep_state.player_id = next_player;
      eclipse_state_.upkeep_state.step = UpkeepState::Step::colony_ships;
      eclipse_state_.upkeep_state.pending_returns.clear();
      continue;
    }

    if (eclipse_state_.upkeep_state.step == UpkeepState::Step::choose_return_track) {
      return;
    }
    if (eclipse_state_.upkeep_state.step == UpkeepState::Step::colony_ships) {
      return;
    }
    if (eclipse_state_.upkeep_state.step == UpkeepState::Step::bankruptcy) {
      if (IsPlayerSolvent(player)) {
        return;
      }
      bool has_option = can_trade(player, TradeConversion::SCIENCE_TO_GOLD) ||
                        can_trade(player, TradeConversion::MATERIALS_TO_GOLD);
      if (!has_option) {
        has_option = HasReclaimableSector(eclipse_state_, player_id);
      }
      if (has_option) {
        return;
      }

      // Capture the score *before* flagging elimination and stripping the
      // player's components: compute_all_player_scores skips eliminated seats,
      // and RemovePlayerFromBoard would zero the board-derived VP anyway.
      // Rulebook (PLAYER ELIMINATION): "Eliminated players count their score."
      player.vp_at_elimination =
          compute_player_score(eclipse_state_, player_id).total_vp;
      player.eliminated = true;
      player.has_passed = true;
      RemovePlayerFromBoard(eclipse_state_, player_id);
    }

    const uint8_t next_player =
        NextActivePlayerInTurnOrder(eclipse_state_, player_id, NumPlayers());
    if (next_player != 255) {
      eclipse_state_.upkeep_state.player_id = next_player;
      eclipse_state_.upkeep_state.step = UpkeepState::Step::colony_ships;
      eclipse_state_.upkeep_state.pending_returns.clear();
      continue;
    }
    BeginCleanup();
    return;
  }
  BeginCleanup();
}

void EclipseState::AdvancePastCurrentUpkeepPlayer() {
  const uint8_t player_id = eclipse_state_.upkeep_state.player_id;
  const uint8_t next_player =
      NextActivePlayerInTurnOrder(eclipse_state_, player_id, NumPlayers());
  if (next_player != 255) {
    eclipse_state_.upkeep_state.player_id = next_player;
    eclipse_state_.upkeep_state.step = UpkeepState::Step::colony_ships;
    eclipse_state_.upkeep_state.pending_returns.clear();
    AdvanceUpkeepState();
  } else {
    BeginCleanup();
  }
}

void EclipseState::AdvanceCleanupState() {
  while (eclipse_state_.upkeep_state.player_id < eclipse_state_.players.size()) {
    const uint8_t player_id = eclipse_state_.upkeep_state.player_id;
    if (eclipse_state_.upkeep_state.step == UpkeepState::Step::choose_return_track) {
      return;
    }

    if (eclipse_state_.upkeep_state.pending_returns.empty()) {
      const uint8_t next_player =
          NextActivePlayerInTurnOrder(eclipse_state_, player_id, NumPlayers());
      if (next_player != 255) {
        eclipse_state_.upkeep_state.player_id = next_player;
        eclipse_state_.upkeep_state.step = UpkeepState::Step::cleanup_graveyards;
        QueueCleanupGraveyardReturns(eclipse_state_, next_player);
        continue;
      }
      FinishCleanup();
      return;
    }

    if (!ProcessCurrentPendingReturnsAuto(eclipse_state_)) {
      return;
    }
  }
  FinishCleanup();
}

void EclipseState::BeginCombat() {
  begin_combat_phase(eclipse_state_);
  DriveCombat();
}

void EclipseState::DriveCombat() {
  CombatState& cs = eclipse_state_.combat_state;
  for (int steps = 0;
       steps < 4000 && cs.phase != CombatState::Phase::inactive; ++steps) {
    const bool volley_active = cs.pending_target_group_player != kNoPlayer &&
                               cs.pending_die_index < cs.pending_die_count;
    // A queued die still needs rolling -> arm a chance node (one per die).
    if (volley_active &&
        cs.pending_die_values[cs.pending_die_index] == 0) {
      pending_random_event_ = PendingRandomEvent::combat_roll;
      return;
    }
    // A rolled die is awaiting the firing player's target choice.
    if (volley_active &&
        cs.pending_die_values[cs.pending_die_index] != 0) {
      return;
    }
    // A reputation tile still needs drawing -> arm a chance node (one per tile).
    if (cs.phase == CombatState::Phase::select_reputation_tile &&
        cs.tile_select_player != kNoPlayer &&
        cs.drawn_tiles_size < cs.rep_draw_target) {
      pending_random_event_ = PendingRandomEvent::reputation_draw;
      return;
    }
    // Player decisions.
    if (cs.phase == CombatState::Phase::choose_engagement_action &&
        cs.pending_player != kNoPlayer) {
      return;
    }
    if (cs.phase == CombatState::Phase::select_ship_order &&
        cs.pending_player != kNoPlayer) {
      return;
    }
    if (cs.phase == CombatState::Phase::attack_population &&
        cs.pop_attack_damage_remaining > 0) {
      return;
    }
    if (cs.phase == CombatState::Phase::select_reputation_tile &&
        cs.tile_select_player != kNoPlayer && cs.drawn_tiles_size > 0) {
      return;
    }
    if (cs.phase == CombatState::Phase::influence_sectors &&
        cs.influence_decision_player != kNoPlayer &&
        cs.influence_decision_sector != 0) {
      return;
    }
    if (cs.phase == CombatState::Phase::discovery_award &&
        cs.discovery_decision_player != kNoPlayer) {
      return;
    }
    advance_combat_state(eclipse_state_);
  }
  if (cs.phase == CombatState::Phase::inactive) {
    FinishCombat();
  }
}

void EclipseState::FinishCombat() {
  // Repair: zero all unit damage, reset combat state, set phase to UPKEEP.
  for (Unit& u : const_cast<FixedVector<Unit, 128>&>(eclipse_state_.unit_registry)) {
    u.damage = 0;
  }
  eclipse_state_.combat_state.Reset();
  BeginUpkeep();
}

std::vector<Action> EclipseState::CombatLegalActions() const {
  std::vector<Action> actions;
  const ::State& s = eclipse_state_;
  const CombatState& cs = s.combat_state;
  if (cs.phase == CombatState::Phase::inactive) return actions;

  // A rolled die awaiting the firing player's target choice. Legal targets are
  // the enemy ships in the active pair (the player may assign to any of them;
  // hits are resolved on application).
  if (cs.pending_target_group_player != kNoPlayer &&
      cs.pending_die_index < cs.pending_die_count &&
      cs.pending_die_values[cs.pending_die_index] != 0) {
    const uint8_t attacker = cs.pending_target_group_player;
    for (size_t i = 0; i < s.unit_registry.size() && i < 128; ++i) {
      if (IsLegalDieTarget(s, attacker, i)) {
        actions.push_back(action_combat_dice_target_start +
                          static_cast<Action>(i));
      }
    }
    std::sort(actions.begin(), actions.end());
    return actions;
  }

  if (cs.phase == CombatState::Phase::attack_population &&
      cs.pop_attack_damage_remaining > 0) {
    if (const Sector* sec = s.galaxy.FindSectorById(cs.pop_attack_sector_id)) {
      for (int slot = 0; slot < 16; ++slot) {
        if ((sec->occupied_slots_mask & static_cast<uint16_t>(1u << slot)) != 0) {
          actions.push_back(action_combat_pop_target_start + slot);
        }
      }
    }
    std::sort(actions.begin(), actions.end());
    return actions;
  }

  switch (cs.phase) {
    case CombatState::Phase::choose_engagement_action: {
      if (cs.pending_player != kNoPlayer) {
        actions.push_back(action_combat_attack);
        for (uint8_t i = 0; i < cs.retreat_destinations_size; ++i) {
          const uint16_t sid = cs.retreat_destinations[i];
          const HexCoord h = s.galaxy.FindSectorCoord(sid);
          if (h.q != -128) {
            actions.push_back(action_combat_retreat_to_cell_start + hex_to_index(h.q, h.r));
          }
        }
      } else {
        actions.push_back(action_combat_continue);
      }
      break;
    }
    case CombatState::Phase::select_ship_order: {
      if (cs.pending_player != kNoPlayer && cs.ship_order_size > 0) {
        for (uint8_t i = 0; i < cs.ship_order_size; ++i) {
          actions.push_back(action_combat_ship_order_start +
                            static_cast<Action>(cs.ship_order_queue[i]));
        }
      } else {
        actions.push_back(action_combat_continue);
      }
      break;
    }
    case CombatState::Phase::select_reputation_tile: {
      if (cs.tile_select_player != kNoPlayer && cs.drawn_tiles_size > 0) {
        for (uint8_t i = 0; i < cs.drawn_tiles_size; ++i) {
          actions.push_back(action_combat_rep_select_start + i);
        }
        actions.push_back(action_combat_rep_skip);
      } else {
        actions.push_back(action_combat_continue);
      }
      break;
    }
    case CombatState::Phase::influence_sectors: {
      if (cs.influence_decision_player != kNoPlayer &&
          cs.influence_decision_sector != 0) {
        actions.push_back(action_combat_influence_yes);
        actions.push_back(action_combat_influence_no);
      } else {
        actions.push_back(action_combat_continue);
      }
      break;
    }
    case CombatState::Phase::discovery_award: {
      if (cs.discovery_decision_player != kNoPlayer) {
        actions.push_back(action_combat_discovery_reward);
        actions.push_back(action_combat_discovery_vp);
      } else {
        actions.push_back(action_combat_continue);
      }
      break;
    }
    case CombatState::Phase::missile_phase:
    case CombatState::Phase::engagement_firing:
    case CombatState::Phase::attack_population:
    case CombatState::Phase::repair:
    case CombatState::Phase::determine_battles:
      actions.push_back(action_combat_continue);
      break;
    default:
      break;
  }
  std::sort(actions.begin(), actions.end());
  return actions;
}

void EclipseState::ApplyCombatSubAction(Action action_id) {
  ::State& s = eclipse_state_;
  CombatState& cs = s.combat_state;

  // Firing player assigns the rolled die awaiting a target. Volley completion
  // (advancing the initiative cursor and phase) is handled inside the combat
  // system's ApplyPlayerDieTarget / OnVolleyComplete.
  if (cs.pending_target_group_player != kNoPlayer &&
      cs.pending_die_index < cs.pending_die_count &&
      cs.pending_die_values[cs.pending_die_index] != 0) {
    if (action_id < action_combat_dice_target_start ||
        action_id >= action_combat_rep_select_start) {
      return;
    }
    const size_t target_idx =
        static_cast<size_t>(action_id - action_combat_dice_target_start);
    if (!IsLegalDieTarget(s, cs.pending_target_group_player, target_idx)) {
      return;
    }
    ApplyPlayerDieTarget(s, target_idx);
    return;
  }

  if (cs.phase == CombatState::Phase::attack_population &&
      cs.pop_attack_damage_remaining > 0) {
    if (action_id < action_combat_pop_target_start ||
        action_id >= action_combat_influence_yes) {
      return;
    }
    const int slot = action_id - action_combat_pop_target_start;
    if (slot < 0 || slot >= 16) return;
    Sector* sec = s.galaxy.FindSectorById(cs.pop_attack_sector_id);
    if (sec == nullptr) return;
    const uint16_t bit = static_cast<uint16_t>(1u << slot);
    if ((sec->occupied_slots_mask & bit) == 0) return;
    sec->occupied_slots_mask &= static_cast<uint16_t>(~bit);
    int graveyard = 0;
    const SectorDefinition* def = get_sector_definition(sec->sector_id);
    if (def && slot < static_cast<int>(def->slots.size())) {
      switch (def->slots[slot].type) {
        case PlanetType::MONEY:
        case PlanetType::ADV_MONEY:
        case PlanetType::ANY:
        case PlanetType::ADV_ANY:
          graveyard = 0;
          break;
        case PlanetType::SCIENCE:
        case PlanetType::ADV_SCIENCE:
          graveyard = 1;
          break;
        case PlanetType::MATERIALS:
        case PlanetType::ADV_MATERIALS:
          graveyard = 2;
          break;
      }
    }
    if (cs.pop_attack_owner < s.players.size()) {
      s.players[cs.pop_attack_owner].graveyard_counts[graveyard]++;
    }
    --cs.pop_attack_damage_remaining;
    if (cs.pop_attack_damage_remaining == 0) {
      cs.pop_attack_sector_id = 0;
      cs.pop_attack_player = kNoPlayer;
      cs.pop_attack_owner = kNoPlayer;
    }
  }

  // Ship order selection for tied-initiative groups.
  if (cs.phase == CombatState::Phase::select_ship_order) {
    if (cs.pending_player == kNoPlayer || cs.ship_order_size == 0) return;
    if (action_id < action_combat_ship_order_start ||
        action_id >= action_combat_ship_order_start + 4) return;
    ShipType chosen = static_cast<ShipType>(
        action_id - action_combat_ship_order_start);
    bool found = false;
    for (uint8_t i = 0; i < cs.ship_order_size; ++i) {
      if (cs.ship_order_queue[i] == chosen) { found = true; break; }
    }
    if (!found) return;
    // Permute queue so chosen is first; rest preserve relative order.
    for (uint8_t i = 0; i < cs.ship_order_size; ++i) {
      if (cs.ship_order_queue[i] == chosen) {
        for (uint8_t j = i; j > 0; --j) {
          cs.ship_order_queue[j] = cs.ship_order_queue[j - 1];
        }
        cs.ship_order_queue[0] = chosen;
        break;
      }
    }
    cs.ship_order_idx = 0;
    cs.phase = CombatState::Phase::engagement_firing;
    cs.active_ship_type = cs.ship_order_queue[0];
    cs.pending_player = kNoPlayer;  // cleared; CombatLegalActions uses this
    return;
  }

  switch (cs.phase) {
    case CombatState::Phase::choose_engagement_action: {
      if (cs.pending_player == kNoPlayer) return;
      if (action_id == action_combat_attack) {
        cs.pending_player = kNoPlayer;
        cs.phase = CombatState::Phase::engagement_firing;
        return;
      }
      if (action_id >= action_combat_retreat_to_cell_start &&
          action_id < action_combat_dice_target_start) {
        const int cell = action_id - action_combat_retreat_to_cell_start;
        if (cell < 0 || cell >= GALAXY_CELL_COUNT) return;
        const HexCoord h = index_to_hex(cell);
        const uint16_t destination_sector_id = s.galaxy.at(h.q, h.r).sector_id;
        bool legal_destination = false;
        for (uint8_t i = 0; i < cs.retreat_destinations_size; ++i) {
          if (cs.retreat_destinations[i] == destination_sector_id) {
            legal_destination = true;
            break;
          }
        }
        if (!legal_destination) return;

        const InitiativeGroup& g = cs.initiative_timeline[cs.initiative_idx];
        AddRetreatingGroup(s, g.player_id, g.type, destination_sector_id);
        ++cs.initiative_idx;
        cs.pending_player = kNoPlayer;
        return;
      }
      return;
    }
    case CombatState::Phase::select_reputation_tile: {
      if (cs.tile_select_player == kNoPlayer || cs.drawn_tiles_size == 0) return;
      const uint8_t player = cs.tile_select_player;
      bool keep_tile = false;
      uint8_t idx = 0;
      if (action_id >= action_combat_rep_select_start &&
          action_id < action_combat_rep_select_start + cs.drawn_tiles_size) {
        idx = static_cast<uint8_t>(action_id - action_combat_rep_select_start);
        keep_tile = true;
      } else if (action_id == action_combat_rep_skip) {
        keep_tile = false;
      } else {
        return;
      }
      if (keep_tile) {
        // Place the tile in a Reputation-capable slot (AMBASSADOR_OR_REP or
        // REP_ONLY). Cap at 5 by evicting the highest-index rep tile first.
        ::Player& p = s.players[player];
        int last_rep_slot = -1;
        for (size_t i = 0; i < p.reputation_track.size(); ++i) {
          if (!p.reputation_track[i].holds_ambassador &&
              p.reputation_track[i].kind != ReputationSlotKind::AMBASSADOR_ONLY) {
            last_rep_slot = static_cast<int>(i);
          }
        }
        if (last_rep_slot >= 0) {
          p.reputation_track[last_rep_slot].rep_value = cs.drawn_tiles[idx];
        } else {
          // All rep-capable slots occupied; force-evict the highest-index
          // rep tile to make room. Return the evicted tile to the bag.
          for (size_t i = p.reputation_track.size(); i > 0; --i) {
            const size_t slot_idx = i - 1;
            if (!p.reputation_track[slot_idx].holds_ambassador &&
                p.reputation_track[slot_idx].kind != ReputationSlotKind::AMBASSADOR_ONLY) {
              s.reputation_tiles.push_back(p.reputation_track[slot_idx].rep_value);
              p.reputation_track[slot_idx].rep_value = cs.drawn_tiles[idx];
              break;
            }
          }
        }
      }
      // Return the unselected tiles to the bag. Order is irrelevant because
      // future draws sample from the bag via reputation_draw chance nodes, so
      // no reshuffle (and no hidden RNG) is needed.
      for (uint8_t i = 0; i < cs.drawn_tiles_size; ++i) {
        if (keep_tile && i == idx) continue;
        s.reputation_tiles.push_back(cs.drawn_tiles[i]);
      }
      cs.drawn_tiles_size = 0;
      cs.rep_draw_target = 0;
      cs.tile_select_player = kNoPlayer;
      return;
    }
    case CombatState::Phase::influence_sectors: {
      if (cs.influence_decision_player == kNoPlayer) return;
      if (action_id == action_combat_influence_yes) {
        ::Player& p = s.players[cs.influence_decision_player];
        if (p.available_influence_discs() > 0) {
          p.disks_on_sectors++;
          if (Sector* sec = s.galaxy.FindSectorById(cs.influence_decision_sector)) {
            sec->owner_id = cs.influence_decision_player;
          }
        }
        ++cs.influence_scan_index;
        cs.influence_turn_order_index = 0;
      } else if (action_id == action_combat_influence_no) {
        ++cs.influence_turn_order_index;
      }
      cs.influence_decision_player = kNoPlayer;
      cs.influence_decision_sector = 0;
      return;
    }
    case CombatState::Phase::discovery_award: {
      if (cs.discovery_decision_player == kNoPlayer) return;
      Sector* discovery_sector = s.galaxy.FindSectorById(cs.discovery_decision_sector);
      if (discovery_sector == nullptr) return;
      if (action_id == action_combat_discovery_reward) {
        const uint8_t p = cs.discovery_decision_player;
        DiscoveryBit drawn = s.current_revealed_discovery;
        if (drawn == DiscoveryBit::NONE) drawn = RevealDiscovery(s, *discovery_sector);
        if (drawn == DiscoveryBit::NONE ||
            !apply_discovery_reward(s, p, *discovery_sector, drawn)) {
          s.players[p].discovery_vp_tiles_kept++;
        }
      } else if (action_id == action_combat_discovery_vp) {
        s.players[cs.discovery_decision_player].discovery_vp_tiles_kept++;
      }
      discovery_sector->discovery_tile_present = false;
      discovery_sector->discovery_tile = DiscoveryBit::NONE;
      s.current_revealed_discovery = DiscoveryBit::NONE;
      cs.discovery_decision_player = kNoPlayer;
      cs.discovery_decision_sector = 0;
      return;
    }
    default:
      break;
  }
}

void EclipseState::ApplyUpkeepAction(Action action_id) {
  UpkeepState& us = eclipse_state_.upkeep_state;
  const uint8_t player_id = us.player_id;
  ::Player& player = eclipse_state_.players[player_id];

  if (us.step == UpkeepState::Step::colony_ships) {
    if (action_id == action_upkeep_colony_done) {
      us.step = UpkeepState::Step::bankruptcy;
      AdvanceUpkeepState();
      return;
    }
    if (action_id >= action_colony_ship_start && action_id < action_influence_start) {
      int encoded = static_cast<int>(action_id - action_colony_ship_start);
      int cell = encoded / COLONY_SHIP_CODES_PER_CELL;
      int rem = encoded % COLONY_SHIP_CODES_PER_CELL;
      SPIEL_CHECK_TRUE(!SectorHasOpponentShips(eclipse_state_, player_id,
                                               static_cast<uint8_t>(cell)));
      SPIEL_CHECK_TRUE(use_colony_ship(
          eclipse_state_, player_id, static_cast<uint8_t>(cell),
          static_cast<uint8_t>(rem / COLONY_SHIP_TRACKS),
          static_cast<PopTrack>(rem % COLONY_SHIP_TRACKS)));
      return;
    }
  }

  if (us.step == UpkeepState::Step::bankruptcy) {
    if (action_id == action_upkeep_pay_done) {
      player.resources.gold = static_cast<uint8_t>(
          static_cast<int>(player.resources.gold) + PlayerIncome(player) -
          PlayerUpkeepCost(player));
      player.resources.materials = static_cast<uint8_t>(
          static_cast<int>(player.resources.materials) +
          PlayerMaterialsProduction(player));
      player.resources.science = static_cast<uint8_t>(
          static_cast<int>(player.resources.science) +
          PlayerScienceProduction(player));
      AdvancePastCurrentUpkeepPlayer();
      return;
    }
    if (action_id >= action_trade_start && action_id < action_colony_ship_start) {
      SPIEL_CHECK_TRUE(execute_trade(
          eclipse_state_, player_id,
          static_cast<TradeConversion>(action_id - action_trade_start)));
      AdvanceUpkeepState();
      return;
    }
    if (action_id >= action_reclaim_from_cell_start &&
        action_id < action_choose_return_track_start) {
      std::vector<PendingReturn> pending_returns;
      const uint8_t cell_idx =
          static_cast<uint8_t>(action_id - action_reclaim_from_cell_start);
      SPIEL_CHECK_TRUE(
          abandon_sector(eclipse_state_, player_id, cell_idx, &pending_returns));
      us.pending_returns = std::move(pending_returns);
      if (!ProcessCurrentPendingReturnsAuto(eclipse_state_)) {
        return;
      }
      AdvanceUpkeepState();
      return;
    }
  }

  if (us.step == UpkeepState::Step::choose_return_track) {
    if (action_id >= action_choose_return_track_start &&
        action_id < action_choose_return_track_start + 3 &&
        !us.pending_returns.empty()) {
      const PopTrack track = static_cast<PopTrack>(
          action_id - action_choose_return_track_start);
      const std::vector<PopTrack> legal = get_legal_return_tracks(
          player, us.pending_returns.front().type, us.pending_returns.front().is_orbital);
      SPIEL_CHECK_TRUE(std::find(legal.begin(), legal.end(), track) != legal.end());
      apply_return_to_track(player, track);
      us.pending_returns.erase(us.pending_returns.begin());
      us.step = eclipse_state_.current_phase == RoundPhase::UPKEEP
                    ? UpkeepState::Step::bankruptcy
                    : UpkeepState::Step::cleanup_graveyards;
      if (eclipse_state_.current_phase == RoundPhase::UPKEEP) {
        if (!ProcessCurrentPendingReturnsAuto(eclipse_state_)) {
          return;
        }
        AdvanceUpkeepState();
      } else {
        AdvanceCleanupState();
      }
      return;
    }
  }
}

void EclipseState::DoApplyAction(Action action_id) {
  if (pending_random_event_ != PendingRandomEvent::none) {
    const PendingRandomEvent event = pending_random_event_;
    ResolveChanceEvent(action_id);
    if (event == PendingRandomEvent::explore_draw) {
      pending_random_event_ = PendingRandomEvent::none;
      // An empty ring bag can end the last activation outright; otherwise a
      // player decision phase (place/draw-again/select) follows. Any further
      // draw (Draco's second tile) is re-armed from the decision branch below.
      if (eclipse_state_.explore_state.phase == ExplorePhase::inactive) {
        AdvanceTurn();
      }
    } else if (event == PendingRandomEvent::combat_roll ||
               event == PendingRandomEvent::reputation_draw) {
      // One die / tile resolved. Re-arm the next chance node, stop at a player
      // decision, or finish combat.
      pending_random_event_ = PendingRandomEvent::none;
      DriveCombat();
    }
    return;
  }

  const uint8_t current_player = eclipse_state_.current_player;
  if (eclipse_state_.current_phase == RoundPhase::ACTION &&
      IsActionPhaseBonusAction(action_id)) {
    ApplyActionPhaseBonusAction(eclipse_state_, current_player, action_id);
    return;
  }

  // Resolve a Diplomacy response before resuming an interrupted Action or
  // Reaction. CurrentPlayer() identifies the partner or proposer as needed.
  if (eclipse_state_.diplomacy_state.phase != DiplomacyState::Phase::inactive) {
    ApplyDiplomacySubAction(action_id);
    return;
  }

  if (eclipse_state_.minor_species_pending_track != 255) {
    const uint8_t track = action_id - action_minor_species_track_start;
    SPIEL_CHECK_TRUE(action_id >= action_minor_species_track_start &&
                     action_id < action_minor_species_track_end);
    SPIEL_CHECK_TRUE(execute_minor_species_pick_track(
        eclipse_state_, current_player, static_cast<PopTrack>(track)));
    return;
  }

  // Resolve a step of an in-flight Explore action without advancing the turn,
  // until all activations are done (phase returns to inactive).
  if (eclipse_state_.explore_state.phase != ExplorePhase::inactive) {
    ApplyExploreSubAction(action_id);
    if (eclipse_state_.explore_state.phase == ExplorePhase::draw_tile) {
      pending_random_event_ = PendingRandomEvent::explore_draw;
      return;
    }
    if (eclipse_state_.explore_state.phase != ExplorePhase::inactive) {
      return;
    }
    AdvanceTurn();
    return;
  }

  // Resolve a step of an in-flight Research action without advancing the turn,
  // until all activations are done (phase returns to inactive).
  if (eclipse_state_.research_state.phase != ::ResearchState::Phase::inactive) {
    ApplyResearchSubAction(action_id);
    if (eclipse_state_.research_state.phase != ::ResearchState::Phase::inactive) {
      return;
    }
    AdvanceTurn();
    return;
  }

  // Resolve a step of an in-flight Build action without advancing the turn,
  // until all activations are done (phase returns to inactive).
  if (eclipse_state_.build_state.phase != ::BuildState::Phase::inactive) {
    ApplyBuildSubAction(action_id);
    if (eclipse_state_.build_state.phase != ::BuildState::Phase::inactive) {
      return;
    }
    AdvanceTurn();
    return;
  }

  // Resolve a step of an in-flight Influence action without advancing the turn,
  // until all activations are done (phase returns to inactive).
  if (eclipse_state_.influence_state.phase != ::InfluenceState::Phase::inactive) {
    ApplyInfluenceSubAction(action_id);
    if (eclipse_state_.influence_state.phase != ::InfluenceState::Phase::inactive) {
      return;
    }
    AdvanceTurn();
    return;
  }

  // Resolve a step of an in-flight Upgrade action without advancing the turn,
  // until all activations are done (phase returns to inactive).
  if (eclipse_state_.upgrade_state.phase != ::UpgradeState::Phase::inactive) {
    ApplyUpgradeSubAction(action_id);
    if (eclipse_state_.upgrade_state.phase != ::UpgradeState::Phase::inactive) {
      return;
    }
    AdvanceTurn();
    return;
  }

  // Resolve a step of an in-flight Move action without advancing the turn,
  // until all activations are done (phase returns to inactive).
  if (eclipse_state_.move_state.phase != ::MoveState::Phase::inactive) {
    ApplyMoveSubAction(action_id);
    if (eclipse_state_.move_state.phase != ::MoveState::Phase::inactive) {
      return;
    }
    AdvanceTurn();
    return;
  }

  if (eclipse_state_.upkeep_state.step != UpkeepState::Step::inactive) {
    ApplyUpkeepAction(action_id);
    return;
  }

  if (eclipse_state_.current_phase == RoundPhase::COMBAT &&
      eclipse_state_.combat_state.phase != CombatState::Phase::inactive) {
    ApplyCombatSubAction(action_id);
    // Drive forward to the next chance roll / player decision / completion.
    // Dice and tile draws are resolved via chance nodes, so no RNG here.
    DriveCombat();
    return;
  }

  // ── REACTION actions (after passing) ─────────────────────────────────
  // Each reaction gives exactly 1 Activation, ignoring species/tech bonuses.
  // The disc cost is tracked via disks_on_reactions (Reaction Track) rather
  // than disks_on_actions (Action Track).
  if (action_id == action_reaction_upgrade) {
    if (current_player < eclipse_state_.players.size()) {
      bool started = begin_upgrade(eclipse_state_, current_player);
      if (started) {
        eclipse_state_.upgrade_state.activations_remaining = 1;
        ++eclipse_state_.players[current_player].disks_on_reactions;
      }
      if (eclipse_state_.upgrade_state.phase != ::UpgradeState::Phase::inactive) {
        return;
      }
    }
  } else if (action_id == action_reaction_build) {
    if (current_player < eclipse_state_.players.size()) {
      bool started = begin_build(eclipse_state_, current_player);
      if (started) {
        eclipse_state_.build_state.activations_remaining = 1;
        ++eclipse_state_.players[current_player].disks_on_reactions;
      }
      if (eclipse_state_.build_state.phase != ::BuildState::Phase::inactive) {
        return;
      }
    }
  } else if (action_id == action_reaction_move) {
    if (current_player < eclipse_state_.players.size()) {
      bool started = begin_move(eclipse_state_, current_player);
      if (started) {
        eclipse_state_.move_state.activations_remaining = 1;
        ++eclipse_state_.players[current_player].disks_on_reactions;
      }
      if (eclipse_state_.move_state.phase != ::MoveState::Phase::inactive) {
        return;
      }
    }
  } else if (action_id == action_pass) {
    if (current_player < eclipse_state_.players.size()) {
      ::Player& player = eclipse_state_.players[current_player];
      if (!player.has_passed) {
        player.has_passed = true;
        eclipse_state_.pass_order.push_back(current_player);
        if (eclipse_state_.pass_order.size() == 1) {
          player.resources.gold = static_cast<uint8_t>(
              static_cast<int>(player.resources.gold) + 2);
        }
      }
    }
  } else if (action_id == action_explore) {
    if (current_player < eclipse_state_.players.size()) {
      bool started = begin_explore(eclipse_state_, current_player);
      if (started) {
        ++eclipse_state_.players[current_player].disks_on_actions;
      }
      // begin_explore moves to choose_zone (wait for the player) unless there
      // were no legal zones, in which case it stays inactive and we advance.
      if (eclipse_state_.explore_state.phase != ExplorePhase::inactive) {
        return;
      }
    }
  } else if (action_id == action_research) {
    if (current_player < eclipse_state_.players.size()) {
      const auto& player = eclipse_state_.players[current_player];
      if (player.available_influence_discs() > 0 && player.resources.science >= 2) {
        // Start research action
        eclipse_state_.research_state.phase = ::ResearchState::Phase::choose_tech;
        eclipse_state_.research_state.player_id = current_player;
        // Get number of research activations from species
        const SpeciesData& species_data = SPECIES_TABLE[static_cast<size_t>(player.species_id)];
        eclipse_state_.research_state.activations_remaining = species_data.activations.research;
        ++eclipse_state_.players[current_player].disks_on_actions;
        return;
      }
    }
  } else if (action_id == action_influence_start) {
    if (current_player < eclipse_state_.players.size()) {
      const auto& player = eclipse_state_.players[current_player];
      if (player.available_influence_discs() > 0) {
        bool started = begin_influence(eclipse_state_, current_player);
        if (started) {
          ++eclipse_state_.players[current_player].disks_on_actions;
        }
        if (eclipse_state_.influence_state.phase != ::InfluenceState::Phase::inactive) {
          return;
        }
      }
    }
  } else if (action_id == action_build) {
    if (current_player < eclipse_state_.players.size()) {
      const auto& player = eclipse_state_.players[current_player];
      if (player.available_influence_discs() > 0) {
        bool started = begin_build(eclipse_state_, current_player);
        if (started) {
          ++eclipse_state_.players[current_player].disks_on_actions;
        }
        if (eclipse_state_.build_state.phase != ::BuildState::Phase::inactive) {
          return;
        }
      }
    }
  } else if (action_id == action_upgrade) {
    if (current_player < eclipse_state_.players.size()) {
      const auto& player = eclipse_state_.players[current_player];
      if (player.available_influence_discs() > 0) {
        bool started = begin_upgrade(eclipse_state_, current_player);
        if (started) {
          ++eclipse_state_.players[current_player].disks_on_actions;
        }
        if (eclipse_state_.upgrade_state.phase != ::UpgradeState::Phase::inactive) {
          return;
        }
      }
    }
  } else if (action_id == action_move) {
    if (current_player < eclipse_state_.players.size()) {
      const auto& player = eclipse_state_.players[current_player];
      if (player.available_influence_discs() > 0) {
        bool started = begin_move(eclipse_state_, current_player);
        if (started) {
          ++eclipse_state_.players[current_player].disks_on_actions;
        }
        if (eclipse_state_.move_state.phase != ::MoveState::Phase::inactive) {
          return;
        }
      }
    }
  } else if (action_id >= action_artifact_key_track_start &&
             action_id < action_artifact_key_track_end) {
    // Artifact Key resource-choice sub-action.
    // Find the player with pending chunks and grant 5 of the chosen type.
    uint8_t resource_type = action_id - action_artifact_key_track_start;
    for (::Player& p : eclipse_state_.players) {
      if (p.pending_artifact_key_chunks > 0) {
        switch (resource_type) {
          case 0: p.resources.gold += 5; break;
          case 1: p.resources.science += 5; break;
          case 2: p.resources.materials += 5; break;
        }
        --p.pending_artifact_key_chunks;
        return;
      }
    }
    return;
  }

  AdvanceTurn();
}

void EclipseState::AdvanceTurn() {
  // Do not advance the turn while any player has pending Artifact Key resource
  // choices — they must pick a resource type per chunk before play continues.
  for (const ::Player& p : eclipse_state_.players) {
    if (p.pending_artifact_key_chunks > 0) return;
  }

  const uint8_t current_player = eclipse_state_.current_player;

  // Detect an Act of Aggression: any sector the current player moved into or
  // attacked into this Action where a Diplomatic partner has a Ship or
  // Control. If detected, break those Diplomatic Relations and enqueue the
  // deferred return-track choice. The aggressor becomes the Traitor Tile
  // holder (rulebook p.14-15).
  if (eclipse_state_.current_phase == RoundPhase::ACTION) {
    if (eclipse_state_.diplomacy_state.phase == DiplomacyState::Phase::inactive) {
      break_all_diplomacy_for(eclipse_state_, current_player);
    }
  }

  bool all_passed = true;
  for (const auto& player : eclipse_state_.players) {
    if (!player.eliminated && !player.has_passed) {
      all_passed = false;
      break;
    }
  }

  if (all_passed) {
    BeginCombat();
    return;
  }

  const int num_players = NumPlayers();
  int current_index = -1;
  for (int i = 0; i < num_players; ++i) {
    if (eclipse_state_.turn_order[i] == current_player) {
      current_index = i;
      break;
    }
  }

  for (int step = 1; step <= num_players; ++step) {
    const int next_index = (current_index + step) % num_players;
    const uint8_t next_player_id = eclipse_state_.turn_order[next_index];
    if (next_player_id < eclipse_state_.players.size() &&
        !eclipse_state_.players[next_player_id].eliminated) {
      eclipse_state_.current_player = next_player_id;
      return;
    }
  }
}

}  // namespace eclipse
}  // namespace open_spiel
