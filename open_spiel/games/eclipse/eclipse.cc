//
// Created by Mihai on 01/06/2026.
//

#include "open_spiel/games/eclipse/eclipse.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "open_spiel/games/eclipse/systems/actions/explore.h"
#include "open_spiel/games/eclipse/systems/setup.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/observer.h"

namespace open_spiel {
namespace eclipse {

namespace {

constexpr Action chance_resolve = 0;

// Action layout:
//   0        = PASS  (also the no-op chance-resolve token)
//   1-8      = RESEARCH (index 0-7)
//   9-16     = BUILD (index 0-7)
//   17       = start an EXPLORE action
//   18 / 19  = explore: place / discard the drawn tile
//   20-25    = explore: place with rotation 0-5
//   26 / 27  = explore: take control / decline
//   28 / 29  = explore: discovery reward / 2 VP
//   30 / 31  = explore (Draco): keep drawn tile 0 / 1
//   32 / 33  = explore (Draco): flip a second tile / proceed with one
//   34       = explore: stop (decline remaining activations)
//   35..259  = explore: choose zone == galaxy cell index (one id per hex)
constexpr Action action_pass = 0;
constexpr Action action_research_start = 1;
constexpr Action action_build_start = 9;
constexpr Action action_explore = 17;
constexpr Action explore_place = 18;
constexpr Action explore_discard = 19;
constexpr Action explore_rotation_start = 20;
constexpr Action explore_claim_yes = 26;
constexpr Action explore_claim_no = 27;
constexpr Action explore_discovery_reward = 28;
constexpr Action explore_discovery_vp = 29;
constexpr Action explore_select_tile_start = 30;
constexpr Action explore_draw_again = 32;
constexpr Action explore_skip_second = 33;
constexpr Action explore_stop = 34;
constexpr Action explore_zone_start = 35;  // + galaxy cell index (0..224)
constexpr int num_distinct_actions = explore_zone_start + GALAXY_CELL_COUNT;

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
    case EclipseState::PendingRandomEvent::discovery_draw:
      return "discovery_draw";
    case EclipseState::PendingRandomEvent::combat_roll:
      return "combat_roll";
  }
  return "unknown";
}

EclipseState::PendingRandomEvent PendingRandomEventFromInt(int value) {
  if (value < 0 ||
      value > static_cast<int>(EclipseState::PendingRandomEvent::combat_roll)) {
    throw std::invalid_argument("invalid pending random event");
  }
  return static_cast<EclipseState::PendingRandomEvent>(value);
}

}  // namespace

EclipseGame::EclipseGame(const GameParameters& params)
    : Game(game_type, params),
      rng_(std::mt19937_64(static_cast<uint64_t>(ParameterValue<int>(
          "rng_seed")))) {}

int EclipseGame::NumDistinctActions() const { return num_distinct_actions; }

int EclipseGame::NumPlayers() const { return ParameterValue<int>("players"); }

int EclipseGame::MaxGameLength() const { return 1000; }

std::vector<int> EclipseGame::ObservationTensorShape() const { return {128}; }

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

  // Mid-Explore: only the choices valid for the current sub-phase are legal.
  const ::State& s = eclipse_state_;
  if (s.explore_state.phase != ExplorePhase::inactive) {
    return ExploreLegalActions();
  }

  std::vector<Action> actions;
  actions.push_back(action_pass);

  uint8_t current_player = eclipse_state_.current_player;
  if (current_player < eclipse_state_.players.size() &&
      !eclipse_state_.players[current_player].has_passed) {
    const auto& player = eclipse_state_.players[current_player];
    const bool has_action_disk = available_influence_discs(player) > 0;
    if (has_action_disk && player.resources.science >= 2) {
      for (int i = 0; i < 8; ++i) {
        actions.push_back(action_research_start + i);
      }
    }
    if (has_action_disk && player.resources.materials >= 3) {
      for (int i = 0; i < 4; ++i) {
        actions.push_back(action_build_start + i);
      }
    }
    if (has_action_disk && has_explore_zone(s, current_player)) {
      actions.push_back(action_explore);
    }
  }

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
          available_influence_discs(s.players[es.player_id]) > 0) {
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
    default:
      break;
  }
  // OpenSpiel requires LegalActions sorted ascending; zone ids are emitted in
  // grid-discovery order and explore_stop trails the zone block numerically.
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
  if (pending_random_event_ != PendingRandomEvent::none) {
    return "RESOLVE_" + PendingRandomEventToString(pending_random_event_);
  }
  if (action_id == action_pass) {
    return "PASS";
  }
  if (action_id == action_explore) {
    return "EXPLORE";
  }
  if (action_id >= action_research_start && action_id < action_build_start) {
    return "RESEARCH_" + std::to_string(action_id - action_research_start);
  }
  if (action_id >= action_build_start && action_id < action_explore) {
    return "BUILD_" + std::to_string(action_id - action_build_start);
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
         eclipse_state_.current_round > 9;
}

std::vector<double> EclipseState::Returns() const {
  std::vector<double> returns(NumPlayers(), 0.0);
  if (!IsTerminal()) {
    return returns;
  }
  for (size_t i = 0; i < returns.size() && i < eclipse_state_.players.size();
       ++i) {
    returns[i] = static_cast<double>(eclipse_state_.players[i].score);
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

void EclipseState::ObservationTensor(Player player, absl::Span<float> values) const {
  std::fill(values.begin(), values.end(), 0.0f);

  if (pending_random_event_ != PendingRandomEvent::none) {
    values[52] = static_cast<float>(pending_random_event_);
    return;
  }
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());

  const auto& me = eclipse_state_.players[player];
  values[0] = static_cast<float>(me.score);
  values[1] = static_cast<float>(me.resources.gold);
  values[2] = static_cast<float>(me.resources.science);
  values[3] = static_cast<float>(me.resources.materials);
  values[4] = me.has_passed ? 1.0f : 0.0f;

  int idx = 5;
  for (int other = 0; other < NumPlayers(); ++other) {
    if (other == player) {
      continue;
    }
    if (other < static_cast<int>(eclipse_state_.players.size())) {
      const auto& other_player = eclipse_state_.players[other];
      values[idx++] = static_cast<float>(other_player.score);
      values[idx++] = other_player.has_passed ? 1.0f : 0.0f;
    } else {
      idx += 2;
    }
  }

  values[50] = static_cast<float>(eclipse_state_.current_round);
  values[51] = static_cast<float>(eclipse_state_.current_phase);

  // Explore sub-state (public): lets an agent at a place/rotate/claim decision
  // node actually see the drawn tile, zone and phase it is deciding on.
  const ExploreState& es = eclipse_state_.explore_state;
  if (es.phase != ExplorePhase::inactive) {
    values[53] = 1.0f;
    values[54] = static_cast<float>(es.phase);
    values[55] = static_cast<float>(es.activations_remaining);
    values[56] = static_cast<float>(es.ring);
    values[57] = static_cast<float>(es.zone_q);
    values[58] = static_cast<float>(es.zone_r);
    values[59] = static_cast<float>(es.drawn_count);
    values[60] = static_cast<float>(es.drawn_sector_ids[0]);
    values[61] = static_cast<float>(es.drawn_sector_ids[1]);
    values[62] = static_cast<float>(es.selected_sector_id);
    values[63] = static_cast<float>(es.chosen_rotation);
  }
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
      eclipse_state_ = InitializeDeterministicSetupState(setup_config_);
      ResolveInitialSetupRandomness(eclipse_game_->rng(), setup_config_,
                                    eclipse_state_);

      std::vector<PlayerConfig> player_choices;
      player_choices.reserve(setup_config_.players);
      for (const StagedPlayerConfig& staged_player :
           setup_config_.staged_players) {
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
    case PendingRandomEvent::none:
      SpielFatalError("no pending random event to resolve");
    case PendingRandomEvent::discovery_draw:
    case PendingRandomEvent::combat_roll:
      SpielFatalError("pending random event is declared but unimplemented");
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
    default:
      break;
  }
}

void EclipseState::DoApplyAction(Action action_id) {
  if (pending_random_event_ != PendingRandomEvent::none) {
    bool was_explore_draw =
        pending_random_event_ == PendingRandomEvent::explore_draw;
    ResolveChanceEvent(action_id);
    if (was_explore_draw) {
      pending_random_event_ = PendingRandomEvent::none;
      // An empty ring bag can end the last activation outright; otherwise a
      // player decision phase (place/draw-again/select) follows. Any further
      // draw (Draco's second tile) is re-armed from the decision branch below.
      if (eclipse_state_.explore_state.phase == ExplorePhase::inactive) {
        AdvanceTurn();
      }
    }
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

  uint8_t current_player = eclipse_state_.current_player;

  if (action_id == action_pass) {
    if (current_player < eclipse_state_.players.size()) {
      eclipse_state_.players[current_player].has_passed = true;
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
  } else if (action_id >= action_research_start &&
             action_id < action_build_start) {
    // NOTE: PLACEHOLDER - research is not wired to research_tech() yet.
    if (current_player < eclipse_state_.players.size()) {
      auto& player = eclipse_state_.players[current_player];
      if (available_influence_discs(player) > 0 && player.resources.science >= 2) {
        ++player.disks_on_actions;
        player.resources.science -= 2;
        player.score += 2;
      }
    }
  } else if (action_id >= action_build_start && action_id < action_explore) {
    // NOTE: PLACEHOLDER - build is not implemented yet.
    if (current_player < eclipse_state_.players.size()) {
      auto& player = eclipse_state_.players[current_player];
      if (available_influence_discs(player) > 0 && player.resources.materials >= 3) {
        ++player.disks_on_actions;
        player.resources.materials -= 3;
        player.score += 3;
      }
    }
  }

  AdvanceTurn();
}

void EclipseState::AdvanceTurn() {
  const uint8_t current_player = eclipse_state_.current_player;

  bool all_passed = true;
  for (const auto& player : eclipse_state_.players) {
    if (!player.has_passed) {
      all_passed = false;
      break;
    }
  }

  if (all_passed) {
    eclipse_state_.current_round += 1;
    if (eclipse_state_.current_round <= 9) {
      for (auto& player : eclipse_state_.players) {
        player.has_passed = false;
        player.disks_on_actions = 0;
      }
      eclipse_state_.current_player = eclipse_state_.turn_order[0];
    }
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
        !eclipse_state_.players[next_player_id].has_passed) {
      eclipse_state_.current_player = next_player_id;
      return;
    }
  }
}

}  // namespace eclipse
}  // namespace open_spiel
