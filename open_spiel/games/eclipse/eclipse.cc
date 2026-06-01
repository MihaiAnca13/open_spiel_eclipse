//
// Created by Mihai on 01/06/2026.
//

#include "open_spiel/games/eclipse/eclipse.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "open_spiel/games/eclipse/systems/setup.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/observer.h"

namespace open_spiel {
namespace eclipse {

namespace {

constexpr Action kChanceResolve = 0;

// Action ID: 0 = PASS
// Action ID: 1-8 = RESEARCH (index 0 to 7)
// Action ID: 9-16 = BUILD (index 0 to 7)
// Action ID: 17 = DUMMY EXPLORE
constexpr Action kActionPass = 0;
constexpr Action kActionResearchStart = 1;
constexpr Action kActionBuildStart = 9;
constexpr Action kActionExplore = 17;

const GameType kGameType{
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

REGISTER_SPIEL_GAME(kGameType, CreateGame);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::string PendingRandomEventToString(
    EclipseState::PendingRandomEvent pending_event) {
  switch (pending_event) {
    case EclipseState::PendingRandomEvent::kNone:
      return "none";
    case EclipseState::PendingRandomEvent::kInitialSetup:
      return "initial_setup";
    case EclipseState::PendingRandomEvent::kExploreDraw:
      return "explore_draw";
    case EclipseState::PendingRandomEvent::kDiscoveryDraw:
      return "discovery_draw";
    case EclipseState::PendingRandomEvent::kCombatRoll:
      return "combat_roll";
  }
  return "unknown";
}

EclipseState::PendingRandomEvent PendingRandomEventFromInt(int value) {
  if (value < 0 ||
      value > static_cast<int>(EclipseState::PendingRandomEvent::kCombatRoll)) {
    throw std::invalid_argument("invalid pending random event");
  }
  return static_cast<EclipseState::PendingRandomEvent>(value);
}

}  // namespace

EclipseGame::EclipseGame(const GameParameters& params)
    : Game(kGameType, params),
      rng_(std::mt19937_64(static_cast<uint64_t>(ParameterValue<int>(
          "rng_seed")))) {}

int EclipseGame::NumDistinctActions() const { return 32; }

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
  if (pending_random_event_ != PendingRandomEvent::kNone) {
    return kChancePlayerId;
  }
  return eclipse_state_.current_player;
}

std::vector<Action> EclipseState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }
  if (pending_random_event_ != PendingRandomEvent::kNone) {
    return {kChanceResolve};
  }

  std::vector<Action> actions;
  actions.push_back(kActionPass);

  uint8_t current_player = eclipse_state_.current_player;
  if (current_player < eclipse_state_.players.size() &&
      !eclipse_state_.players[current_player].has_passed) {
    const auto& player = eclipse_state_.players[current_player];
    if (player.resources.science >= 2) {
      for (int i = 0; i < 8; ++i) {
        actions.push_back(kActionResearchStart + i);
      }
    }
    if (player.resources.materials >= 3) {
      for (int i = 0; i < 4; ++i) {
        actions.push_back(kActionBuildStart + i);
      }
    }
    actions.push_back(kActionExplore);
  }

  return actions;
}

std::string EclipseState::ActionToString(Player player, Action action_id) const {
  if (pending_random_event_ != PendingRandomEvent::kNone) {
    return "RESOLVE_" + PendingRandomEventToString(pending_random_event_);
  }
  if (action_id == kActionPass) {
    return "PASS";
  }
  if (action_id == kActionExplore) {
    return "EXPLORE";
  }
  if (action_id >= kActionResearchStart && action_id < kActionBuildStart) {
    return "RESEARCH_" + std::to_string(action_id - kActionResearchStart);
  }
  if (action_id >= kActionBuildStart && action_id < kActionExplore) {
    return "BUILD_" + std::to_string(action_id - kActionBuildStart);
  }
  return "UNKNOWN_ACTION(" + std::to_string(action_id) + ")";
}

std::string EclipseState::ToString() const {
  std::stringstream ss;
  if (pending_random_event_ != PendingRandomEvent::kNone) {
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
  return pending_random_event_ == PendingRandomEvent::kNone &&
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
  if (pending_random_event_ == PendingRandomEvent::kNone) {
    return {};
  }
  return {{kChanceResolve, 1.0}};
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
  if (pending_random_event_ != PendingRandomEvent::kNone) {
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
  return ss.str();
}

void EclipseState::ObservationTensor(Player player, absl::Span<float> values) const {
  std::fill(values.begin(), values.end(), 0.0f);

  if (pending_random_event_ != PendingRandomEvent::kNone) {
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
}

void EclipseState::RestoreFromSnapshot(
    const SetupConfig& config, const ::State& state,
    PendingRandomEvent pending_random_event) {
  setup_config_ = config;
  eclipse_state_ = state;
  pending_random_event_ = pending_random_event;
}

void EclipseState::ResolveChanceEvent(Action action_id) {
  SPIEL_CHECK_EQ(action_id, kChanceResolve);

  switch (pending_random_event_) {
    case PendingRandomEvent::kInitialSetup: {
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
      pending_random_event_ = PendingRandomEvent::kNone;
      return;
    }
    case PendingRandomEvent::kNone:
      SpielFatalError("no pending random event to resolve");
    case PendingRandomEvent::kExploreDraw:
    case PendingRandomEvent::kDiscoveryDraw:
    case PendingRandomEvent::kCombatRoll:
      SpielFatalError("pending random event is declared but unimplemented");
  }
}

void EclipseState::DoApplyAction(Action action_id) {
  if (pending_random_event_ != PendingRandomEvent::kNone) {
    ResolveChanceEvent(action_id);
    return;
  }

  uint8_t current_player = eclipse_state_.current_player;

  // NOTE: ASSUMPTION / PLACEHOLDER
  // These actions are actually not implemented correctly yet. just placeholder code
  if (action_id == kActionPass) {
    if (current_player < eclipse_state_.players.size()) {
      eclipse_state_.players[current_player].has_passed = true;
    }
  } else if (action_id == kActionExplore) {
    if (current_player < eclipse_state_.players.size()) {
      eclipse_state_.players[current_player].score += 1;
    }
  } else if (action_id >= kActionResearchStart &&
             action_id < kActionBuildStart) {
    if (current_player < eclipse_state_.players.size()) {
      auto& player = eclipse_state_.players[current_player];
      if (player.resources.science >= 2) {
        player.resources.science -= 2;
        player.score += 2;
      }
    }
  } else if (action_id >= kActionBuildStart && action_id < kActionExplore) {
    if (current_player < eclipse_state_.players.size()) {
      auto& player = eclipse_state_.players[current_player];
      if (player.resources.materials >= 3) {
        player.resources.materials -= 3;
        player.score += 3;
      }
    }
  }

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
