//
// Created by Mihai on 01/06/2026.
//

#include "open_spiel/games/eclipse/eclipse.h"

#include <algorithm>
#include <sstream>

#include "open_spiel/games/eclipse/systems/setup.h"
#include "open_spiel/observer.h"

namespace open_spiel {
namespace eclipse {

namespace {

// Definition of the game type/properties
const GameType kGameType{
    /*short_name=*/"eclipse",
    /*long_name=*/"Eclipse: New Dawn for the Galaxy",
    /*dynamics=*/GameType::Dynamics::kSequential,
    /*chance_mode=*/GameType::ChanceMode::kDeterministic,
    /*information=*/GameType::Information::kImperfectInformation,
    /*utility=*/GameType::Utility::kGeneralSum,
    /*reward_model=*/GameType::RewardModel::kTerminal,
    /*max_num_players=*/6,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false, // We use observation_tensor instead
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{
        {"players", GameParameter(4)},
        {"seed", GameParameter(0)}
    }
};

std::shared_ptr<const Game> CreateGame(const GameParameters& params) {
  return std::make_shared<EclipseGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, CreateGame);
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Encode/decode simple actions
// Action ID: 0 = PASS
// Action ID: 1-8 = RESEARCH (index 0 to 7)
// Action ID: 9-16 = BUILD (index 0 to 7)
// Action ID: 17 = DUMMY EXPLORE
constexpr Action kActionPass = 0;
constexpr Action kActionResearchStart = 1;
constexpr Action kActionBuildStart = 9;
constexpr Action kActionExplore = 17;

}  // namespace

EclipseGame::EclipseGame(const GameParameters& params)
    : Game(kGameType, params) {}

int EclipseGame::NumDistinctActions() const {
  return 32; // Static placeholder size for action space
}

int EclipseGame::NumPlayers() const {
  return ParameterValue<int>("players");
}

int EclipseGame::MaxGameLength() const {
  return 1000; // Placeholder implementation can take many low-value actions.
}

std::vector<int> EclipseGame::ObservationTensorShape() const {
  // Simple observation shape for proof of concept
  // Let's say 128 floats of state variables (resources, scores, technologies, etc.)
  return {128};
}

std::unique_ptr<State> EclipseGame::NewInitialState() const {
  return std::make_unique<EclipseState>(shared_from_this());
}

EclipseState::EclipseState(std::shared_ptr<const Game> game)
    : State(game) {
  const EclipseGame* eclipse_game = static_cast<const EclipseGame*>(game.get());
  int num_players = eclipse_game->NumPlayers();
  int seed = eclipse_game->GetSeedParam();

  // Stage 1: Setup pre-choice state
  eclipse_state_ = initialize_pre_choice_state(seed, num_players, NPCDifficulty::EASY);

  // Stage 2: Commit default player choices
  std::vector<PlayerConfig> choices;
  for (int i = 0; i < num_players; ++i) {
    choices.push_back(PlayerConfig{
        .species = Species::TERRAN_FACTIONS,
        .is_ai = true
    });
  }
  finalize_game_setup(eclipse_state_, choices);
  initialized_ = true;
}

Player EclipseState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  return eclipse_state_.current_player;
}

std::vector<Action> EclipseState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  }
  
  std::vector<Action> actions;
  
  // Every active player can always PASS
  actions.push_back(kActionPass);

  // Placeholder actions if player has not passed
  uint8_t current_p = eclipse_state_.current_player;
  if (current_p < eclipse_state_.players.size() && !eclipse_state_.players[current_p].has_passed) {
    // Add research actions
    for (int i = 0; i < 8; ++i) {
      actions.push_back(kActionResearchStart + i);
    }
    // Add build actions
    for (int i = 0; i < 4; ++i) {
      actions.push_back(kActionBuildStart + i);
    }
    actions.push_back(kActionExplore);
  }

  return actions;
}

std::string EclipseState::ActionToString(Player player, Action action_id) const {
  if (action_id == kActionPass) {
    return "PASS";
  } else if (action_id == kActionExplore) {
    return "EXPLORE";
  } else if (action_id >= kActionResearchStart && action_id < kActionBuildStart) {
    return "RESEARCH_" + std::to_string(action_id - kActionResearchStart);
  } else if (action_id >= kActionBuildStart && action_id < kActionExplore) {
    return "BUILD_" + std::to_string(action_id - kActionBuildStart);
  }
  return "UNKNOWN_ACTION(" + std::to_string(action_id) + ")";
}

std::string EclipseState::ToString() const {
  std::stringstream ss;
  ss << "Eclipse Game State:\n";
  ss << "Round: " << (int)eclipse_state_.current_round << "\n";
  ss << "Phase: " << (int)eclipse_state_.current_phase << "\n";
  ss << "Current Player: " << (int)eclipse_state_.current_player << "\n";
  ss << "Players:\n";
  for (const auto& p : eclipse_state_.players) {
    ss << "  Player " << (int)p.id << " [Score: " << (int)p.score 
       << ", Money: " << static_cast<int>(p.resources.gold)
       << ", Science: " << static_cast<int>(p.resources.science)
       << ", Materials: " << static_cast<int>(p.resources.materials)
       << ", Passed: " << (p.has_passed ? "Yes" : "No") << "]\n";
  }
  return ss.str();
}

bool EclipseState::IsTerminal() const {
  // Eclipse ends after round 9 (or in this prototype we can also check if all players have passed 9 times/rounds)
  return eclipse_state_.current_round > 9;
}

std::vector<double> EclipseState::Returns() const {
  std::vector<double> returns(NumPlayers(), 0.0);
  if (!IsTerminal()) {
    return returns;
  }
  for (size_t i = 0; i < returns.size(); ++i) {
    if (i < eclipse_state_.players.size()) {
      returns[i] = static_cast<double>(eclipse_state_.players[i].score);
    }
  }
  return returns;
}

ActionsAndProbs EclipseState::ChanceOutcomes() const {
  // Setup doesn't have runtime chance nodes in this basic wrapper
  return {};
}

std::string EclipseState::InformationStateString(Player player) const {
  // Information State represents the perfect-recall history/observations of the player
  // For proof of concept, return egocentric ObservationString
  return ObservationString(player);
}

std::string EclipseState::ObservationString(Player player) const {
  // Fog of war observation string
  std::stringstream ss;
  ss << "Observation for Player " << player << ":\n";
  ss << "My Score: ";
  if (player < (int)eclipse_state_.players.size()) {
    const auto& me = eclipse_state_.players[player];
    ss << static_cast<int>(me.score)
       << ", Money: " << static_cast<int>(me.resources.gold)
       << ", Science: " << static_cast<int>(me.resources.science)
       << ", Materials: " << static_cast<int>(me.resources.materials) << "\n";
  }
  ss << "Visible sectors owned by me or empty near me.\n";
  return ss.str();
}

void EclipseState::ObservationTensor(Player player, absl::Span<float> values) const {
  // Make sure the span matches our ObservationTensorShape()
  std::fill(values.begin(), values.end(), 0.0f);

  if (player >= (int)eclipse_state_.players.size()) {
    return;
  }

  // Populate egocentric player properties
  const auto& me = eclipse_state_.players[player];
  values[0] = static_cast<float>(me.score);
  values[1] = static_cast<float>(me.resources.gold);
  values[2] = static_cast<float>(me.resources.science);
  values[3] = static_cast<float>(me.resources.materials);
  values[4] = me.has_passed ? 1.0f : 0.0f;

  // Other players properties (relative index)
  int idx = 5;
  for (int other = 0; other < NumPlayers(); ++other) {
    if (other == player) continue;
    if (other < (int)eclipse_state_.players.size()) {
      const auto& p = eclipse_state_.players[other];
      values[idx++] = static_cast<float>(p.score);
      values[idx++] = p.has_passed ? 1.0f : 0.0f;
    } else {
      idx += 2;
    }
  }

  // Round information
  values[50] = static_cast<float>(eclipse_state_.current_round);
  values[51] = static_cast<float>(eclipse_state_.current_phase);
}

void EclipseState::DoApplyAction(Action action_id) {
  uint8_t current_p = eclipse_state_.current_player;

  if (action_id == kActionPass) {
    if (current_p < eclipse_state_.players.size()) {
      eclipse_state_.players[current_p].has_passed = true;
    }
  } else if (action_id == kActionExplore) {
    // NOTE: ASSUMPTION / PLACEHOLDER
    // The core explore rules (drawing sectors, placement) are not yet fully implemented in C++.
    // Here we temporarily simulate a successful explore action by incrementing the score.
    if (current_p < eclipse_state_.players.size()) {
      eclipse_state_.players[current_p].score += 1;
    }
  } else if (action_id >= kActionResearchStart && action_id < kActionBuildStart) {
    // NOTE: ASSUMPTION / PLACEHOLDER
    // The core research rules (buying tech from market trays) are not yet fully implemented in C++.
    // Here we temporarily simulate research by deducting science points and adding score points.
    if (current_p < eclipse_state_.players.size()) {
      eclipse_state_.players[current_p].resources.science -= 2;
      eclipse_state_.players[current_p].score += 2;
    }
  } else if (action_id >= kActionBuildStart && action_id < kActionExplore) {
    // NOTE: ASSUMPTION / PLACEHOLDER
    // The core build rules (building units at controlled sectors) are not yet fully implemented in C++.
    // Here we temporarily simulate build actions by deducting materials and adding score points.
    if (current_p < eclipse_state_.players.size()) {
      eclipse_state_.players[current_p].resources.materials -= 3;
      eclipse_state_.players[current_p].score += 3;
    }
  }

  // Move to next player in turn order
  bool all_passed = true;
  for (const auto& p : eclipse_state_.players) {
    if (!p.has_passed) {
      all_passed = false;
      break;
    }
  }

  if (all_passed) {
    // End round, reset pass state
    eclipse_state_.current_round += 1;
    if (eclipse_state_.current_round <= 9) {
      for (auto& p : eclipse_state_.players) {
        p.has_passed = false;
      }
      eclipse_state_.current_player = eclipse_state_.turn_order[0];
    }
  } else {
    // Find next player who hasn't passed
    int num_players = NumPlayers();
    int current_idx = -1;
    for (int i = 0; i < num_players; ++i) {
      if (eclipse_state_.turn_order[i] == current_p) {
        current_idx = i;
        break;
      }
    }

    for (int step = 1; step <= num_players; ++step) {
      int next_idx = (current_idx + step) % num_players;
      uint8_t next_player_id = eclipse_state_.turn_order[next_idx];
      if (next_player_id < eclipse_state_.players.size() && !eclipse_state_.players[next_player_id].has_passed) {
        eclipse_state_.current_player = next_player_id;
        break;
      }
    }
  }
}

} // namespace eclipse
} // namespace open_spiel
