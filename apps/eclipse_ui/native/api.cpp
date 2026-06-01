#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "open_spiel/games/eclipse/systems/setup.h"
#include "open_spiel/json/include/nlohmann/json.hpp"

namespace py = pybind11;

namespace {

std::string InitializePreChoiceApi(unsigned int seed, uint8_t num_players,
                                   const std::string& difficulty_str) {
  NPCDifficulty difficulty = NPCDifficulty::EASY;
  if (difficulty_str == "Medium") {
    difficulty = NPCDifficulty::MEDIUM;
  } else if (difficulty_str == "Hard") {
    difficulty = NPCDifficulty::HARD;
  }

  std::mt19937_64 rng(seed);
  State state = initialize_pre_choice_state(rng, num_players, difficulty);
  nlohmann::json json_state = state;
  return json_state.dump();
}

std::string FinalizeGameSetupApi(unsigned int seed, const std::string& state_json,
                                 const std::string& player_choices_json) {
  nlohmann::json state_value = nlohmann::json::parse(state_json);
  State state = state_value.get<State>();

  nlohmann::json choices_value = nlohmann::json::parse(player_choices_json);
  std::vector<PlayerConfig> choices =
      choices_value.get<std::vector<PlayerConfig>>();

  std::mt19937_64 rng(seed);
  finalize_game_setup(rng, state, choices);

  nlohmann::json result = state;
  return result.dump();
}

}  // namespace

PYBIND11_MODULE(eclipse_ui_native, module) {
  module.def("initialize_pre_choice", &InitializePreChoiceApi,
             "Stage 1 setup that returns serialized pre-choice state JSON.");
  module.def("finalize_game_setup", &FinalizeGameSetupApi,
             "Stage 2 setup that returns finalized state JSON.");
}
