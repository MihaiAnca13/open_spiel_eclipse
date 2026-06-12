#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/eclipse/eclipse.h"
#include "open_spiel/games/eclipse/systems/setup.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"

namespace py = pybind11;

namespace {

// Build a loaded EclipseGame from the setup_config embedded in a canonical
// game-state blob (EclipseState::Serialize format). Only "players"/"rng_seed"
// affect the loaded game; the full per-player state is restored on deserialize.
std::shared_ptr<const open_spiel::Game> LoadGameFromBlob(
    const nlohmann::json& blob) {
  const auto& cfg = blob.at("setup_config");
  open_spiel::GameParameters params;
  params["players"] = open_spiel::GameParameter(cfg.at("players").get<int>());
  params["rng_seed"] = open_spiel::GameParameter(
      static_cast<int>(cfg.at("rng_seed").get<uint64_t>()));
  return open_spiel::LoadGame("eclipse", params);
}

// Apply one action to the deserialized state, then auto-resolve any chance
// nodes so the returned blob always sits at a human/terminal decision point.
std::string ApplyActionApi(const std::string& blob_json, int action_id) {
  const nlohmann::json blob = nlohmann::json::parse(blob_json);
  std::shared_ptr<const open_spiel::Game> game = LoadGameFromBlob(blob);
  const auto* eclipse_game =
      static_cast<const open_spiel::eclipse::EclipseGame*>(game.get());

  std::unique_ptr<open_spiel::State> state = game->DeserializeState(blob_json);
  state->ApplyAction(static_cast<open_spiel::Action>(action_id));

  while (state->IsChanceNode()) {
    const open_spiel::Action outcome =
        open_spiel::SampleAction(state->ChanceOutcomes(), eclipse_game->rng())
            .first;
    state->ApplyAction(outcome);
  }
  return state->Serialize();
}

// Report the decision facing the current player for a given game-state blob.
std::string LegalActionsApi(const std::string& blob_json) {
  const nlohmann::json blob = nlohmann::json::parse(blob_json);
  std::shared_ptr<const open_spiel::Game> game = LoadGameFromBlob(blob);
  std::unique_ptr<open_spiel::State> state = game->DeserializeState(blob_json);

  nlohmann::json result = nlohmann::json::object();
  const open_spiel::Player current_player = state->CurrentPlayer();
  result["current_player"] = current_player;
  result["is_terminal"] = state->IsTerminal();

  nlohmann::json legal = nlohmann::json::array();
  nlohmann::json action_strings = nlohmann::json::object();
  if (!state->IsTerminal()) {
    for (const open_spiel::Action action : state->LegalActions()) {
      legal.push_back(action);
      action_strings[std::to_string(action)] =
          state->ActionToString(current_player, action);
    }
  }
  result["legal_actions"] = legal;
  result["action_strings"] = action_strings;
  return result.dump();
}

std::string InitializePreChoiceApi(const std::string& config_json) {
  const nlohmann::json config_value = nlohmann::json::parse(config_json);
  const SetupConfig config = config_value.get<SetupConfig>();
  const SetupSnapshot snapshot = CreatePreChoiceSnapshot(config);
  return nlohmann::json(snapshot).dump();
}

std::string FinalizeGameSetupApi(const std::string& snapshot_json,
                                 const std::string& player_choices_json) {
  const nlohmann::json snapshot_value = nlohmann::json::parse(snapshot_json);
  const SetupSnapshot snapshot = snapshot_value.get<SetupSnapshot>();

  const nlohmann::json choices_value = nlohmann::json::parse(player_choices_json);
  const std::vector<PlayerConfig> player_choices =
      choices_value.get<std::vector<PlayerConfig>>();

  return nlohmann::json(FinalizeSetupSnapshot(snapshot, player_choices)).dump();
}

std::string GetGameMetadataApi() {
  nlohmann::json metadata = nlohmann::json::object();

  nlohmann::json species = nlohmann::json::array();
  for (const auto s : ALL_SPECIES) {
    nlohmann::json j = s;
    species.push_back(j.get<std::string>());
  }
  metadata["species"] = species;

  nlohmann::json tech_catalog = nlohmann::json::object();
  const size_t tech_count = sizeof(TECH_TABLE) / sizeof(TECH_TABLE[0]);
  for (size_t i = 0; i < tech_count; ++i) {
    const auto& tech = TECH_TABLE[i];
    nlohmann::json tech_json = nlohmann::json::object();
    tech_json["category"] = tech.category;
    tech_json["order"] = static_cast<uint64_t>(tech.bit);
    tech_json["base_cost"] = tech.base_cost;
    tech_json["min_cost"] = tech.min_cost;
    tech_json["copies"] = tech.copies;
    tech_catalog[tech.name] = tech_json;
  }
  metadata["tech_catalog"] = tech_catalog;

  nlohmann::json npc_difficulties = nlohmann::json::array();
  npc_difficulties.push_back("Easy");
  npc_difficulties.push_back("Medium");
  npc_difficulties.push_back("Hard");
  metadata["npc_difficulties"] = npc_difficulties;

  return metadata.dump();
}

}  // namespace

PYBIND11_MODULE(eclipse_ui_native, module) {
  module.def("initialize_pre_choice", &InitializePreChoiceApi,
             "Create a deterministic pre-choice setup snapshot from UI config.");
  module.def("finalize_game_setup", &FinalizeGameSetupApi,
             "Finalize a pre-choice snapshot with explicit player choices.");
  module.def("get_game_metadata", &GetGameMetadataApi,
             "Return game metadata (species, tech catalog, etc.) from C++ tables.");
  module.def("apply_action", &ApplyActionApi,
             "Apply one action to a game-state blob, auto-resolving chance, and "
             "return the new blob.");
  module.def("legal_actions", &LegalActionsApi,
             "Return {current_player, is_terminal, legal_actions, "
             "action_strings} for a game-state blob.");
}
