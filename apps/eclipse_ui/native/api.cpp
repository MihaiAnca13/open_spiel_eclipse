#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "open_spiel/games/eclipse/systems/setup.h"
#include "open_spiel/json/include/nlohmann/json.hpp"

namespace py = pybind11;

namespace {

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
}
