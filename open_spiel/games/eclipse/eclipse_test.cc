#include "open_spiel/games/eclipse/eclipse.h"

#include <vector>

#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace eclipse {
namespace {

namespace testing = open_spiel::testing;

std::shared_ptr<const Game> LoadEclipseGame(int players, int rng_seed) {
  return LoadGame("eclipse(players=" + std::to_string(players) +
                  ",rng_seed=" + std::to_string(rng_seed) + ")");
}

void BasicEclipseTests() {
  testing::LoadGameTest("eclipse(players=4,rng_seed=7)");
  testing::ChanceOutcomesTest(*LoadEclipseGame(4, 7));
}

void InitialStateChanceNodeTest() {
  auto game = LoadEclipseGame(4, 7);
  auto state = game->NewInitialState();

  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  const ActionsAndProbs chance_outcomes = state->ChanceOutcomes();
  SPIEL_CHECK_EQ(chance_outcomes.size(), 1);
  SPIEL_CHECK_EQ(chance_outcomes[0].first, 0);
  SPIEL_CHECK_EQ(chance_outcomes[0].second, 1.0);

  state->ApplyAction(0);
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  SPIEL_CHECK_GE(state->CurrentPlayer(), 0);
  SPIEL_CHECK_LT(state->CurrentPlayer(), game->NumPlayers());
}

void RandomSimulationAndSerializationTest() {
  testing::RandomSimTest(*LoadEclipseGame(4, 7), /*num_sims=*/10,
                         /*serialize=*/true, /*verbose=*/false);
}

void DeterministicReplayTest() {
  auto game_a = LoadEclipseGame(4, 17);
  auto game_b = LoadEclipseGame(4, 17);
  auto game_c = LoadEclipseGame(4, 18);

  auto state_a = game_a->NewInitialState();
  auto state_b = game_b->NewInitialState();
  auto state_c = game_c->NewInitialState();

  state_a->ApplyAction(0);
  state_b->ApplyAction(0);
  state_c->ApplyAction(0);

  SPIEL_CHECK_EQ(state_a->ToString(), state_b->ToString());
  SPIEL_CHECK_NE(state_a->ToString(), state_c->ToString());

  auto* eclipse_a = dynamic_cast<EclipseState*>(state_a.get());
  auto* eclipse_b = dynamic_cast<EclipseState*>(state_b.get());
  SPIEL_CHECK_TRUE(eclipse_a != nullptr);
  SPIEL_CHECK_TRUE(eclipse_b != nullptr);

  state_a->ApplyAction(0);
  state_b->ApplyAction(0);
  SPIEL_CHECK_EQ(nlohmann::json(eclipse_a->RawState()).dump(),
                 nlohmann::json(eclipse_b->RawState()).dump());
}

void SetupHelperParityTest() {
  auto game = LoadGame(
      "eclipse",
      {
          {"players", GameParameter(4)},
          {"rng_seed", GameParameter(23)},
          {"species_p0", GameParameter(std::string("Terran Factions"))},
          {"species_p1", GameParameter(std::string("Planta"))},
          {"species_p2", GameParameter(std::string("Orion Hegemony"))},
          {"species_p3", GameParameter(std::string("Hydran Progress"))},
      });
  auto state = game->NewInitialState();
  state->ApplyAction(0);

  SetupConfig config;
  config.players = 4;
  config.rng_seed = 23;
  config.npc_difficulty = NPCDifficulty::EASY;
  config.staged_players = {
      StagedPlayerConfig{.species = Species::TERRAN_FACTIONS, .is_ai = false},
      StagedPlayerConfig{.species = Species::PLANTA, .is_ai = false},
      StagedPlayerConfig{.species = Species::ORION_HEGEMONY, .is_ai = false},
      StagedPlayerConfig{.species = Species::HYDRAN_PROGRESS, .is_ai = false},
  };

  const SetupSnapshot snapshot = CreatePreChoiceSnapshot(config);
  const SetupSnapshot finalized = FinalizeSetupSnapshot(
      snapshot,
      {
          PlayerConfig{.species = Species::TERRAN_FACTIONS, .is_ai = false},
          PlayerConfig{.species = Species::PLANTA, .is_ai = false},
          PlayerConfig{.species = Species::ORION_HEGEMONY, .is_ai = false},
          PlayerConfig{.species = Species::HYDRAN_PROGRESS, .is_ai = false},
      });

  auto* eclipse_state = dynamic_cast<EclipseState*>(state.get());
  SPIEL_CHECK_TRUE(eclipse_state != nullptr);
  SPIEL_CHECK_EQ(nlohmann::json(finalized.state).dump(),
                 nlohmann::json(eclipse_state->RawState()).dump());
}

void AppConfigSnapshotTest() {
  SetupConfig config;
  config.players = 3;
  config.rng_seed = 99;
  config.npc_difficulty = NPCDifficulty::MEDIUM;
  config.staged_players = {
      StagedPlayerConfig{.species = Species::TERRAN_FACTIONS, .is_ai = false},
      StagedPlayerConfig{.species = Species::PLANTA, .is_ai = true},
      StagedPlayerConfig{.species = Species::MECHANEMA, .is_ai = true},
  };

  const SetupSnapshot snapshot = CreatePreChoiceSnapshot(config);
  SPIEL_CHECK_EQ(snapshot.config.players, 3);
  SPIEL_CHECK_EQ(snapshot.config.rng_seed, 99);
  SPIEL_CHECK_EQ(snapshot.state.players.size(), 3);
  SPIEL_CHECK_FALSE(snapshot.finalized);
}

}  // namespace
}  // namespace eclipse
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::eclipse::BasicEclipseTests();
  open_spiel::eclipse::InitialStateChanceNodeTest();
  open_spiel::eclipse::RandomSimulationAndSerializationTest();
  open_spiel::eclipse::DeterministicReplayTest();
  open_spiel::eclipse::SetupHelperParityTest();
  open_spiel::eclipse::AppConfigSnapshotTest();
}
