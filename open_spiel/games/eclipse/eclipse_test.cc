#include "open_spiel/games/eclipse/eclipse.h"

#include <algorithm>
#include <vector>

#include "open_spiel/games/eclipse/systems/actions/explore.h"
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

// Builds a minimal raw State with a single player of the given species.
::State MakeSinglePlayerState(Species species) {
  ::State s;
  ::Player player{};
  player.id = 0;
  player.species_id = species;
  player.score = 0;
  player.disks_on_sectors = 0;
  player.disks_on_actions = 0;
  s.players.push_back(player);
  return s;
}

void ExplorePureHelpersTest() {
  // Ring bag bit 0 maps to the first sector of each ring (SECTOR_TABLE order).
  SPIEL_CHECK_EQ(ring_bit_to_sector_id(SectorType::INNER, 0), 101);
  SPIEL_CHECK_EQ(ring_bit_to_sector_id(SectorType::MIDDLE, 0), 201);
  SPIEL_CHECK_EQ(ring_bit_to_sector_id(SectorType::OUTER, 0), 301);
  SPIEL_CHECK_EQ(ring_bit_to_sector_id(SectorType::INNER, 9), 110);  // last inner

  // 6-edge circular rotation of the wormhole mask.
  SPIEL_CHECK_EQ(rotate_edge_mask(0b000001, 1), 0b000010);
  SPIEL_CHECK_EQ(rotate_edge_mask(0b100000, 1), 0b000001);  // wraps around
  SPIEL_CHECK_EQ(rotate_edge_mask(0b011011, 0), 0b011011);  // identity
  SPIEL_CHECK_EQ(rotate_edge_mask(0b111111, 3), 0b111111);
}

void ExploreZoneAndConnectionTest() {
  ::State s = MakeSinglePlayerState(Species::TERRAN_FACTIONS);
  // Anchor controlled by player 0 at the center.
  Sector& anchor = s.galaxy.at(0, 0);
  anchor.sector_id = 221;  // wormholes 0b011011
  anchor.owner_id = 0;
  anchor.coords = {0, 0};
  anchor.rotation = 0;

  // Empty neighbours of the anchor are legal explore zones.
  std::vector<HexCoord> zones = legal_explore_zones(s, 0);
  SPIEL_CHECK_FALSE(zones.empty());

  // A drawn tile placed in the eastern zone has at least one connecting rotation.
  ExploreState& es = s.explore_state;
  es.player_id = 0;
  es.zone_q = 1;
  es.zone_r = 0;
  es.selected_sector_id = 305;  // wormholes 0b001011
  std::vector<uint8_t> rotations = legal_explore_rotations(s, 0);
  SPIEL_CHECK_FALSE(rotations.empty());
}

void ExploreClaimControlTest() {
  ::State s = MakeSinglePlayerState(Species::TERRAN_FACTIONS);
  Sector& cell = s.galaxy.at(1, 0);
  cell.sector_id = 306;  // no ancients, no discovery
  cell.owner_id = 255;
  cell.coords = {1, 0};
  cell.discovery_tile_present = false;

  ExploreState& es = s.explore_state;
  es.phase = ExplorePhase::claim_control;
  es.player_id = 0;
  es.activations_remaining = 1;
  es.zone_q = 1;
  es.zone_r = 0;
  es.selected_sector_id = 306;

  SPIEL_CHECK_TRUE(claim_explore_control(s, 0, /*take_control=*/true));
  SPIEL_CHECK_EQ(s.galaxy.at(1, 0).owner_id, 0);
  SPIEL_CHECK_EQ(s.players[0].disks_on_sectors, 1);
  SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::inactive);
}

void ExploreAncientBlocksControlTest() {
  ::State s = MakeSinglePlayerState(Species::TERRAN_FACTIONS);
  Sector& cell = s.galaxy.at(1, 0);
  cell.sector_id = 305;  // 1 starting ancient
  cell.owner_id = 255;
  cell.coords = {1, 0};
  cell.discovery_tile_present = false;

  ExploreState& es = s.explore_state;
  es.phase = ExplorePhase::claim_control;
  es.player_id = 0;
  es.activations_remaining = 1;
  es.zone_q = 1;
  es.zone_r = 0;
  es.selected_sector_id = 305;

  claim_explore_control(s, 0, /*take_control=*/true);
  SPIEL_CHECK_EQ(s.galaxy.at(1, 0).owner_id, 255);  // ancients block control
  SPIEL_CHECK_EQ(s.players[0].disks_on_sectors, 0);

  // Descendants of Draco may take control of an ancient sector.
  ::State draco = MakeSinglePlayerState(Species::DESCENDANTS_OF_DRACO);
  Sector& dcell = draco.galaxy.at(1, 0);
  dcell.sector_id = 305;
  dcell.owner_id = 255;
  dcell.coords = {1, 0};
  ExploreState& des = draco.explore_state;
  des.phase = ExplorePhase::claim_control;
  des.player_id = 0;
  des.activations_remaining = 1;
  des.zone_q = 1;
  des.zone_r = 0;
  des.selected_sector_id = 305;
  claim_explore_control(draco, 0, /*take_control=*/true);
  SPIEL_CHECK_EQ(draco.galaxy.at(1, 0).owner_id, 0);
}

void ExploreDiscoveryVpTest() {
  ::State s = MakeSinglePlayerState(Species::TERRAN_FACTIONS);
  Sector& cell = s.galaxy.at(1, 0);
  cell.sector_id = 317;  // no ancients, has a discovery tile
  cell.owner_id = 255;
  cell.coords = {1, 0};
  cell.discovery_tile_present = true;

  ExploreState& es = s.explore_state;
  es.phase = ExplorePhase::claim_control;
  es.player_id = 0;
  es.activations_remaining = 1;
  es.zone_q = 1;
  es.zone_r = 0;
  es.selected_sector_id = 317;

  claim_explore_control(s, 0, /*take_control=*/true);
  SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::discovery_reward);

  resolve_explore_discovery(s, 0, /*take_reward=*/false);  // keep 2 VP
  SPIEL_CHECK_EQ(s.players[0].score, 2);
  SPIEL_CHECK_FALSE(s.galaxy.at(1, 0).discovery_tile_present);
  SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::inactive);
}

void ExploreSpeciesRandomSimTest() {
  // Planta (2 explore activations) and Descendants of Draco (draw two tiles per
  // activation) exercise the multi-activation and double-draw chance paths.
  auto game = LoadGame(
      "eclipse",
      {
          {"players", GameParameter(2)},
          {"rng_seed", GameParameter(11)},
          {"species_p0", GameParameter(std::string("Planta"))},
          {"species_p1", GameParameter(std::string("Descendants of Draco"))},
      });
  testing::RandomSimTest(*game, /*num_sims=*/10, /*serialize=*/true,
                         /*verbose=*/false);
}

void ExploreFullActionViaApiTest() {
  auto game = LoadEclipseGame(2, 7);
  auto state = game->NewInitialState();
  state->ApplyAction(0);  // resolve initial setup chance node

  auto* eclipse_state = dynamic_cast<EclipseState*>(state.get());
  SPIEL_CHECK_TRUE(eclipse_state != nullptr);

  std::vector<Action> legal = state->LegalActions();
  const Action explore_start = 17;
  if (std::find(legal.begin(), legal.end(), explore_start) == legal.end()) {
    return;  // no legal explore zones at game start (not expected, but safe)
  }

  state->ApplyAction(explore_start);

  int steps = 0;
  while (eclipse_state->RawState().explore_state.phase != ExplorePhase::inactive &&
         steps < 100) {
    if (state->IsChanceNode()) {
      const ActionsAndProbs outcomes = state->ChanceOutcomes();
      SPIEL_CHECK_FALSE(outcomes.empty());
      double total = 0.0;
      for (const auto& [outcome, prob] : outcomes) total += prob;
      SPIEL_CHECK_FLOAT_NEAR(total, 1.0, 1e-9);
      state->ApplyAction(outcomes[0].first);
    } else {
      std::vector<Action> acts = state->LegalActions();
      SPIEL_CHECK_FALSE(acts.empty());
      state->ApplyAction(acts[0]);  // acts[0] makes progress (place/rotate/claim)
    }
    steps++;
  }
  // The Explore action fully resolves and the turn moves on.
  SPIEL_CHECK_TRUE(eclipse_state->RawState().explore_state.phase ==
                   ExplorePhase::inactive);
  SPIEL_CHECK_FALSE(state->IsChanceNode());
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
  open_spiel::eclipse::ExplorePureHelpersTest();
  open_spiel::eclipse::ExploreZoneAndConnectionTest();
  open_spiel::eclipse::ExploreClaimControlTest();
  open_spiel::eclipse::ExploreAncientBlocksControlTest();
  open_spiel::eclipse::ExploreDiscoveryVpTest();
  open_spiel::eclipse::ExploreSpeciesRandomSimTest();
  open_spiel::eclipse::ExploreFullActionViaApiTest();
}
