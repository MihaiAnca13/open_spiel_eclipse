#include "open_spiel/games/eclipse/eclipse.h"

#include <algorithm>
#include <vector>

#include "open_spiel/games/eclipse/systems/actions/explore.h"
#include "open_spiel/games/eclipse/systems/actions/research.h"
#include "open_spiel/games/eclipse/systems/actions/build.h"
#include "open_spiel/games/eclipse/systems/actions/influence.h"
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

Action FindActionByName(const State& state, const std::string& name) {
  for (Action action : state.LegalActions()) {
    if (state.ActionToString(state.CurrentPlayer(), action) == name) {
      return action;
    }
  }
  return -1;
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
  s.sector_bag_inner = (1u << 10) - 1;  // inner ring stocked (neighbours of center)
  // Anchor controlled by player 0 at the center.
  Sector& anchor = s.galaxy.at(0, 0);
  anchor.sector_id = 221;  // wormholes 0b011011
  anchor.owner_id = 0;
  anchor.coords = {0, 0};
  anchor.rotation = 0;

  // Empty neighbours of the anchor are legal explore zones.
  std::vector<HexCoord> zones = legal_explore_zones(s, 0);
  SPIEL_CHECK_FALSE(zones.empty());

  // A drawn tile placed in a connected neighbour zone has at least one rotation.
  ExploreState& es = s.explore_state;
  es.player_id = 0;
  es.zone_q = 1;
  es.zone_r = -1;
  es.selected_sector_id = 305;  // wormholes 0b110100
  std::vector<uint8_t> rotations = legal_explore_rotations(s, 0);
  SPIEL_CHECK_FALSE(rotations.empty());
}

void ExploreExhaustedRingTest() {
  ::State s = MakeSinglePlayerState(Species::TERRAN_FACTIONS);
  Sector& anchor = s.galaxy.at(0, 0);  // neighbours are all distance 1 (inner ring)
  anchor.sector_id = 221;
  anchor.owner_id = 0;
  anchor.coords = {0, 0};

  // Empty inner bag: an exhausted ring cannot be explored (no reshuffle).
  s.sector_bag_inner = 0;
  SPIEL_CHECK_FALSE(is_legal_explore_zone(s, 0, 1, 0));
  SPIEL_CHECK_TRUE(legal_explore_zones(s, 0).empty());
  SPIEL_CHECK_FALSE(has_explore_zone(s, 0));

  // With one inner tile left, the ring is explorable again.
  s.sector_bag_inner = 1;
  SPIEL_CHECK_TRUE(is_legal_explore_zone(s, 0, 1, -1));
  SPIEL_CHECK_TRUE(has_explore_zone(s, 0));
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

void ExploreStopAndDracoDrawTest() {
  // hex<->index bijection round-trips.
  for (int idx : {0, 7, 112, 224}) {
    HexCoord h = index_to_hex(idx);
    SPIEL_CHECK_EQ(hex_to_index(h.q, h.r), idx);
  }

  // choose_explore_zone accepts a legal hex and rejects an illegal one.
  {
    ::State s = MakeSinglePlayerState(Species::TERRAN_FACTIONS);
    s.sector_bag_inner = (1u << 10) - 1;  // inner ring (neighbours of center) stocked
    Sector& anchor = s.galaxy.at(0, 0);
    anchor.sector_id = 221;
    anchor.owner_id = 0;
    anchor.coords = {0, 0};
    s.explore_state.phase = ExplorePhase::choose_zone;
    s.explore_state.player_id = 0;
    SPIEL_CHECK_TRUE(is_legal_explore_zone(s, 0, 1, -1));   // adjacent, connected, empty
    SPIEL_CHECK_FALSE(is_legal_explore_zone(s, 0, 4, 4));   // not adjacent
    SPIEL_CHECK_FALSE(choose_explore_zone(s, 0, HexCoord{4, 4}));
    SPIEL_CHECK_TRUE(choose_explore_zone(s, 0, HexCoord{1, -1}));
    SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::draw_tile);
  }

  // Stopping ends the Explore action regardless of remaining activations.
  {
    ::State s = MakeSinglePlayerState(Species::PLANTA);
    s.explore_state.phase = ExplorePhase::choose_zone;
    s.explore_state.player_id = 0;
    s.explore_state.activations_remaining = 2;
    stop_exploring(s);
    SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::inactive);
  }

  // Draco draws one tile, may stop there, or draw a second and choose one.
  {
    ::State s = MakeSinglePlayerState(Species::DESCENDANTS_OF_DRACO);
    s.sector_bag_outer = 0b11;  // two outer tiles available (bits 0 and 1)
    s.explore_state.player_id = 0;
    s.explore_state.ring = SectorType::OUTER;
    s.explore_state.phase = ExplorePhase::draw_tile;

    apply_explore_draw(s, /*ring_bit=*/0);  // flips outer bit 0 -> sector 301
    SPIEL_CHECK_EQ(s.explore_state.drawn_count, 1);
    SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::draw_again_decision);

    SPIEL_CHECK_TRUE(skip_second_draw(s, 0));
    SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::place_or_discard);
    SPIEL_CHECK_EQ(s.explore_state.selected_sector_id, 301);
  }

  {
    ::State s = MakeSinglePlayerState(Species::DESCENDANTS_OF_DRACO);
    s.sector_bag_outer = 0b11;  // two outer tiles available (bits 0 and 1)
    s.explore_state.player_id = 0;
    s.explore_state.ring = SectorType::OUTER;
    s.explore_state.phase = ExplorePhase::draw_tile;

    apply_explore_draw(s, /*ring_bit=*/0);  // flips outer bit 0 -> sector 301
    SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::draw_again_decision);
    SPIEL_CHECK_TRUE(draw_again(s, 0));
    SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::draw_tile);

    apply_explore_draw(s, /*ring_bit=*/1);  // flips outer bit 1 -> sector 302
    SPIEL_CHECK_EQ(s.explore_state.drawn_count, 2);
    SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::select_drawn_tile);
    SPIEL_CHECK_TRUE(select_drawn_tile(s, 0, 0));
    SPIEL_CHECK_TRUE(s.explore_state.phase == ExplorePhase::place_or_discard);
    SPIEL_CHECK_EQ(s.explore_state.selected_sector_id, 301);
  }
}

// Drives an Explore action through the public API, preferring progressing
// actions so the full place -> rotate -> claim path is exercised.
void ExploreFullActionViaApiTest() {
  auto game = LoadEclipseGame(2, 7);
  auto state = game->NewInitialState();
  state->ApplyAction(0);  // resolve initial setup chance node

  auto* eclipse_state = dynamic_cast<EclipseState*>(state.get());
  SPIEL_CHECK_TRUE(eclipse_state != nullptr);

  const Action explore_start = FindActionByName(*state, "EXPLORE");
  if (explore_start < 0) {
    return;  // no legal explore zones at game start (not expected, but safe)
  }

  state->ApplyAction(explore_start);
  SPIEL_CHECK_EQ(eclipse_state->RawState().players[0].disks_on_actions, 1);
  bool placed = false;

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
      // Prefer a progressing action over stop/discard/skip so we reach placement.
      Action chosen = acts[0];
      for (Action a : acts) {
        std::string name = state->ActionToString(state->CurrentPlayer(), a);
        if (name.rfind("EXPLORE_STOP", 0) != 0 &&
            name.rfind("EXPLORE_DISCARD", 0) != 0 &&
            name.rfind("EXPLORE_SKIP", 0) != 0) {
          chosen = a;
          break;
        }
      }
      if (eclipse_state->RawState().explore_state.phase ==
          ExplorePhase::choose_rotation) {
        placed = true;
      }
      state->ApplyAction(chosen);
    }
    steps++;
  }
  // The Explore action fully resolves, a tile was placed, and the turn moves on.
  SPIEL_CHECK_TRUE(placed);
  SPIEL_CHECK_TRUE(eclipse_state->RawState().explore_state.phase ==
                   ExplorePhase::inactive);
  SPIEL_CHECK_FALSE(state->IsChanceNode());
  SPIEL_CHECK_EQ(eclipse_state->RawState().players[0].disks_on_actions, 1);
}

void ResearchRareTechTrackTest() {
  ::State s = MakeSinglePlayerState(Species::TERRAN_FACTIONS);
  ::Player& player = s.players[0];
  player.resources.science = 20;
  player.researched_techs_military = static_cast<uint64_t>(TechBit::NEUTRON_BOMBS);
  player.researched_techs_grid = static_cast<uint64_t>(TechBit::GAUSS_SHIELD);
  player.researched_techs_nano = 0;
  s.add_to_tech_tray(TechBit::ABSORPTION_SHIELD);

  const TechDefinition& rare = TECH_TABLE[TECH_TABLE_SIZE];
  SPIEL_CHECK_EQ(static_cast<uint64_t>(rare.bit),
                 static_cast<uint64_t>(TechBit::ABSORPTION_SHIELD));
  SPIEL_CHECK_TRUE(research_tech(s, 0, rare, TechCategory::GRID));

  const uint64_t rare_bit = static_cast<uint64_t>(TechBit::ABSORPTION_SHIELD);
  SPIEL_CHECK_FALSE((player.researched_techs_military & rare_bit) != 0);
  SPIEL_CHECK_TRUE((player.researched_techs_grid & rare_bit) != 0);
  SPIEL_CHECK_FALSE((player.researched_techs_nano & rare_bit) != 0);
  SPIEL_CHECK_EQ(get_track_tile_count(player, TechCategory::MILITARY), 1);
  SPIEL_CHECK_EQ(get_track_tile_count(player, TechCategory::GRID), 2);
  SPIEL_CHECK_EQ(get_track_tile_count(player, TechCategory::NANO), 0);
}

void ResearchInfluenceDiscRewardsTest() {
  std::shared_ptr<const Game> game = LoadEclipseGame(2, 7);
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(0);

  auto* eclipse_state = static_cast<EclipseState*>(state.get());
  ::State raw = eclipse_state->RawState();
  raw.current_player = 0;
  raw.players[0].has_passed = false;
  raw.players[0].resources.science = 20;
  raw.players[0].disks_on_actions = 0;
  raw.players[0].extra_influence_discs = 0;
  raw.players[0].researched_techs_grid = 0;
  raw.tech_tray.fill(0);
  raw.add_to_tech_tray(TechBit::QUANTUM_GRID);

  const auto eclipse_game = std::static_pointer_cast<const EclipseGame>(game);
  eclipse_state->RestoreFromSnapshot(eclipse_game->InitialSetupConfig(), raw,
                                     EclipseState::PendingRandomEvent::none);

  const uint8_t initial_available =
      eclipse_state->RawState().players[0].available_influence_discs();
  const Action research_action = FindActionByName(*state, "RESEARCH");
  SPIEL_CHECK_GE(research_action, 0);
  state->ApplyAction(research_action);

  const Action quantum_grid_action = FindActionByName(*state, "RESEARCH_Quantum Grid");
  SPIEL_CHECK_GE(quantum_grid_action, 0);
  state->ApplyAction(quantum_grid_action);

  const ::Player& player = eclipse_state->RawState().players[0];
  SPIEL_CHECK_EQ(player.disks_on_sectors, raw.players[0].disks_on_sectors);
  SPIEL_CHECK_EQ(player.extra_influence_discs, 2);
  SPIEL_CHECK_EQ(player.available_influence_discs(), initial_available + 1);
}

void ResearchActionTest() {
  std::shared_ptr<const Game> game = LoadGame("eclipse");
  std::unique_ptr<State> state = game->NewInitialState();

  // Run through initial setup
  while (state->IsChanceNode()) {
    const ActionsAndProbs outcomes = state->ChanceOutcomes();
    state->ApplyAction(outcomes[0].first);
  }

  auto* eclipse_state = static_cast<EclipseState*>(state.get());
  const ::State& s = eclipse_state->RawState();
  uint8_t acting_player = state->CurrentPlayer();

  // Acting player should have some starting science
  SPIEL_CHECK_GT(s.players[acting_player].resources.science, 0);

  // Get legal actions - should include RESEARCH if player has >=2 science
  std::vector<Action> legal = state->LegalActions();
  bool has_research = false;
  for (Action a : legal) {
    std::string name = state->ActionToString(state->CurrentPlayer(), a);
    if (name == "RESEARCH") {
      has_research = true;
      break;
    }
  }
  SPIEL_CHECK_TRUE(has_research);

  // Find the RESEARCH action
  Action research_action = -1;
  for (Action a : legal) {
    std::string name = state->ActionToString(state->CurrentPlayer(), a);
    if (name == "RESEARCH") {
      research_action = a;
      break;
    }
  }
  SPIEL_CHECK_NE(research_action, -1);

  // Start research action
  state->ApplyAction(research_action);
  eclipse_state = static_cast<EclipseState*>(state.get());
  const ::State& s2 = eclipse_state->RawState();

  // Should now be in research phase
  SPIEL_CHECK_EQ(s2.research_state.phase, ResearchState::Phase::choose_tech);
  SPIEL_CHECK_EQ(s2.research_state.player_id, acting_player);

  // Get legal research actions
  legal = state->LegalActions();
  SPIEL_CHECK_FALSE(legal.empty());

  // Find a standard tech that's affordable
  Action chosen_tech = -1;
  for (Action a : legal) {
    std::string name = state->ActionToString(state->CurrentPlayer(), a);
    if (name.rfind("RESEARCH_", 0) == 0 && name != "RESEARCH_STOP") {
      // Check if it's a standard tech (not rare with _ON_ suffix)
      if (name.find("_ON_") == std::string::npos) {
        chosen_tech = a;
        break;
      }
    }
  }

  if (chosen_tech != -1) {
    // Record initial state
    uint8_t initial_science = s2.players[acting_player].resources.science;
    uint64_t initial_techs = s2.players[acting_player].researched_techs_military |
                             s2.players[acting_player].researched_techs_grid |
                             s2.players[acting_player].researched_techs_nano;

    // Research the tech
    state->ApplyAction(chosen_tech);
    eclipse_state = static_cast<EclipseState*>(state.get());
    const ::State& s3 = eclipse_state->RawState();

    // Verify tech was added (at least one mask changed)
    uint64_t final_techs = s3.players[acting_player].researched_techs_military |
                           s3.players[acting_player].researched_techs_grid |
                           s3.players[acting_player].researched_techs_nano;
    SPIEL_CHECK_NE(final_techs, initial_techs);
    // Science should have decreased
    SPIEL_CHECK_LT(s3.players[acting_player].resources.science, initial_science);

    // Research phase should be inactive (single activation species)
    SPIEL_CHECK_EQ(s3.research_state.phase, ResearchState::Phase::inactive);
  }
}

void InfluenceReclaimCubesTest() {
  // Test reclaim logic with wildcards, orbitals, and saturation cascades
  ::State s = MakeSinglePlayerState(Species::TERRAN_FACTIONS);
  ::Player& p = s.players[0];

  // Set up starting resources & tracks
  p.resources.gold_prod = 5;
  p.resources.science_prod = 5;
  p.resources.materials_prod = 5;

  // Let's place an influence disc on a sector and colonize slots
  Sector& cell = s.galaxy.at(1, 0);
  cell.sector_id = 101; // Castor: ADV_MATERIALS, MATERIALS, MONEY
  cell.owner_id = 0;
  cell.coords = {1, 0};
  p.disks_on_sectors = 1;

  // Colonize Castor slots: ADV_MATERIALS (index 0) and MONEY (index 2)
  cell.occupied_slots_mask = (1u << 0) | (1u << 2);

  // Reclaim disc from Castor!
  // Normal return: ADV_MATERIALS (matching Materials track), MONEY (matching Money track)
  // Since both tracks have 5 cubes (< 12), they should be auto-returned immediately.
  SPIEL_CHECK_TRUE(execute_reclaim_from_sector(s, 0, hex_to_index(1, 0)));
  SPIEL_CHECK_EQ(cell.owner_id, 255);
  SPIEL_CHECK_EQ(cell.occupied_slots_mask, 0);
  SPIEL_CHECK_EQ(p.disks_on_sectors, 0);
  // Track values should have incremented by 1
  SPIEL_CHECK_EQ(p.resources.gold_prod, 6);
  SPIEL_CHECK_EQ(p.resources.materials_prod, 6);
  SPIEL_CHECK_EQ(p.resources.science_prod, 5);
  SPIEL_CHECK_TRUE(s.influence_state.phase == InfluenceState::Phase::inactive);

  // Let's test wildcards (ANY) on Mu Cassiopeiae (sector 108)
  Sector& cell2 = s.galaxy.at(0, 1);
  cell2.sector_id = 108; // Mu Cassiopeiae: ADV_MONEY, SCIENCE, ANY
  cell2.owner_id = 0;
  cell2.coords = {0, 1};
  p.disks_on_sectors = 1;

  // Occupy the ANY slot (index 2)
  cell2.occupied_slots_mask = (1u << 2);

  // Reset tracks to 5
  p.resources.gold_prod = 5;
  p.resources.science_prod = 5;
  p.resources.materials_prod = 5;

  SPIEL_CHECK_TRUE(execute_reclaim_from_sector(s, 0, hex_to_index(0, 1)));
  // Should transition to choose_return_track phase!
  SPIEL_CHECK_TRUE(s.influence_state.phase == InfluenceState::Phase::choose_return_track);
  SPIEL_CHECK_EQ(s.influence_state.pending_returns.size(), 1);
  SPIEL_CHECK_TRUE(s.influence_state.pending_returns[0].type == PlanetType::ANY);

  // Let's try to choose an invalid track or valid track
  // We choose materials (track 2)
  SPIEL_CHECK_TRUE(execute_choose_return_track(s, 0, 2)); // PopTrack::MATERIALS
  SPIEL_CHECK_EQ(p.resources.materials_prod, 6);
  SPIEL_CHECK_EQ(p.resources.gold_prod, 5);
  SPIEL_CHECK_EQ(p.resources.science_prod, 5);
  SPIEL_CHECK_TRUE(s.influence_state.phase == InfluenceState::Phase::inactive);

  // Let's test Orbitals
  Sector& cell3 = s.galaxy.at(2, 0);
  cell3.sector_id = 201; // Alpha Centauri: MATERIALS, MONEY
  cell3.owner_id = 0;
  cell3.coords = {2, 0};
  cell3.orbital_built = true;
  p.disks_on_sectors = 1;

  // Occupy the orbital slot (index 2, which is def->slots.size() since Alpha Centauri has 2 slots)
  cell3.occupied_slots_mask = (1u << 2);

  p.resources.gold_prod = 5;
  p.resources.science_prod = 5;
  p.resources.materials_prod = 5;

  SPIEL_CHECK_TRUE(execute_reclaim_from_sector(s, 0, hex_to_index(2, 0)));
  SPIEL_CHECK_TRUE(s.influence_state.phase == InfluenceState::Phase::choose_return_track);
  SPIEL_CHECK_EQ(s.influence_state.pending_returns.size(), 1);
  SPIEL_CHECK_TRUE(s.influence_state.pending_returns[0].is_orbital);

  // For orbital, PopTrack::MATERIALS (2) should be illegal because orbital only allows gold/science
  SPIEL_CHECK_FALSE(execute_choose_return_track(s, 0, 2));

  // Choose gold (track 0)
  SPIEL_CHECK_TRUE(execute_choose_return_track(s, 0, 0));
  SPIEL_CHECK_EQ(p.resources.gold_prod, 6);
  SPIEL_CHECK_EQ(p.resources.science_prod, 5);
  SPIEL_CHECK_TRUE(s.influence_state.phase == InfluenceState::Phase::inactive);

  // Let's test saturation/overflow!
  Sector& cell4 = s.galaxy.at(0, 2);
  cell4.sector_id = 201; // Alpha Centauri: MATERIALS, MONEY
  cell4.owner_id = 0;
  cell4.coords = {0, 2};
  p.disks_on_sectors = 1;

  // Occupy MONEY slot (index 1)
  cell4.occupied_slots_mask = (1u << 1);

  // Make MONEY track fully saturated (12)
  p.resources.gold_prod = 12;
  p.resources.science_prod = 5;
  p.resources.materials_prod = 5;

  SPIEL_CHECK_TRUE(execute_reclaim_from_sector(s, 0, hex_to_index(0, 2)));
  // Since gold_prod is 12, it must trigger a choice for the spillover!
  SPIEL_CHECK_TRUE(s.influence_state.phase == InfluenceState::Phase::choose_return_track);
  SPIEL_CHECK_EQ(s.influence_state.pending_returns.size(), 1);

  // We should not be able to choose track 0 (gold_prod is full)
  SPIEL_CHECK_FALSE(execute_choose_return_track(s, 0, 0));

  // Choose science (track 1)
  SPIEL_CHECK_TRUE(execute_choose_return_track(s, 0, 1));
  SPIEL_CHECK_EQ(p.resources.gold_prod, 12);
  SPIEL_CHECK_EQ(p.resources.science_prod, 6);
  SPIEL_CHECK_TRUE(s.influence_state.phase == InfluenceState::Phase::inactive);
}

void InfluenceFullActionTest() {
  std::shared_ptr<const Game> game = LoadEclipseGame(2, 7);
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(0); // resolve setup

  EclipseState* eclipse_state = static_cast<EclipseState*>(state.get());
  ::State& raw = const_cast<::State&>(eclipse_state->RawState());
  raw.current_player = 0;
  raw.players[0].has_passed = false;
  raw.players[0].resources.gold_prod = 5;
  raw.players[0].resources.science_prod = 5;
  raw.players[0].resources.materials_prod = 5;

  // Let's make sure the INFLUENCE action is available
  std::vector<Action> legal = state->LegalActions();
  Action influence_act = -1;
  for (Action a : legal) {
    if (state->ActionToString(state->CurrentPlayer(), a) == "INFLUENCE") {
      influence_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(influence_act, -1);

  // Apply INFLUENCE action!
  state->ApplyAction(influence_act);
  SPIEL_CHECK_TRUE(raw.influence_state.phase == InfluenceState::Phase::choose_influence);
  SPIEL_CHECK_EQ(raw.players[0].disks_on_actions, 1); // Spends 1 disc

  // Let's set up cell (1, 0) to be owned and colonized with wildcard ANY
  Sector& cell = raw.galaxy.at(1, 0);
  cell.sector_id = 108; // Mu Cassiopeiae: ADV_MONEY, SCIENCE, ANY
  cell.owner_id = 0;
  cell.coords = {1, 0};
  cell.occupied_slots_mask = (1u << 2); // ANY slot colonized
  raw.players[0].disks_on_sectors = 1;

  // Get current legal actions: should contain RECLAIM_FROM_1_0
  legal = state->LegalActions();
  Action reclaim_act = -1;
  for (Action a : legal) {
    std::string name = state->ActionToString(state->CurrentPlayer(), a);
    if (name == "RECLAIM_FROM_1_0") {
      reclaim_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(reclaim_act, -1);

  // Apply reclaim action!
  state->ApplyAction(reclaim_act);

  // Should transition to choose_return_track and expect a choice of track (0, 1, 2)
  SPIEL_CHECK_TRUE(raw.influence_state.phase == InfluenceState::Phase::choose_return_track);
  legal = state->LegalActions();
  SPIEL_CHECK_EQ(legal.size(), 3); // Money, Science, Materials all have space
  SPIEL_CHECK_EQ(state->ActionToString(state->CurrentPlayer(), legal[0]), "RETURN_CUBE_TO_MONEY");
  SPIEL_CHECK_EQ(state->ActionToString(state->CurrentPlayer(), legal[1]), "RETURN_CUBE_TO_SCIENCE");
  SPIEL_CHECK_EQ(state->ActionToString(state->CurrentPlayer(), legal[2]), "RETURN_CUBE_TO_MATERIALS");

  // Choose Materials (index 2, which maps to legal[2])
  state->ApplyAction(legal[2]);

  // Mu Cassiopeiae should now be vacant and owned by 255
  SPIEL_CHECK_EQ(cell.owner_id, 255);
  SPIEL_CHECK_EQ(cell.occupied_slots_mask, 0);
  SPIEL_CHECK_EQ(raw.players[0].disks_on_sectors, 0);
  SPIEL_CHECK_EQ(raw.players[0].resources.materials_prod, 6);

  // Activations remaining should have decremented from 2 to 1 (since Terrans have 2 influence activations)
  SPIEL_CHECK_EQ(raw.influence_state.activations_remaining, 1);
  SPIEL_CHECK_TRUE(raw.influence_state.phase == InfluenceState::Phase::choose_influence);

  // Stop early!
  legal = state->LegalActions();
  Action stop_act = -1;
  for (Action a : legal) {
    if (state->ActionToString(state->CurrentPlayer(), a) == "INFLUENCE_STOP") {
      stop_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(stop_act, -1);
  state->ApplyAction(stop_act);

  // Phase should now be inactive and turn should have advanced to player 1
  SPIEL_CHECK_TRUE(raw.influence_state.phase == InfluenceState::Phase::inactive);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

void BuildFullActionTest() {
  std::shared_ptr<const Game> game = LoadEclipseGame(2, 7);
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(0); // resolve setup

  EclipseState* eclipse_state = static_cast<EclipseState*>(state.get());
  ::State& raw = const_cast<::State&>(eclipse_state->RawState());
  raw.current_player = 0;
  raw.players[0].has_passed = false;
  raw.players[0].resources.materials = 10;

  // Verify BUILD action is available
  std::vector<Action> legal = state->LegalActions();
  Action build_act = -1;
  for (Action a : legal) {
    if (state->ActionToString(state->CurrentPlayer(), a) == "BUILD") {
      build_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(build_act, -1);

  // Apply BUILD action
  state->ApplyAction(build_act);
  SPIEL_CHECK_TRUE(raw.build_state.phase == BuildState::Phase::choose_build);
  SPIEL_CHECK_EQ(raw.players[0].disks_on_actions, 1);

  // Build an Interceptor (BuildType 0) at cell 221 (Procyon starting sector for player 0)
  // materials cost is 3 for Terrans.
  legal = state->LegalActions();
  Action interceptor_build = -1;
  for (Action a : legal) {
    std::string name = state->ActionToString(state->CurrentPlayer(), a);
    if (name.find("BUILD_INTERCEPTOR_") != std::string::npos) {
      interceptor_build = a;
      break;
    }
  }
  SPIEL_CHECK_NE(interceptor_build, -1);

  // Apply build action!
  state->ApplyAction(interceptor_build);

  // Remaining activations should be 1 (Terrans have 2 activations)
  SPIEL_CHECK_EQ(raw.build_state.activations_remaining, 1);
  SPIEL_CHECK_EQ(raw.players[0].resources.materials, 7); // 10 - 3

  // Stop early
  legal = state->LegalActions();
  Action stop_act = -1;
  for (Action a : legal) {
    if (state->ActionToString(state->CurrentPlayer(), a) == "BUILD_STOP") {
      stop_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(stop_act, -1);
  state->ApplyAction(stop_act);

  // Build phase should now be inactive, and turn should advance to player 1
  SPIEL_CHECK_TRUE(raw.build_state.phase == BuildState::Phase::inactive);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

void UpgradeFullActionTest() {
  std::shared_ptr<const Game> game = LoadEclipseGame(2, 7);
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(0); // resolve setup

  EclipseState* eclipse_state = static_cast<EclipseState*>(state.get());
  ::State& raw = const_cast<::State&>(eclipse_state->RawState());
  raw.current_player = 0;
  raw.players[0].has_passed = false;
  raw.players[0].resources.materials = 10;
  raw.players[0].resources.science = 10;

  // Verify UPGRADE action is available
  std::vector<Action> legal = state->LegalActions();
  Action upgrade_act = -1;
  for (Action a : legal) {
    if (state->ActionToString(state->CurrentPlayer(), a) == "UPGRADE") {
      upgrade_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(upgrade_act, -1);

  // Apply UPGRADE action
  state->ApplyAction(upgrade_act);
  SPIEL_CHECK_TRUE(raw.upgrade_state.phase == UpgradeState::Phase::choose_upgrade);
  SPIEL_CHECK_EQ(raw.players[0].disks_on_actions, 1);

  // Upgrade Interceptor slot 0 (add a part if legal, or remove if legal)
  legal = state->LegalActions();
  Action interceptor_slot0 = -1;
  for (Action a : legal) {
    std::string name = state->ActionToString(state->CurrentPlayer(), a);
    if (name == "UPGRADE_INTERCEPTOR_SLOT0") {
      interceptor_slot0 = a;
      break;
    }
  }
  SPIEL_CHECK_NE(interceptor_slot0, -1);

  // Apply upgrade action
  state->ApplyAction(interceptor_slot0);

  // Activations should decrement only if a part was placed (not for removal)
  // Stop early
  legal = state->LegalActions();
  Action stop_act = -1;
  for (Action a : legal) {
    if (state->ActionToString(state->CurrentPlayer(), a) == "UPGRADE_STOP") {
      stop_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(stop_act, -1);
  state->ApplyAction(stop_act);

  // Upgrade phase should now be inactive, and turn should advance to player 1
  SPIEL_CHECK_TRUE(raw.upgrade_state.phase == UpgradeState::Phase::inactive);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

void MoveFullActionTest() {
  std::shared_ptr<const Game> game = LoadEclipseGame(2, 7);
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(0); // resolve setup

  EclipseState* eclipse_state = static_cast<EclipseState*>(state.get());
  ::State& raw = const_cast<::State&>(eclipse_state->RawState());
  raw.current_player = 0;
  raw.players[0].has_passed = false;

  // Verify MOVE action is available
  std::vector<Action> legal = state->LegalActions();
  Action move_act = -1;
  for (Action a : legal) {
    if (state->ActionToString(state->CurrentPlayer(), a) == "MOVE") {
      move_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(move_act, -1);

  // Apply MOVE action
  state->ApplyAction(move_act);
  SPIEL_CHECK_TRUE(raw.move_state.phase == MoveState::Phase::choose_move);
  SPIEL_CHECK_EQ(raw.players[0].disks_on_actions, 1);

  // Check if any legal move steps exist (depends on ship positions)
  legal = state->LegalActions();
  Action stop_act = -1;
  for (Action a : legal) {
    if (state->ActionToString(state->CurrentPlayer(), a) == "MOVE_STOP") {
      stop_act = a;
      break;
    }
  }
  SPIEL_CHECK_NE(stop_act, -1);
  state->ApplyAction(stop_act);

  // Move phase should now be inactive, and turn should advance to player 1
  SPIEL_CHECK_TRUE(raw.move_state.phase == MoveState::Phase::inactive);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
}

void StrictMainActionFilteringTest() {
  std::shared_ptr<const Game> game = LoadEclipseGame(2, 7);
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(0);  // resolve setup

  EclipseState* eclipse_state = static_cast<EclipseState*>(state.get());
  ::State& raw = const_cast<::State&>(eclipse_state->RawState());
  raw.current_player = 0;
  raw.players[0].has_passed = false;
  raw.players[0].resources.science = 0;
  raw.players[0].resources.materials = 0;
  raw.unit_registry.clear();

  for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
    for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
      Sector& sector = raw.galaxy.at(q, r);
      sector.owner_id = 255;
      sector.occupied_slots_mask = 0;
    }
  }
  raw.players[0].disks_on_sectors = 0;

  const std::vector<Action> legal = state->LegalActions();
  std::vector<std::string> names;
  names.reserve(legal.size());
  for (Action action : legal) {
    names.push_back(state->ActionToString(state->CurrentPlayer(), action));
  }

  SPIEL_CHECK_EQ(std::find(names.begin(), names.end(), "RESEARCH"), names.end());
  SPIEL_CHECK_EQ(std::find(names.begin(), names.end(), "BUILD"), names.end());
  SPIEL_CHECK_EQ(std::find(names.begin(), names.end(), "INFLUENCE"), names.end());
  SPIEL_CHECK_EQ(std::find(names.begin(), names.end(), "MOVE"), names.end());
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
  open_spiel::eclipse::ExploreExhaustedRingTest();
  open_spiel::eclipse::ExploreClaimControlTest();
  open_spiel::eclipse::ExploreAncientBlocksControlTest();
  open_spiel::eclipse::ExploreDiscoveryVpTest();
  open_spiel::eclipse::ExploreStopAndDracoDrawTest();
  open_spiel::eclipse::ExploreSpeciesRandomSimTest();
  open_spiel::eclipse::ExploreFullActionViaApiTest();
  open_spiel::eclipse::ResearchRareTechTrackTest();
  open_spiel::eclipse::ResearchInfluenceDiscRewardsTest();
  open_spiel::eclipse::ResearchActionTest();
  open_spiel::eclipse::InfluenceReclaimCubesTest();
  open_spiel::eclipse::InfluenceFullActionTest();
  open_spiel::eclipse::BuildFullActionTest();
  open_spiel::eclipse::UpgradeFullActionTest();
  open_spiel::eclipse::MoveFullActionTest();
  open_spiel::eclipse::StrictMainActionFilteringTest();
}
