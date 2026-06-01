#include "open_spiel/games/eclipse/eclipse.h"

#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace eclipse {
namespace {

namespace testing = open_spiel::testing;

void BasicEclipseTests() {
  testing::LoadGameTest("eclipse(players=4,seed=7)");
  testing::NoChanceOutcomesTest(*LoadGame("eclipse(players=4,seed=7)"));
}

void InitialStateTests() {
  auto game = LoadGame("eclipse(players=4,seed=7)");
  auto state = game->NewInitialState();

  SPIEL_CHECK_FALSE(state->IsTerminal());
  SPIEL_CHECK_GE(state->CurrentPlayer(), 0);
  SPIEL_CHECK_LT(state->CurrentPlayer(), game->NumPlayers());

  std::vector<Action> actions = state->LegalActions();
  SPIEL_CHECK_FALSE(actions.empty());
  SPIEL_CHECK_EQ(actions.front(), 0);
  SPIEL_CHECK_GT(game->NumDistinctActions(), actions.back());

  for (int step = 0; step < 15 && !state->IsTerminal(); ++step) {
    std::vector<Action> legal_actions = state->LegalActions();
    SPIEL_CHECK_FALSE(legal_actions.empty());
    Action action =
        legal_actions.size() > 1 ? legal_actions[1] : legal_actions[0];
    state->ApplyAction(action);
  }
}

void PassRotationTest() {
  auto game = LoadGame("eclipse(players=2,seed=11)");
  auto state = game->NewInitialState();

  Player initial_player = state->CurrentPlayer();
  state->ApplyAction(0);
  SPIEL_CHECK_NE(state->CurrentPlayer(), initial_player);
}

}  // namespace
}  // namespace eclipse
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::eclipse::BasicEclipseTests();
  open_spiel::eclipse::InitialStateTests();
  open_spiel::eclipse::PassRotationTest();
}
