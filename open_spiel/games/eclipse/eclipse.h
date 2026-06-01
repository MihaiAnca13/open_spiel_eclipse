//
// Created by Mihai on 01/06/2026.
//

#ifndef OPEN_SPIEL_GAMES_ECLIPSE_ECLIPSE_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_ECLIPSE_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "state.h"

namespace open_spiel {
namespace eclipse {

class EclipseGame : public Game {
 public:
  explicit EclipseGame(const GameParameters& params);
  
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override;
  double MinUtility() const override { return 0.0; }
  double MaxUtility() const override { return 255.0; }
  absl::optional<double> UtilitySum() const override { return absl::nullopt; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;

  // Parameter getters
  int GetPlayersParam() const { return ParameterValue<int>("players"); }
  int GetSeedParam() const { return ParameterValue<int>("seed"); }
};

class EclipseState : public State {
 public:
  explicit EclipseState(std::shared_ptr<const Game> game);
  EclipseState(const EclipseState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<EclipseState>(*this);
  }

  ActionsAndProbs ChanceOutcomes() const override;

  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player, absl::Span<float> values) const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  ::State eclipse_state_;
  bool initialized_ = false;
};

} // namespace eclipse
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_ECLIPSE_ECLIPSE_H_
