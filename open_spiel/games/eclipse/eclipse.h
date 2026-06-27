//
// Created by Mihai on 01/06/2026.
//

#ifndef OPEN_SPIEL_GAMES_ECLIPSE_ECLIPSE_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_ECLIPSE_H_

#include <random>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "systems/setup.h"

namespace open_spiel {
namespace eclipse {

class EclipseGame : public Game {
 public:
  explicit EclipseGame(const GameParameters& params);
  
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  std::unique_ptr<State> DeserializeState(const std::string& str) const override;
  int MaxChanceOutcomes() const override { return 22; }
  int NumPlayers() const override;
  double MinUtility() const override { return 0.0; }
  double MaxUtility() const override { return 255.0; }
  absl::optional<double> UtilitySum() const override { return absl::nullopt; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  std::string GetRNGState() const override;
  void SetRNGState(const std::string& rng_state) const override;

  // Parameter getters
  int GetPlayersParam() const { return ParameterValue<int>("players"); }
  uint64_t GetRngSeedParam() const {
    return static_cast<uint64_t>(ParameterValue<int>("rng_seed"));
  }
  bool GetWarpedUniverseParam() const {
    return ParameterValue<bool>("warped_universe");
  }
  SetupConfig InitialSetupConfig() const;

  std::mt19937_64& rng() const { return rng_; }

 private:
  mutable std::mt19937_64 rng_;
};

class EclipseState : public State {
 public:
  enum class PendingRandomEvent : uint8_t {
    none = 0,
    initial_setup = 1,
    explore_draw = 2,
    discovery_draw = 3,
    combat_roll = 4,
    reputation_draw = 5,
  };

  explicit EclipseState(std::shared_ptr<const Game> game);
  EclipseState(const EclipseState&) = default;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;

  ActionsAndProbs ChanceOutcomes() const override;
  std::string Serialize() const override;

  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player, absl::Span<float> values) const override;

  const ::State& RawState() const { return eclipse_state_; }
  PendingRandomEvent pending_random_event() const { return pending_random_event_; }
  void RestoreFromSnapshot(const SetupConfig& config,
                           const ::State& state,
                           PendingRandomEvent pending_random_event);

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  std::shared_ptr<const EclipseGame> eclipse_game_;
  void ResolveChanceEvent(Action action_id);
  std::vector<Action> ExploreLegalActions() const;
  std::vector<Action> ResearchLegalActions() const;
  std::vector<Action> InfluenceLegalActions() const;
  std::vector<Action> BuildLegalActions() const;
  std::vector<Action> UpgradeLegalActions() const;
  std::vector<Action> MoveLegalActions() const;
  std::vector<Action> DiplomacyLegalActions() const;
  std::vector<Action> UpkeepLegalActions() const;
  std::vector<Action> CombatLegalActions() const;
  void ApplyExploreSubAction(Action action_id);
  void ApplyResearchSubAction(Action action_id);
  void ApplyInfluenceSubAction(Action action_id);
  void ApplyBuildSubAction(Action action_id);
  void ApplyUpgradeSubAction(Action action_id);
  void ApplyMoveSubAction(Action action_id);
  void ApplyDiplomacySubAction(Action action_id);
  void ApplyUpkeepAction(Action action_id);
  void ApplyCombatSubAction(Action action_id);
  void AdvanceTurn();
  void BeginUpkeep();
  void BeginCombat();
  // Drive the combat state machine forward until it needs a chance roll
  // (arming pending_random_event_), a player decision, or completes. Called
  // after BeginCombat, after each combat sub-action, and after each combat
  // chance outcome is resolved.
  void DriveCombat();
  void FinishCombat();
  void BeginCleanup();
  void AdvanceUpkeepState();
  void AdvancePastCurrentUpkeepPlayer();
  void AdvanceCleanupState();
  void FinishCleanup();

  ::State eclipse_state_;
  SetupConfig setup_config_;
  PendingRandomEvent pending_random_event_ = PendingRandomEvent::initial_setup;
};

} // namespace eclipse
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_ECLIPSE_ECLIPSE_H_
