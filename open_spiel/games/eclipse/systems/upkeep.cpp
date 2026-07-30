#include "open_spiel/games/eclipse/systems/upkeep.h"

#include <algorithm>

#include "open_spiel/games/eclipse/species.h"
#include "open_spiel/games/eclipse/tech.h"

namespace open_spiel::eclipse {

int CleanupRegularTechDrawCount(int player_count) {
  if (player_count <= 2) return 5;
  if (player_count == 3) return 6;
  if (player_count == 4) return 7;
  if (player_count == 5) return 8;
  return 9;
}

void DrawCleanupTechTiles(::State& state, int player_count) {
  int regular_drawn = 0;
  const int regular_target = CleanupRegularTechDrawCount(player_count);
  while (regular_drawn < regular_target && !state.tech_bag.empty()) {
    const TechBit drawn = state.tech_bag.back();
    state.tech_bag.pop_back();
    state.add_to_tech_tray(drawn);
    if (get_tech_category_from_bit(drawn) != TechCategory::RARE) {
      ++regular_drawn;
    }
  }
}

int PlayerIncome(const ::Player& player) {
  return POPULATION_PRODUCTION_TABLE[player.resources.gold_prod];
}

int PlayerScienceProduction(const ::Player& player) {
  return POPULATION_PRODUCTION_TABLE[player.resources.science_prod];
}

int PlayerMaterialsProduction(const ::Player& player) {
  return POPULATION_PRODUCTION_TABLE[player.resources.materials_prod];
}

int PlayerUpkeepCost(const ::Player& player) {
  const int penalty =
      SPECIES_TABLE[static_cast<size_t>(player.species_id)].starting_disk_penalty;
  const int revealed =
      std::max(0, static_cast<int>(player.disks_on_actions) +
                      static_cast<int>(player.disks_on_reactions) +
                      static_cast<int>(player.disks_on_sectors) + penalty -
                      static_cast<int>(player.extra_influence_discs));
  return INFLUENCE_UPKEEP_TABLE[std::min(revealed, total_influence_discs)];
}

bool IsPlayerSolvent(const ::Player& player) {
  return static_cast<int>(player.resources.gold) + PlayerIncome(player) -
             PlayerUpkeepCost(player) >=
         0;
}

uint8_t FirstActivePlayerInTurnOrder(const ::State& state, int num_players) {
  for (int i = 0; i < num_players; ++i) {
    const uint8_t player_id = state.turn_order[i];
    if (player_id < state.players.size() && !state.players[player_id].eliminated) {
      return player_id;
    }
  }
  return 255;
}

uint8_t NextActivePlayerInTurnOrder(const ::State& state, uint8_t current_player,
                                    int num_players) {
  for (int i = 0; i < num_players; ++i) {
    if (state.turn_order[i] != current_player) continue;
    for (int step = i + 1; step < num_players; ++step) {
      const uint8_t next_player = state.turn_order[step];
      if (next_player < state.players.size() &&
          !state.players[next_player].eliminated) {
        return next_player;
      }
    }
    break;
  }
  return 255;
}

}  // namespace open_spiel::eclipse
