#ifndef OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_UPKEEP_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_UPKEEP_H_

#include <cstdint>

#include "open_spiel/games/eclipse/state.h"

namespace open_spiel::eclipse {

int CleanupRegularTechDrawCount(int player_count);
void DrawCleanupTechTiles(::State& state, int player_count);

int PlayerIncome(const ::Player& player);
int PlayerScienceProduction(const ::Player& player);
int PlayerMaterialsProduction(const ::Player& player);
int PlayerUpkeepCost(const ::Player& player);
bool IsPlayerSolvent(const ::Player& player);

uint8_t FirstActivePlayerInTurnOrder(const ::State& state, int num_players);
uint8_t NextActivePlayerInTurnOrder(const ::State& state, uint8_t current_player,
                                    int num_players);

}  // namespace open_spiel::eclipse

#endif  // OPEN_SPIEL_GAMES_ECLIPSE_SYSTEMS_UPKEEP_H_
