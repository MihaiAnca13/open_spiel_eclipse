#ifndef OPEN_SPIEL_GAMES_ECLIPSE_WARPED_UNIVERSE_WARPED_UNIVERSE_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_WARPED_UNIVERSE_WARPED_UNIVERSE_H_

struct State;

namespace open_spiel::eclipse {

// Rebuilds warp link data structures based on the warped_universe flag and player count.
// This is called during game setup and state deserialization to reconstruct the warp portal
// connections when warped_universe is enabled.
void RebuildWarpLinks(::State& state, int players_count_override = 0);

}  // namespace open_spiel::eclipse

#endif  // OPEN_SPIEL_GAMES_ECLIPSE_WARPED_UNIVERSE_WARPED_UNIVERSE_H_
