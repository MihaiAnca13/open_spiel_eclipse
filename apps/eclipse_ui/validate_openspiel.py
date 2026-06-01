from __future__ import annotations

import pyspiel


def main() -> None:
    game_name = "eclipse"
    names = pyspiel.registered_names()
    assert game_name in names, f"{game_name!r} not found in registered_names()"

    game = pyspiel.load_game(game_name, {"players": 4, "rng_seed": 7})
    state = game.new_initial_state()

    assert not state.is_terminal()
    assert state.is_chance_node()
    assert game.num_distinct_actions() == 32
    assert state.legal_actions()

    for _ in range(8):
        if state.is_terminal():
            break
        legal_actions = state.legal_actions()
        action = legal_actions[1] if len(legal_actions) > 1 else legal_actions[0]
        state.apply_action(action)

    print("Validated OpenSpiel registration and rollout for eclipse.")
    print(state)


if __name__ == "__main__":
    main()
