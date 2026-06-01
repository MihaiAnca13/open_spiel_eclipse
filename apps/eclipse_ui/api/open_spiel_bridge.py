from __future__ import annotations

from typing import Any

import pyspiel


def load_eclipse_game(players: int = 4, seed: int = 7) -> pyspiel.Game:
    return pyspiel.load_game("eclipse", {"players": players, "seed": seed})


def state_to_lightzero_dict(state: pyspiel.State, player: int | None = None) -> dict[str, Any]:
    current_player = state.current_player()
    observer_player = current_player if player is None else player
    legal_actions = list(state.legal_actions())
    num_actions = state.get_game().num_distinct_actions()
    action_mask = [0] * num_actions
    for action in legal_actions:
        action_mask[action] = 1

    return {
        "current_player": current_player,
        "player": observer_player,
        "legal_actions": legal_actions,
        "action_mask": action_mask,
        "observation": list(state.observation_tensor(observer_player)),
    }
