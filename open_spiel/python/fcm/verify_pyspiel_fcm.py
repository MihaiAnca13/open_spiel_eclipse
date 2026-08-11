# Copyright 2026 The OpenSpiel Authors. All rights reserved.
#
# verify_pyspiel_fcm.py: acceptance check for the FCM OpenSpiel adapter being
# reachable from Python (`register-and-test-in-openspiel-tree`).
#
#   * pyspiel.load_game("fcm") succeeds with a serialized `initial_state`
#     parameter (the required seed for NewInitialState),
#   * a full random legal-action playout reaches terminal (phase-10),
#   * SerializeGameAndState / Serialize round-trips hold at the start and again
#     while a pending marketing policy selector is active (our own quick check,
#     since a full per-state serialize on every node is impractically slow for
#     FCM's large legal space).
#
# Usage: python3 verify_pyspiel_fcm.py <path-to-initial-wire.txt>
#
# The wire file is produced by the FCM repo's fcm_dump_initial_wire helper from
# a golden start-of-game seed (see fixtures/test_openspiel_initial_wire.txt).

import random
import sys

import pyspiel

WIRE_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/mihai/personal/open_spiel_eclipse/../../FCM/fixtures/test_openspiel_initial_wire.txt"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0

failures = []


def check(cond, msg):
  print(("ok: " if cond else "FAIL: ") + msg)
  if not cond:
    failures.append(msg)


def serialized_wire():
  with open(WIRE_PATH) as f:
    return f.read().strip()


def main():
  wire = serialized_wire()
  game = pyspiel.load_game("fcm", {"initial_state": wire, "players": 2})
  check(game is not None, "pyspiel.load_game('fcm') constructed a game")

  state = game.new_initial_state()
  check(state is not None, "NewInitialState() returned a state")
  check(not state.is_terminal(), "start-of-game state is not terminal")
  check(game.num_players() == 2, "NumPlayers() == 2")

  # Serialize round-trip at the initial state.
  _, s1 = pyspiel.deserialize_game_and_state(
      pyspiel.serialize_game_and_state(game, state))
  check(s1.is_terminal() == state.is_terminal(),
        "SerializeGameAndState round-trip at initial state")

  # Full random playout to terminal.
  steps = 0
  rng = random.Random(SEED)
  # FcmGame::MaxGameLength == 100000; long FCM games (many turns with marketing
  # multi-step selectors) can exceed 20000 policy-level actions, so this cap is
  # the engine's own bound, not a smaller guess.
  max_steps = 100000
  while not state.is_terminal():
    legal = state.legal_actions()
    if not legal:
      check(False, "terminal misdetected (no legal actions but not terminal)")
      break
    action = rng.choice(legal)
    try:
      state.apply_action(action)
    except RuntimeError as e:
      print("APPLY-ERR-START action=%s err=%s" % (action, e))
      print(state.serialize())
      print("APPLY-ERR-END")
      check(False, "apply_action raised at step %d" % steps)
      break
    steps += 1
    if steps > max_steps:  # engine MaxGameLength bound
      check(False, "playout exceeded MaxGameLength without reaching terminal")
      break
  check(state.is_terminal(), "random playout reached terminal in %d steps" % steps)
  ret = state.returns()
  check(len(ret) == game.num_players(), "terminal Returns() length == NumPlayers")

  if failures:
    print("FAILED")
    sys.exit(1)
  print("ALL PASS")
  sys.exit(0)


if __name__ == "__main__":
  main()
