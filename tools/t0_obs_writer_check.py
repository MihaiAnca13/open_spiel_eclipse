#!/usr/bin/env python3
"""T0's other confirmation: do the 12 GB observation-writer wins carry to BIG?

docs/eclipse_rl_todo.md T0 asks for two checks beyond the throughput ladder:
  * observation_tensor_into ~= 5 us
  * observation_tensor costs ~24x observation_tensor_into -- any rl_environment
    built without observations_as_numpy=True pays that on every single step.

This is CPU-only, so it must NOT run concurrently with the throughput ladder:
it would steal cores from the env workers and corrupt the env phase it is meant
to corroborate.

States are sampled along a random playout rather than measured at the initial
state. The writer's cost tracks how much of the board is populated, so the
opening is the cheapest state in the game and timing only it would report a
number the training loop never sees.
"""
import statistics
import time

import numpy as np
import pyspiel

REPS = 400


def sample_states(game, depth, seed):
  """One state per 10 plies along a random playout, plus the opening."""
  rng = np.random.RandomState(seed)
  states = []
  s = game.new_initial_state()
  for ply in range(depth):
    if s.is_terminal():
      break
    if s.is_chance_node():
      outcomes, probs = zip(*s.chance_outcomes())
      s.apply_action(int(rng.choice(outcomes, p=np.asarray(probs))))
      continue
    if ply % 10 == 0:
      states.append(s.clone())
    legal = s.legal_actions()
    s.apply_action(int(legal[rng.randint(len(legal))]))
  return states


def bench(states, size):
  """(into_us, tensor_us) medians over REPS calls per state.

  The buffer MUST be float32. A float64 buffer used to be accepted silently:
  pybind's forcecast converted it to a float32 temporary, wrote the observation
  into the temporary and discarded it, so the call did nothing AND paid a 37,596
  element dtype conversion. Timing that path reported 15.2 us and produced a
  bogus "the writer regressed on this machine" conclusion. The binding now
  rejects non-float32 buffers outright.
  """
  buf = np.empty(size, dtype=np.float32)
  into, plain = [], []
  for s in states:
    t0 = time.perf_counter()
    for _ in range(REPS):
      s.observation_tensor_into(0, buf)
    into.append((time.perf_counter() - t0) / REPS)
    t0 = time.perf_counter()
    for _ in range(REPS):
      s.observation_tensor(0)
    plain.append((time.perf_counter() - t0) / REPS)
  return statistics.median(into) * 1e6, statistics.median(plain) * 1e6


def main():
  game = pyspiel.load_game("eclipse(players=4)")
  size = int(np.prod(game.observation_tensor_shape()))

  print("=== T0 observation writer check, BIG/behemoth (CPU) ===")
  print(f"obs size            : {size} floats")

  # Opening AND mid-game, separately. The writer's cost tracks how populated the
  # board is, so a single number cannot distinguish "this CPU is slower" from
  # "this measurement sampled later states than the one it is compared against".
  # The recorded ~5 us has no state attached to it, so both are reported and
  # the reader gets to decide which one the 5 us was.
  opening = [game.new_initial_state()]
  # The opening is a chance node in Eclipse; walk to the first decision node.
  s = opening[0]
  rng0 = np.random.RandomState(0)
  while s.is_chance_node():
    outcomes, probs = zip(*s.chance_outcomes())
    s.apply_action(int(rng0.choice(outcomes, p=np.asarray(probs))))
  opening = [s]

  mid = sample_states(game, depth=400, seed=1)
  for label, states in (("opening", opening), ("mid-game", mid)):
    into_us, plain_us = bench(states, size)
    print(f"\n  {label} ({len(states)} state(s))")
    print(f"    observation_tensor_into : {into_us:8.2f} us")
    print(f"    observation_tensor      : {plain_us:8.2f} us   "
          f"({plain_us / into_us:.1f}x into; the doc's ratio is ~24x)")
    if label == "mid-game":
      mid_into = into_us

  # What the number actually costs the training loop, which is the only thing
  # that matters here. One observation_tensor_into per env per step, spread over
  # num_workers processes -- so the per-call figure has to be divided by the
  # worker count before it can be compared against the measured env phase.
  print("\n  share of the measured env phase (T0 ladder, 256 envs/16 workers,"
        " env = 7.4 ms/step):")
  per_worker = (256 / 16) * mid_into / 1e3
  print(f"    16 envs/worker x {mid_into:.1f} us = {per_worker:.2f} ms/step "
        f"= {100 * per_worker / 7.4:.0f}% of the env phase")
  verdict = "CARRIES" if mid_into < 10.0 else "DOES NOT CARRY"
  print(f"\n  verdict: the 12 GB writer win {verdict} to BIG "
        f"(the plan expects ~5 us).")


if __name__ == "__main__":
  main()
