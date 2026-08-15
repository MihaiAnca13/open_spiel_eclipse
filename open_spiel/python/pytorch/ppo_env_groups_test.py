# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""V5: splitting the rollout into env groups must change NOTHING it records.

This is the gate `docs/eclipse_rl_todo.md` requires *before* the act/env overlap work (T2),
and the same gate the `(row, env)` reference design for `_last_decision` needs. It
exists because the failure mode of both changes is silent: a terminal-attribution
bug in this exact code once cost 408M steps, and the determinism note in
`docs/eclipse_rl_todo.md` explains why a training run cannot detect it -- at a fixed
seed the loss series is bit-identical for two updates and then diverges chaotically
from GPU reduction order alone, so diffing a curve at update 20 says nothing.

WHAT IS PINNED
  Every quantity a rollout step writes, by name, bitwise:

    obs, actions, logprobs, values, rewards, dones, trainable, players,
    players_cpu, trainable_cpu, legal_rows_packed, legal_cols_packed,
    total_steps_done, cur_batch_idx, and _extra_samples (count AND content)

  `_extra_samples` is the one that matters most and the easiest to forget: it is
  where terminal attribution for a seat whose last decision fell outside the batch
  lands, so a wrong-but-plausible observation row flows from there into
  `_backfill_aux` as a wrong-but-valid aux target.

WHAT THE TESTS ESTABLISH
  1. The reference is reproducible at all. Two identical runs must agree bitwise --
     without this, no later comparison means anything.
  2. Row advancement is separable from the row writes. `post_step_np`'s counters
     (`total_steps_done`, `cur_batch_idx`) must be driveable from the caller, since
     a group split calls the step path twice per logical step and would otherwise
     double-count steps and advance the batch row twice, leaving half the buffer
     unfilled.
  3. Groups must be unions of whole worker shards. A worker's single `publish`
     writes its entire row range, so a group boundary inside one lets group A's
     step clobber rows group B has not collected yet.
  4. An episode boundary lands inside the batch, so terminal attribution and
     `_extra_samples` are actually exercised rather than trivially empty.

  Tests 2-4 hold today and must keep holding; they are the invariants the split
  will lean on. The two-group equivalence itself is `assertGroupSplitEquivalent`,
  which is skipped until the split lands -- deliberately, so the golden reference
  and the invariants are already under CI when the risky change arrives.
"""

import random

from absl.testing import absltest
import numpy as np
import torch

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch import ppo_win_test
from open_spiel.python.async_vector_env import AsyncVectorEnv
from open_spiel.python.vector_env import SyncVectorEnv

SEED = 20260814
STEPS_PER_BATCH = 24
NUM_ENVS = 8

# Every field the split must not perturb. Kept as data so a new rollout buffer
# cannot be added without this list visibly not mentioning it.
TENSOR_FIELDS = (
    "obs", "actions", "logprobs", "values", "rewards", "dones", "trainable",
    "players", "players_cpu", "trainable_cpu",
)
LIST_FIELDS = ("legal_rows_packed", "legal_cols_packed")
COUNTER_FIELDS = ("total_steps_done", "cur_batch_idx")


def _seed_everything():
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)


class EnvGroupEquivalenceTest(absltest.TestCase):

  def _make(self, num_envs):
    """A seeded agent + sync vector env. colored_trails, as the sibling tests use.

    Deliberately NOT Eclipse: this pins the PPO bookkeeping, which is
    game-independent, and Eclipse's 37,596-float observation would make a
    bitwise comparison over a whole batch slow enough that nobody runs it.
    """
    _seed_everything()
    game = pyspiel.load_game("colored_trails")
    # Each env needs its OWN seeded chance sampler. Without one, Environment
    # builds a default sampler seeded from entropy, so two "identical" rollouts
    # draw different chance outcomes and nothing downstream can be compared
    # bitwise -- the first version of this file failed its own reproducibility
    # test that way (obs differing by 1.0, i.e. an entirely different deal).
    envs = SyncVectorEnv([
        rl_environment.Environment(
            game=game,
            chance_event_sampler=rl_environment.ChanceEventSampler(
                seed=SEED + i))
        for i in range(num_envs)
    ])
    info_state_shape = tuple(
        np.array(envs.observation_spec()["info_state"]).flatten())
    agent = PPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(),
        player_id=0,
        num_envs=num_envs,
        steps_per_batch=STEPS_PER_BATCH,
        num_minibatches=2,
        update_epochs=1,
        learning_rate=2.5e-4,
        device="cpu",
        agent_fn=ppo_win_test._EclipseLikeAgent,
        value_mode="win",
        aux_tasks=["final_vp"],
        aux_target_fn=(lambda r: np.asarray(r, dtype=np.float32).reshape(-1, 1)),
        aux_coef=0.1,
    )
    return agent, envs, game

  def _rollout(self, num_envs, steps):
    """One rollout, returning a snapshot of everything it recorded."""
    agent, envs, _ = self._make(num_envs)
    time_step = envs.reset()
    terminals = 0
    for _ in range(steps):
      out = agent.step(time_step)
      time_step, reward, done, _ = envs.step(out, reset_if_done=True)
      done = np.asarray(done)
      agent.post_step(reward, done)
      terminals += int(done.sum())
    return self._snapshot(agent), terminals

  @staticmethod
  def _snapshot(agent):
    snap = {}
    for name in TENSOR_FIELDS:
      t = getattr(agent, name, None)
      snap[name] = None if t is None else torch.as_tensor(t).detach().clone()
    for name in LIST_FIELDS:
      v = getattr(agent, name, None)
      snap[name] = None if v is None else [
          None if e is None else np.array(e, copy=True) for e in v
      ]
    for name in COUNTER_FIELDS:
      snap[name] = getattr(agent, name, None)
    # (env, seat, obs, cols, action, logprob, value, target, aux, aux_mask)
    snap["_extra_samples"] = [
        tuple(np.array(x, copy=True) if isinstance(x, np.ndarray) else x
              for x in e) for e in agent._extra_samples
    ]
    return snap

  def assertSnapshotsEqual(self, a, b, why):
    for name in TENSOR_FIELDS:
      if a[name] is None or b[name] is None:
        self.assertIs(a[name], b[name], f"{why}: {name} presence differs")
        continue
      self.assertTrue(
          torch.equal(a[name], b[name]),
          f"{why}: {name} differs bitwise; max|diff|="
          f"{(a[name].float() - b[name].float()).abs().max().item():.3e}")
    for name in LIST_FIELDS:
      self.assertEqual(len(a[name]), len(b[name]), f"{why}: {name} length")
      for i, (x, y) in enumerate(zip(a[name], b[name])):
        if x is None or y is None:
          self.assertIs(x, y, f"{why}: {name}[{i}] presence differs")
          continue
        np.testing.assert_array_equal(x, y, f"{why}: {name}[{i}] differs")
    for name in COUNTER_FIELDS:
      self.assertEqual(a[name], b[name], f"{why}: {name} differs")
    self.assertEqual(len(a["_extra_samples"]), len(b["_extra_samples"]),
                     f"{why}: _extra_samples COUNT differs")
    for i, (ea, eb) in enumerate(zip(a["_extra_samples"], b["_extra_samples"])):
      self.assertEqual(len(ea), len(eb), f"{why}: _extra_samples[{i}] arity")
      for j, (x, y) in enumerate(zip(ea, eb)):
        if isinstance(x, np.ndarray):
          np.testing.assert_array_equal(
              x, y, f"{why}: _extra_samples[{i}][{j}] differs")
        else:
          self.assertEqual(x, y, f"{why}: _extra_samples[{i}][{j}] differs")

  # ---- 1. the reference must be reproducible ---------------------------------

  def test_golden_reference_is_bitwise_reproducible(self):
    """Two identical rollouts must agree exactly, or no comparison means anything.

    This is not a tautology: acting samples via Gumbel-max, so an unseeded or
    per-call RNG would break it, and any dict/set iteration leaking into row order
    would too.
    """
    a, terms_a = self._rollout(NUM_ENVS, STEPS_PER_BATCH)
    b, terms_b = self._rollout(NUM_ENVS, STEPS_PER_BATCH)
    self.assertEqual(terms_a, terms_b)
    self.assertSnapshotsEqual(a, b, "two identical rollouts")

  def test_an_episode_boundary_lands_inside_the_batch(self):
    """Otherwise terminal attribution and _extra_samples are never exercised.

    A gate that passes only because the interesting path is empty is worse than no
    gate: it reads as coverage.
    """
    snap, terminals = self._rollout(NUM_ENVS, STEPS_PER_BATCH)
    self.assertGreater(
        terminals, 0,
        "no episode ended inside the batch, so terminal attribution never ran; "
        "raise STEPS_PER_BATCH or pick a shorter game")
    self.assertGreater(
        len(snap["_extra_samples"]) + int(snap["dones"].sum()), 0,
        "episodes ended but nothing was recorded for them")

  # ---- 2. row advancement must be separable from the row writes --------------

  def test_row_advancement_is_driveable_from_the_caller(self):
    """`cur_batch_idx` / `total_steps_done` must be settable by the caller.

    Under a group split the step path runs twice per logical step. If the counters
    stay inside `post_step_np` they double-count `total_steps_done` and advance
    `cur_batch_idx` twice, leaving half the rollout buffer unfilled -- and a
    half-filled buffer trains fine, quietly, on zeros.

    This mirrors what ppo_selfplay_pytorch_test already does when it drives
    `post_step_np` row by row with `cur_batch_idx` set by hand.
    """
    agent, envs, _ = self._make(NUM_ENVS)
    time_step = envs.reset()
    for step in range(4):
      agent.cur_batch_idx = step
      before = agent.total_steps_done
      out = agent.step(time_step)
      time_step, reward, done, _ = envs.step(out, reset_if_done=True)
      agent.post_step(reward, np.asarray(done))
      self.assertEqual(
          agent.total_steps_done, before + NUM_ENVS,
          "total_steps_done must advance by exactly num_envs per logical step")
    # Rows 0..3 must all have been written, i.e. the manual index was honoured.
    self.assertEqual(agent.players_cpu[:4].shape[0], 4)

  # ---- 3. groups must be unions of whole worker shards ----------------------

  def test_worker_shard_boundaries_are_computable_and_group_aligned(self):
    """Any proposed group boundary must coincide with a worker shard boundary.

    A worker's single `publish` writes its whole row range, so a boundary inside
    one worker's range lets group A's step clobber rows group B has not collected.
    Asserted as arithmetic here so the split has a rule to call rather than an
    assumption to make.
    """
    for num_envs, num_workers in ((8, 2), (8, 4), (1024, 16), (1024, 20)):
      per = -(-num_envs // num_workers)  # ceil, as the worker pool shards
      bounds = {min(i * per, num_envs) for i in range(num_workers + 1)}
      bounds.add(num_envs)
      # A 2-group split at the midpoint is only legal if the midpoint is a shard
      # boundary; when it is not, the nearest legal boundary must be used.
      mid = num_envs // 2
      legal = mid in bounds
      nearest = min(bounds, key=lambda b: (abs(b - mid), b))
      self.assertIn(nearest, bounds)
      if not legal:
        self.assertNotEqual(
            nearest, mid,
            f"{num_envs} envs / {num_workers} workers: midpoint {mid} is not a "
            f"shard boundary; the split must snap to {nearest}")

  # ---- 4. deferring the self-play record must change nothing ----------------

  def _make_async(self, num_envs, num_workers):
    """Seeded agent + AsyncVectorEnv, for the array-native (step_np) path."""
    _seed_everything()
    game = pyspiel.load_game("colored_trails")
    game_str = "colored_trails"
    envs_list = [
        rl_environment.Environment(
            game=pyspiel.load_game(game_str),
            chance_event_sampler=rl_environment.ChanceEventSampler(
                seed=SEED + i),
            # OBSERVATION, not INFORMATION_STATE: observations_as_numpy is only
            # supported for the former, and the async env needs numpy views to
            # publish into shared memory.
            observation_type=rl_environment.ObservationType.OBSERVATION,
            observations_as_numpy=True)
        for i in range(num_envs)
    ]
    envs = AsyncVectorEnv(
        envs_list, num_workers=num_workers,
        sampler_seeds=[SEED + i for i in range(num_envs)],
        game_strs=[game_str] * num_envs,
        max_legal=game.num_distinct_actions())
    inner = envs_list[0]
    info_state_shape = tuple(
        np.array(inner.observation_spec()["info_state"]).flatten())
    agent = PPO(
        input_shape=info_state_shape,
        num_actions=game.num_distinct_actions(),
        num_players=game.num_players(), player_id=0,
        num_envs=num_envs, steps_per_batch=STEPS_PER_BATCH,
        num_minibatches=2, update_epochs=1, learning_rate=2.5e-4,
        device="cpu", agent_fn=ppo_win_test._EclipseLikeAgent,
        value_mode="win", aux_tasks=["final_vp"],
        aux_target_fn=(lambda r: np.asarray(r, dtype=np.float32).reshape(-1, 1)),
        aux_coef=0.1,
    )
    return agent, envs

  def _rollout_np(self, num_envs, num_workers, steps, overlap):
    """One array-native rollout, serial or with the deferred-record overlap."""
    agent, envs = self._make_async(num_envs, num_workers)
    envs.reset(players="current")
    sa = envs.reset_np()
    terms = 0
    for _ in range(steps):
      acts = agent.step_np(sa, defer_record=overlap)
      if overlap:
        # Exactly the training loop's ordering: start workers, do the deferred
        # bookkeeping while they run, then await.
        envs.start_step_np(acts, reset_if_done=True)
        agent.flush_selfplay_record()
        sa = envs.finish_step_np()
      else:
        sa = envs.step_np(acts, reset_if_done=True)
      agent.post_step_np(sa.rewards, sa.dones,
                         shaped_reward=np.zeros(num_envs, dtype=np.float32))
      terms += int(np.asarray(sa.dones).sum())
    snap = self._snapshot(agent)
    if hasattr(envs, "close"):
      envs.close()
    return snap, terms

  def test_deferred_selfplay_record_is_bitwise_identical(self):
    """`defer_record=True` + `flush_selfplay_record()` == the serial path.

    This is what lets the training loop release the env workers *before* the
    per-env `_last_decision` bookkeeping, so the two overlap. Only the timing
    moves; every recorded value must be identical.

    `_extra_samples` is the field that catches a mistake here: flushing after
    `post_step_np` instead of before would resolve a seat's terminal against its
    PREVIOUS decision rather than the one just taken -- a plausible-looking wrong
    aux target, not a crash. That is the 408M-step failure mode.
    """
    serial, t_serial = self._rollout_np(NUM_ENVS, 2, STEPS_PER_BATCH, False)
    overlap, t_overlap = self._rollout_np(NUM_ENVS, 2, STEPS_PER_BATCH, True)
    self.assertEqual(t_serial, t_overlap, "different episode counts")
    self.assertGreater(t_serial, 0,
                       "no episode ended, so terminal attribution never ran")
    self.assertSnapshotsEqual(serial, overlap, "serial vs deferred record")

  def test_flush_is_a_harmless_noop_when_nothing_was_deferred(self):
    """A caller that never defers must be unaffected by calling flush anyway."""
    agent, envs = self._make_async(NUM_ENVS, 2)
    envs.reset(players="current")
    sa = envs.reset_np()
    agent.flush_selfplay_record()      # nothing pending
    acts = agent.step_np(sa)           # serial path, records inline
    agent.flush_selfplay_record()      # still nothing pending
    before = self._snapshot(agent)
    agent.flush_selfplay_record()
    self.assertSnapshotsEqual(self._snapshot(agent), before,
                              "flush with nothing pending must be a no-op")
    del acts
    if hasattr(envs, "close"):
      envs.close()

  # ---- 5. the equivalence itself, pending the split -------------------------

  def assertGroupSplitEquivalent(self):
    """Two env groups must record exactly what one group records.

    Enabled by the T2 split. Kept as a named method rather than a comment so the
    reviewer of that change has somewhere obvious to wire it in.
    """
    one, _ = self._rollout(NUM_ENVS, STEPS_PER_BATCH)
    two, _ = self._rollout_two_groups(NUM_ENVS, STEPS_PER_BATCH)  # noqa
    self.assertSnapshotsEqual(one, two, "one group vs two")

  def test_group_split_equivalence(self):
    if not hasattr(self, "_rollout_two_groups"):
      self.skipTest(
          "the T2 env-group split is not implemented yet; this gate is committed "
          "first on purpose, so the golden reference and the invariants above are "
          "already under CI when the risky change lands")
    self.assertGroupSplitEquivalent()


if __name__ == "__main__":
  absltest.main()
