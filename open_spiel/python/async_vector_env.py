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
"""Process-parallel vectorized RL environment.

A drop-in (API-compatible) replacement for ``SyncVectorEnv`` that spreads the
game stepping across a pool of worker processes. Sequential stepping is purely
additive (~tens of microseconds per env), so wall time per batch scales ~1:1
with env count and decisions/s are flat no matter how many envs are thrown at a
single process. With one worker process per shard, each core steps its own envs
and decisions/s scale with worker count.

Two interfaces:

  * ``step`` / ``reset``: drop-in ``SyncVectorEnv``-compatible; builds per-env
    ``TimeStep`` objects. Deterministic (bit-for-bit) vs the synchronous path.
    The per-env Python churn this requires makes it a poor fit at very large
    env counts.
  * ``step_np`` / ``reset_np``: vectorized, array-native. Returns ``_StepArrays``
    (numpy obs / seats / legal-row-col indices / rewards / dones) with no
    per-env object construction. Pairs with the PPO array-native path and is
    what keeps decisions/s scaling with worker count.

Heavy data (observations, legal action ids, rewards, dones) flows through
``multiprocessing.shared_memory`` blocks; only two small semaphore tokens per
worker cross process boundaries per step. No per-env Python objects are
pickled per step.

Determinism: workers recreate the exact same envs (same game string and same
per-env ChanceEventSampler seeds), and the batch still advances in lock-step
every PPO step, so a trajectory produced through ``step`` matches the
synchronous setup bit-for-bit.
"""

import collections
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import os
import sys
import threading
import time
import traceback

import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.rl_environment import StepType
from open_spiel.python.rl_environment import TimeStep
from open_spiel.python.vector_env import SyncVectorEnv


class _StepArrays(
    collections.namedtuple("StepArrays", [
        "obs", "seats", "legal_rows", "legal_cols", "rewards", "dones"
    ])):
  """Array-native step payload for the vectorized interface.

  Attributes:
    obs: (num_envs, obs_size) float32 - acting seat's observation row per env.
    seats: (num_envs,) int32 - acting seat per env.
    legal_rows: (M,) int64 - env index for each legal action.
    legal_cols: (M,) int64 - legal action id for each.
    rewards: (num_envs, num_players) float32.
    dones: (num_envs,) bool.
  """


class AsyncVectorEnv(object):
  """Vector env backed by a pool of worker processes."""

  def __init__(self,
               envs,
               num_workers=1,
               sampler_seeds=None,
               max_legal=None,
               game_str=None,
               game_strs=None):
    """Constructor.

    Args:
      envs: list of ``rl_environment.Environment``, used to introspect the
        game shape on this side. The actual envs are recreated inside each
        worker from the same game strings + seeds, so passing a probe list of
        length ``num_envs`` with the same construction params is enough.
      num_workers: number of worker processes.
      sampler_seeds: list of per-env ChanceEventSampler seeds (same length as
        the real env list), used to rebuild deterministic samplers in the
        workers. If None, each worker uses seeds 0..count-1 (non-reproducible
        w.r.t. the synchronous setup).
      max_legal: max number of legal actions across all decision nodes; auto
        probed if None.
      game_str: the game string used to load the envs (e.g.
        ``"eclipse(players=4)"``) when all envs share one configuration.
        Workers rebuild their own fresh game from this string; important
        because the eclipse game object carries a mutable RNG that must not be
        shared across processes, and a parent instance may already have
        advanced it via probe resets.
      game_strs: optional per-env list of game strings (same length as
        ``envs``), used when each environment should be built from a different
        configuration (e.g. per-env ``rng_seed``). Takes precedence over
        ``game_str``.
    """
    if not isinstance(envs, list) or not envs:
      raise ValueError("Need a non-empty list of rl_environment.Environment")
    proto = envs[0]
    self.num_envs = len(envs)
    self.num_players = proto.num_players
    self.game = proto._game
    self.obs_size = int(np.prod(self.game.observation_tensor_shape()))
    self._is_turn_based = proto._is_turn_based
    self._game_str = game_str
    if self._game_str is None:
      gt = self.game.get_type()
      self._game_str = gt.short_name
      spec = gt.parameter_specification
      if spec:
        pieces = [f"{k}={v!s}".lower() if isinstance(v, bool) else k + "=" +
                  str(v) for k, v in spec.items()]
        self._game_str += "(" + ",".join(pieces) + ")"
    self._game_strs = game_strs
    if self._game_strs is None:
      self._game_strs = [self._game_str] * self.num_envs
    if len(self._game_strs) != self.num_envs:
      raise ValueError(
          f"game_strs must have length num_envs ({self.num_envs}), got "
          f"{len(self._game_strs)}")
    self._sampler_seeds = sampler_seeds
    # Width of the shared legal-action buffer. Defaults to the full action
    # space: any smaller value silently truncates legal sets at decision nodes
    # that happen to be wider than whatever was sampled to pick it, and the
    # agent then never sees the dropped actions (they are simply absent from
    # the mask, so nothing errors). The buffer is cheap -- num_envs *
    # num_distinct_actions * 4 bytes, e.g. 5.7 MB for eclipse at 128 envs --
    # so there is no reason to guess. Pass max_legal explicitly only if you
    # have a proof of the true bound for your game.
    self._max_legal = int(max_legal) if max_legal else int(
        self.game.num_distinct_actions())
    self.num_workers = num_workers
    if self._sampler_seeds is None:
      self._sampler_seeds = list(range(self.num_envs))

    self._ctx = mp.get_context("fork")
    self._shm_blocks = {}
    self._worker_procs = []
    self._go = []
    self._done = []
    self._start()

  def _alloc(self, name, shape, dtype):
    block = shm.SharedMemory(
        create=True, size=np.dtype(dtype).itemsize * int(np.prod(shape)))
    arr = np.ndarray(shape, dtype=dtype, buffer=block.buf)
    self._shm_blocks[name] = block
    return arr

  def _start(self):
    num_envs = self.num_envs
    per = -(-num_envs // self.num_workers)
    self.action_buf = self._alloc("actions", (num_envs,), np.int32)
    self.obs_buf = self._alloc("obs", (num_envs, self.obs_size), np.float32)
    self.legal_buf = self._alloc("legal", (num_envs, self._max_legal),
                                 np.int32)
    self.legal_len = self._alloc("legal_len", (num_envs,), np.int32)
    self.rew_buf = self._alloc("rewards", (num_envs, self.num_players),
                               np.float32)
    self.done_buf = self._alloc("done", (num_envs,), np.uint8)
    self.cur_buf = self._alloc("current", (num_envs,), np.int32)

    for w in range(self.num_workers):
      start = w * per
      count = min(per, num_envs - start)
      if count <= 0:
        self._worker_procs.append(None)
        continue
      go = mp.Semaphore(0)
      done = mp.Semaphore(0)
      self._go.append(go)
      self._done.append(done)
      seeds = self._sampler_seeds[start:start + count]
      games = self._game_strs[start:start + count]
      p = self._ctx.Process(
          target=_worker_main,
          args=(start, count, self.num_players, games,
                self._max_legal, self.obs_size, seeds, self.action_buf,
                self.obs_buf, self.legal_buf, self.legal_len, self.rew_buf,
                self.done_buf, self.cur_buf, go, done))
      p.start()
      self._worker_procs.append(p)

  def reset(self, envs_to_reset=None, players=None):
    # Workers publish their initial (pre-first-step) state on startup; this
    # call just collects it. Env-level re-resets happen inside workers via
    # reset_if_done during step(); there is no separate reset round-trip.
    for i in range(len(self._worker_procs)):
      if self._worker_procs[i] is not None:
        self._done[i].acquire()
    return self._build_steps()[0]

  def step(self, step_outputs, reset_if_done=False, players=None):
    num_envs = self.num_envs
    with self._obs_lock():
      for i, out in enumerate(step_outputs):
        self.action_buf[i] = int(out.action)
    for i in range(len(self._worker_procs)):
      if self._worker_procs[i] is not None:
        self._go[i].release()
    for i in range(len(self._worker_procs)):
      if self._worker_procs[i] is not None:
        self._done[i].acquire()
    return self._build_steps()

  def _run_step(self, actions):
    """Writes actions, steps workers, returns after one step round-trip."""
    with self._obs_lock():
      self.action_buf[:] = np.asarray(actions, dtype=np.int32)
    for i in range(len(self._worker_procs)):
      if self._worker_procs[i] is not None:
        self._go[i].release()
    for i in range(len(self._worker_procs)):
      if self._worker_procs[i] is not None:
        self._done[i].acquire()

  def _legal_indices(self):
    """Returns (legal_rows, legal_cols) int64 numpy from the packed buffers."""
    legal_len = self.legal_len
    lens = legal_len.astype(np.int64)
    total = int(lens.sum())
    num_envs = self.num_envs
    if total > 0:
      rows = np.repeat(np.arange(num_envs, dtype=np.int64), lens)
      cols = np.zeros(total, dtype=np.int64)
      pos = 0
      for i in range(num_envs):
        n = int(lens[i])
        if n:
          cols[pos:pos + n] = self.legal_buf[i, :n].astype(np.int64)
          pos += n
    else:
      rows = np.zeros(0, dtype=np.int64)
      cols = np.zeros(0, dtype=np.int64)
    return rows, cols

  def step_np(self, actions, reset_if_done=False):
    """Vectorized step: no per-env TimeStep/StepOutput objects.

    Args:
      actions: int32 numpy array of length num_envs with the chosen action per
        env.
      reset_if_done: ignored (workers always auto-reset on done; returns still
        carry the terminal step's reward/payoff).

    Returns:
      A namedtuple ``_StepArrays`` with fields: obs ((N, obs_size) float32
      copy), seats ((N,) int32 acting seats), legal_cols (packed int32 array of
      legal action ids) and legal_rows (int32 row indices, same length),
      rewards ((N, num_players) float32), dones ((N,) bool).
    """
    self._run_step(actions)
    return self._collect()

  def reset_np(self):
    """Vectorized reset: returns the current (initial) _StepArrays.

    Must be called after ``reset()`` so workers have published their initial
    state (the construction-time publish is consumed by the first reset).
    """
    return self._collect()

  def _collect(self):
    """Assembles _StepArrays from the current shared-memory buffers."""
    rows, cols = self._legal_indices()
    return _StepArrays(
        obs=self.obs_buf.copy(), seats=self.cur_buf.copy(),
        legal_rows=rows, legal_cols=cols,
        rewards=self.rew_buf.copy(), dones=self.done_buf.copy().astype(bool))

  def _obs_lock(self):
    return self.__dict__.setdefault("_buf_lock", threading.Lock())

  def _build_steps(self):
    """Assembles (time_steps, reward, done, unreset) from shared buffers."""
    num_envs = self.num_envs
    time_steps = []
    unreset = []
    reward = []
    done = []
    for i in range(num_envs):
      seat = int(self.cur_buf[i])
      obs_row = self.obs_buf[i].copy()
      n_legal = int(self.legal_len[i])
      legal = self.legal_buf[i, :n_legal].tolist()
      rew = self.rew_buf[i].tolist()
      is_terminal = bool(self.done_buf[i])
      observations = {
          "info_state": [None] * self.num_players,
          "legal_actions": [None] * self.num_players,
          "current_player": seat,
          "serialized_state": [],
      }
      observations["info_state"][seat] = obs_row
      observations["legal_actions"][seat] = legal
      # unreset: the transition that was just stepped (terminal payoff if last).
      unreset.append(
          TimeStep(
              observations=observations,
              rewards=rew,
              discounts=[0.0] * self.num_players if is_terminal else
              [1.0] * self.num_players,
              step_type=StepType.LAST if is_terminal else StepType.MID))
      # time_steps: fresh post-reset state used for the next decision.
      time_steps.append(
          TimeStep(
              observations=observations,
              rewards=None,  # FIRST
              discounts=None,
              step_type=StepType.FIRST))
      reward.append(rew)
      done.append(is_terminal)
    return time_steps, reward, done, unreset

  def __len__(self):
    return self.num_envs

  def close(self):
    for p in self._worker_procs:
      if p is not None:
        p.terminate()
    for b in self._shm_blocks.values():
      b.close()
      b.unlink()


def _worker_main(start, count, num_players, games, max_legal, obs_size,
                 seeds, action_buf, obs_buf, legal_buf, legal_len, rew_buf,
                 done_buf, cur_buf, go, done):
  """Worker process body: owns a shard of ``SyncVectorEnv``."""
  envs = []
  for i in range(count):
    env = rl_environment.Environment(
        game=games[i],
        chance_event_sampler=rl_environment.ChanceEventSampler(
            seed=int(seeds[i])),
        observation_type=rl_environment.ObservationType.OBSERVATION,
        observations_as_numpy=True)
    envs.append(env)
  vec = SyncVectorEnv(envs)

  def publish(ts_new, done_list, rew_list):
    for i in range(count):
      idx = start + i
      seat = int(ts_new[i].observations["current_player"])
      cur_buf[idx] = seat
      obs_buf[idx, :] = np.asarray(
          ts_new[i].observations["info_state"][seat], dtype=np.float32)
      la = ts_new[i].observations["legal_actions"][seat]
      n = len(la)
      # Never truncate: a dropped legal action is invisible to the agent and
      # raises no error anywhere downstream (it is simply absent from the
      # mask), so this must fail loudly rather than silently shrink the
      # action space.
      if n > max_legal:
        raise ValueError(
            f"legal action set of size {n} exceeds max_legal={max_legal} "
            f"(env {idx}, seat {seat}); truncating would silently hide "
            f"{n - max_legal} legal actions from the agent")
      legal_buf[idx, :n] = np.asarray(la, dtype=np.int32)
      legal_len[idx] = n
      rew_buf[idx, :] = np.asarray(rew_list[i][:num_players],
                                   dtype=np.float32)
      done_buf[idx] = 1 if done_list[i] else 0

  # Initial state: reset once, publish terminal=N/A row, tell the parent.
  _PROF = os.environ.get("ASYNC_PROF", "")
  ts = vec.reset(players="current")
  publish(ts, [False] * count, [[0.0] * num_players] * count)
  done.release()

  _ACC_WAIT = _ACC_STEP = _ACC_PUB = 0.0
  _N = 0
  while True:
    t0 = time.perf_counter()
    go.acquire()
    t1 = time.perf_counter()
    outs = []
    for i in range(count):
      outs.append(type("SO", (), {"action": int(action_buf[start + i])})())
    try:
      ts, reward, done_list, _ = vec.step(outs, reset_if_done=True,
                                          players="current")
      t2 = time.perf_counter()
      publish(ts, done_list, reward)
    except BaseException:
      # A worker exception otherwise dies unseen and the parent blocks forever
      # on done.acquire(); surface the cause before going down.
      traceback.print_exc()
      sys.stderr.flush()
      raise
    t3 = time.perf_counter()
    done.release()
    if _PROF:
      _ACC_WAIT += t1 - t0
      _ACC_STEP += t2 - t1
      _ACC_PUB += t3 - t2
      _N += 1
      if _N == 128:
        with open(f"/tmp/opencode/_wprof_{start}.txt", "w") as f:
          f.write(f"count={count} wait={_ACC_WAIT/128*1e3:.2f}ms "
                  f"step={_ACC_STEP/128*1e3:.2f}ms pub={_ACC_PUB/128*1e3:.2f}ms\n")
        break
