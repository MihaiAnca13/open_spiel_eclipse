# Copyright 2022 DeepMind Technologies Limited
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
"""Policy roster + matchmaking for league (population) self-play.

The roster stores neural-policy checkpoints on disk (``state_dict`` + a JSON
metadata index) so a growing pool of snapshots and exploiters can serve as
training opponents for the main policy, and the format automatically through
``ppo.rank_utility`` regardless of the identity of the non-main policies.

Roles:
  main      : the policy being trained (owns gradients, always stored as
              ``main.pt``).
  snapshot  : historical copies of the main policy captured on a cadence.
  exploiter : side-trained clones optimized to beat a frozen victim snapshot.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch


class RosterEntry(object):
  """One policy in the roster."""

  def __init__(self, policy_id, role, birth_update, path=None, win_rate=None):
    self.policy_id = policy_id
    self.role = role
    self.birth_update = birth_update
    self.path = path
    self.win_rate = win_rate

  def to_meta(self):
    return {
        "policy_id": str(self.policy_id),
        "role": self.role,
        "birth_update": int(self.birth_update),
        "path": self.path,
        "win_rate": None if self.win_rate is None else float(self.win_rate),
    }

  @staticmethod
  def from_meta(meta):
    return RosterEntry(
        policy_id=meta["policy_id"],
        role=meta["role"],
        birth_update=meta["birth_update"],
        path=meta.get("path"),
        win_rate=meta.get("win_rate"),
    )


class PolicyRoster(object):
  """Bounded, on-disk pool of policy checkpoints."""

  MAIN_ID = "main"

  def __init__(self, save_dir):
    self.save_dir = Path(save_dir)
    self.save_dir.mkdir(parents=True, exist_ok=True)
    self.entries = {}  # policy_id -> RosterEntry (insertion == birth order)
    self._load_index()

  def _index_path(self):
    return self.save_dir / "roster.json"

  def _load_index(self):
    path = self._index_path()
    if path.exists():
      with open(path, "r") as f:
        metas = json.load(f)
      self.entries = {
          meta["policy_id"]: RosterEntry.from_meta(meta) for meta in metas
      }
    # Ensure main exists (even if empty of weights yet).
    if self.MAIN_ID not in self.entries:
      self.entries[self.MAIN_ID] = RosterEntry(
          policy_id=self.MAIN_ID, role="main", birth_update=0,
          path=str(self.save_dir / f"{self.MAIN_ID}.pt"))

  def _save_index(self):
    metas = [e.to_meta() for e in self.entries.values()]
    with open(self._index_path(), "w") as f:
      json.dump(metas, f, indent=2)

  def reset_weights_dir(self):
    """Removes stale weight files (keeps the index, recreated on demand)."""
    for e in self.entries.values():
      p = Path(e.path)
      if p.exists():
        p.unlink()

  def __len__(self):
    return len(self.entries)

  @property
  def main_id(self):
    return self.MAIN_ID

  def get(self, policy_id):
    return self.entries.get(str(policy_id))

  def save_weights(self, policy_id, net):
    entry = self.entries.setdefault(
        str(policy_id),
        RosterEntry(str(policy_id), "snapshot", 0,
                    path=str(self.save_dir / f"{policy_id}.pt")))
    entry.path = str(self.save_dir / f"{policy_id}.pt")
    torch.save(net.state_dict(), entry.path)
    self._save_index()
    return entry

  def record_main(self, net, update):
    entry = self.entries[self.MAIN_ID]
    entry.birth_update = int(update)
    return self.save_weights(self.MAIN_ID, net)

  def add_snapshot(self, net, update):
    policy_id = f"snap_u{int(update)}"
    if policy_id in self.entries:
      return self.entries[policy_id]
    entry = RosterEntry(policy_id, "snapshot", int(update),
                        path=str(self.save_dir / f"{policy_id}.pt"))
    torch.save(net.state_dict(), entry.path)
    self.entries[policy_id] = entry
    self._save_index()
    return entry

  def add_exploiter(self, net, update, against_victim, win_rate=None):
    policy_id = f"expl_u{int(update)}_v{against_victim}"
    if policy_id in self.entries:
      self.entries[policy_id].win_rate = win_rate
      self._save_index()
      return self.entries[policy_id]
    entry = RosterEntry(policy_id, "exploiter", int(update),
                        win_rate=win_rate,
                        path=str(self.save_dir / f"{policy_id}.pt"))
    torch.save(net.state_dict(), entry.path)
    self.entries[policy_id] = entry
    self._save_index()
    return entry

  def prune(self, keep_recent=4, keep_spaced=4):
    """Bounds snapshot/exploiter storage (keeps main always).

    Keeps the ``keep_recent`` most recent non-main entries plus ``keep_spaced``
    older ones spread across the run's history; deletes the rest's weight files.

    Spacing is by birth_update, not by list index. The caller prunes after every
    snapshot, so this runs against its own output dozens of times, and an
    index-based rule is not stable under that: the survivors are already
    collapsed toward the ends, so it re-selects the ends each pass until nothing
    mid-run is left. Spacing by age re-derives the targets from the true range
    every call.
    """
    non_main = [e for e in self.entries.values() if e.role != "main"]
    if len(non_main) <= keep_recent + keep_spaced:
      return
    ordered = sorted(non_main, key=lambda e: e.birth_update)
    older = ordered[:-keep_recent] if keep_recent else list(ordered)
    keep = {e.policy_id for e in (ordered[-keep_recent:] if keep_recent else [])}

    if keep_spaced and older:
      lo, hi = older[0].birth_update, older[-1].birth_update
      if hi > lo and keep_spaced > 1:
        targets = [lo + (hi - lo) * k / (keep_spaced - 1)
                   for k in range(keep_spaced)]
      else:
        targets = [lo]
      for t in targets:
        # Nearest surviving older entry to this age, preferring the earlier one
        # on a tie so the oldest snapshot is never dropped in favour of a
        # near-duplicate.
        keep.add(min(older,
                     key=lambda e: (abs(e.birth_update - t), e.birth_update))
                 .policy_id)
      # Targets can collide once the older block is sparse. Spend any leftover
      # budget on whichever remaining entry is furthest (in age) from everything
      # already kept, which is the same objective the targets encode.
      budget = keep_recent + keep_spaced
      while len(keep) < budget:
        rest = [e for e in older if e.policy_id not in keep]
        if not rest:
          break
        kept_ages = [e.birth_update for e in ordered if e.policy_id in keep]
        keep.add(max(rest, key=lambda e: min(abs(e.birth_update - a)
                                             for a in kept_ages)).policy_id)

    for e in ordered:
      if e.policy_id in keep:
        continue
      p = Path(e.path)
      if p.exists():
        p.unlink()
      del self.entries[e.policy_id]
    self._save_index()

  def load_net(self, policy_id, agent_fn, num_actions, input_shape, device):
    entry = self.get(policy_id)
    if entry is None:
      return None
    if not os.path.exists(entry.path):
      return None
    net = agent_fn(num_actions, input_shape, device)
    sd = torch.load(entry.path, map_location=device)
    net.load_state_dict(sd)
    net.eval()
    return net.to(device)

  def load_all_nets(self, policy_ids, agent_fn, num_actions, input_shape,
                    device):
    return {pid: self.load_net(pid, agent_fn, num_actions, input_shape,
                               device) for pid in policy_ids}

  def opponent_ids(self, exclude_main=True, roles=None):
    ids = [e.policy_id for e in self.entries.values()
           if (not exclude_main or e.role != "main")]
    if roles is not None:
      ids = [pid for pid in ids if self.get(pid).role in roles]
    return ids


class Matchmaker(object):
  """Samples per-env lineups from a roster for league self-play.

  Each environment gets a fixed 4-seat lineup sampled with a mix of pure
  self-play (all seats = main) and mixed lineups (main + opponent snapshots).
  For v1 the lineup is fixed per env for the whole run (lineups are cheap to
  keep; per-batch prioritized matchmaking is a planned refinement).
  """

  def __init__(self, roster, num_envs, num_players, train_pid="main",
               selfplay_fraction=0.5, old_fraction=0.15, seed=0,
               rng=None, max_live_opponents=4, live_refresh=2000):
    self.roster = roster
    self.num_envs = num_envs
    self.num_players = num_players
    self.train_pid = train_pid
    self.selfplay_fraction = selfplay_fraction
    self.old_fraction = old_fraction
    self.rng = rng if rng is not None else np.random.RandomState(seed)
    # See _live_opponents. 0 disables the bound (pre-2026-08-13 behaviour).
    self.max_live_opponents = int(max_live_opponents)
    self.live_refresh = int(live_refresh)
    self._live = None
    self._live_age = 0

  def _live_opponents(self):
    """A bounded, periodically-resampled subset of the roster to draw from.

    Sampling uniformly over the whole roster makes throughput decay as the run
    gets longer, and the mechanism is not obvious. ``PPO._act_sparse`` groups the
    batch by policy id and runs **one encoder forward per distinct policy**
    (ppo.py: `groups = [flatnonzero(pids == p) for p in unique(pids)]`). The
    total row count is unchanged, but the launches get smaller and more numerous.
    Measured on an RTX 4080 Laptop, 256 rows through the production encoder:

        K distinct policies :   1     2     4     8    16    32
        cost vs K=1         : 1.00x 1.00x 1.12x 2.05x 4.26x 8.32x

    An unbounded roster grows K without limit -- an 8h run at
    --snapshot_every=100 reaches ~35 snapshots, i.e. several times the act cost
    it started with, decaying continuously. Observed directly: a league smoke run
    with 10 snapshots had already grown act from 22.8 to 41.7 ms/step.

    Bounding the *live* set keeps K small while leaving long-run coverage
    uniform: every snapshot still enters play, just clustered in time rather than
    interleaved. This is the usual shape of large-scale league play (a fixed
    number of opponent slots per rollout batch) rather than a new idea.
    """
    ids = self.roster.opponent_ids(exclude_main=True)
    if self.max_live_opponents <= 0 or len(ids) <= self.max_live_opponents:
      return ids
    stale = (self._live is None
             or self._live_age >= self.live_refresh
             # A snapshot in the live set was removed from the roster.
             or not set(self._live).issubset(ids))
    if stale:
      self._live = [str(p) for p in self.rng.choice(
          ids, size=self.max_live_opponents, replace=False)]
      self._live_age = 0
    self._live_age += 1
    return self._live

  def sample_lineup(self):
    """One (num_players,) lineup of policy ids.

    Mixed lineups always keep ``train_pid`` on seat 0 (so main always learns
    from that env); the remaining seats draw opponents, occasionally main.
    """
    if self.rng.rand() < self.selfplay_fraction:
      return [self.train_pid] * self.num_players
    opponents = self._live_opponents()
    picks = [self.train_pid]
    for _ in range(1, self.num_players):
      if opponents and self.rng.rand() < (1.0 - self.old_fraction):
        picks.append(str(self.rng.choice(opponents, size=1)[0]))
      else:
        picks.append(self.train_pid)
    return picks

  def lineups(self):
    """(num_envs, num_players) policy-id array for all environments."""
    return np.asarray([self.sample_lineup() for _ in range(self.num_envs)],
                      dtype=object)
