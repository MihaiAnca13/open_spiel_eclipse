# Copyright 2019 DeepMind Technologies Limited
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

"""Factorization of Eclipse's flat action space, derived from the engine.

Eclipse exposes 11,117 flat action ids, and a `Linear(width, 11117)` policy head
is 80-86% of the whole network. It also generalizes badly: the weight vector for
"colonize cell 42, slot 3, money track" shares nothing with "colonize cell 42,
slot 4, science track", so every one of the 5,400 colony-ship actions has to be
learned independently.

But the ids are not arbitrary -- the engine builds them as products
(`cell*24 + slot*3 + track` and friends). Recovering that structure lets the head
be a sum of small factor embeddings,

    W[a] = sum_f E[decode[a, f]]

so learning something about cell 42 transfers to every action targeting cell 42,
and the parameter count drops from 11117*H to a few hundred*H.

The factor assignment is read out of ``action_to_string`` rather than hardcoded
from the id arithmetic. The arithmetic is genuinely treacherous here: the
constants in eclipse.cc carry stale comments (they were written when
UPGRADE_PART_COUNT was 30, so e.g. `action_move` is annotated 8502 while its real
value is 9142), and several ids inside the nominal ranges are dead. Parsing the
strings is authoritative, self-checking, and survives future re-numbering.
"""

import re

import numpy as np

# Every action gets exactly this many embedding slots. The widest family is
# COLONY_SHIP: family + cell + slot + track. Unused slots point at a per-family
# "absent" row, which is constant within a family and therefore acts as a family
# bias -- no coupling between families.
NUM_SLOTS = 4

_PATTERNS = [
    # name, regex, factor names in group order
    ("colony", re.compile(r"^COLONY_SHIP_(-?\d+_-?\d+)_SLOT(\d+)_(\w+)$"),
     ("cell", "slot", "track")),
    ("build", re.compile(r"^BUILD_([A-Z_]+?)_(-?\d+_-?\d+)$"), ("btype", "cell")),
    ("upgrade", re.compile(r"^UPGRADE_([A-Z]+)_SLOT(\d+)_(.+)$"),
     ("ship", "slot", "part")),
    ("move_unit", re.compile(r"^MOVE_UNIT_(\d+)_(?!WARP$)(\w+)$"),
     ("unit", "dir")),
    ("move_warp", re.compile(r"^MOVE_UNIT_(\d+)_WARP$"), ("unit",)),
    ("warp_dest", re.compile(r"^MOVE_WARP_TO_(-?\d+_-?\d+)$"), ("cell",)),
    ("influence_to", re.compile(r"^INFLUENCE_TO_(-?\d+_-?\d+)$"), ("cell",)),
    ("reclaim", re.compile(r"^RECLAIM_FROM_(-?\d+_-?\d+)$"), ("cell",)),
    ("retreat", re.compile(r"^RETREAT_TO_(-?\d+_-?\d+)$"), ("cell",)),
    ("infl_cell", re.compile(r"^COMBAT_INFLUENCE_TO_(-?\d+_-?\d+)$"), ("cell",)),
]


class ActionFactorization:
  """Decode table mapping each action id to its embedding rows.

  Attributes:
    decode: (num_actions, NUM_SLOTS) int64 array of row indices into the
      embedding table.
    num_rows: number of embedding rows required.
    families: per-action family name (for reporting).
    stats: dict of family -> action count.
  """

  def __init__(self, decode, num_rows, families, stats):
    self.decode = decode
    self.num_rows = num_rows
    self.families = families
    self.stats = stats

  def summary(self):
    factored = sum(n for f, n in self.stats.items() if f != "atom")
    return (f"{len(self.families)} actions -> {self.num_rows} embedding rows "
            f"({factored} factored, {self.stats.get('atom', 0)} unfactored); "
            f"families: " +
            ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items())))


def build_action_factorization(action_strings):
  """Builds the decode table from ``action_strings[a] -> str``.

  Args:
    action_strings: sequence of length num_actions with each action's string.

  Returns:
    An ActionFactorization.
  """
  num_actions = len(action_strings)
  rows = {}          # (factor, value) -> row index

  def row_of(factor, value):
    key = (factor, value)
    if key not in rows:
      rows[key] = len(rows)
    return rows[key]

  decode = np.zeros((num_actions, NUM_SLOTS), dtype=np.int64)
  families = []
  stats = {}
  for action in range(num_actions):
    text = action_strings[action]
    family, values = "atom", ()
    for name, pattern, factor_names in _PATTERNS:
      match = pattern.match(text)
      if match:
        family = name
        values = tuple(zip(factor_names, match.groups()))
        break
    if family == "atom":
      # Its own row: headers, stops, tech picks, dice targets, diplomacy, and
      # the dead ids all keep an independent weight vector. Deliberately no
      # pruning of the dead ones -- a "never legal" id costs one row and cannot
      # be selected, whereas pruning by *observed* legality would cap what a
      # better policy can reach.
      slots = [row_of("atom", action)]
    else:
      slots = [row_of("family", family)]
      slots += [row_of(f, v) for f, v in values]
    absent = row_of("absent", family)
    slots += [absent] * (NUM_SLOTS - len(slots))
    if len(slots) != NUM_SLOTS:
      raise ValueError(
          f"action {action} ({text}) needs {len(slots)} slots > {NUM_SLOTS}")
    decode[action] = slots
    families.append(family)
    stats[family] = stats.get(family, 0) + 1
  return ActionFactorization(decode, len(rows), families, stats)


def factorization_from_game(game, player=0):
  """Convenience: build the factorization from a pyspiel game.

  Uses a non-chance state, because at a chance node ``action_to_string`` returns
  the chance-resolution string for every id.
  """
  state = game.new_initial_state()
  while state.is_chance_node():
    state.apply_action(state.chance_outcomes()[0][0])
  if player is None or state.current_player() < 0:
    player = 0
  strings = [state.action_to_string(state.current_player(), a)
             for a in range(game.num_distinct_actions())]
  return build_action_factorization(strings)
