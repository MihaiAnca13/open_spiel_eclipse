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
"""Python mirror of the Eclipse observation-tensor layout.

The authority is ``open_spiel/games/eclipse/observation.h``. This module
restates it so Python code never hardcodes an offset again -- the previous
scattered constants (``SCORE_SELF_SLOT``, ``GALAXY_BASE``, ``CELL_STRIDE``, the
offsets buried in ``phi_soft_*``) were spread across ``ppo_eclipse.py`` and all
broke silently whenever the C++ layout moved.

``validate(game)`` is the guard: it asserts TOTAL equals the game's real
observation size, so a C++ layout change fails loudly here instead of
mis-reshaping a tensor deep inside the encoder.

Layout shape, seat-relative from the viewing player:

    A  global                        146
    B  6 x player block (547)       3282   slot 0 is ALWAYS the viewer
    C  galaxy 225 cells x 69       15525   cell-major, channel-minor
    D  tech market                    88
    E  combat                        940
    F  upkeep                          91
    G  action sub-states              367
                                  ------
                                   20439
"""

# ── cardinalities ─────────────────────────────────────────────────────────
MAX_SEATS = 6
SEAT_SLOTS = MAX_SEATS
REL_SEAT_WIDTH = MAX_SEATS + 2      # seat 0..5, NPC, none
SPECIES_COUNT = 7
SHIP_TYPE_COUNT = 7
PLAYER_SHIP_TYPES = 4
NPC_SHIP_TYPES = 3
PLANET_TYPE_COUNT = 8
DIE_COLOR_COUNT = 5
SHIP_PART_COUNT = 43
BLUEPRINT_SLOTS = 8
TECH_BIT_COUNT = 40
TECH_TRACK_COUNT = 3
TECH_TRAY_COUNT = 40
BUILD_TYPE_COUNT = 6
REP_SLOTS = 5
REP_SLOT_KIND_COUNT = 3
REP_TILE_VALUE_COUNT = 5
MINOR_SPECIES_COUNT = 9
SECTOR_TYPE_COUNT = 7
RING_KINDS = 4
HEX_DIRECTIONS = 6
GALAXY_DIM = 15
GALAXY_CELLS = GALAXY_DIM * GALAXY_DIM   # 225

# ── block sizes ───────────────────────────────────────────────────────────
GLOBAL_SIZE = 146
PLAYER_SIZE = 547
CELL_CHANNELS = 88
GALAXY_SIZE = GALAXY_CELLS * CELL_CHANNELS
TECH_MARKET_SIZE = 88
COMBAT_SIZE = 940
UPKEEP_SIZE = 91
ACTION_STATES_SIZE = 367

# ── block offsets ─────────────────────────────────────────────────────────
GLOBAL_START = 0
PLAYERS_START = GLOBAL_START + GLOBAL_SIZE
GALAXY_START = PLAYERS_START + SEAT_SLOTS * PLAYER_SIZE
TECH_MARKET_START = GALAXY_START + GALAXY_SIZE
COMBAT_START = TECH_MARKET_START + TECH_MARKET_SIZE
UPKEEP_START = COMBAT_START + COMBAT_SIZE
ACTION_STATES_START = UPKEEP_START + UPKEEP_SIZE
TOTAL = ACTION_STATES_START + ACTION_STATES_SIZE

# ── offsets inside one player block ───────────────────────────────────────
P_OCCUPIED = 0
P_IS_VIEWER = 1
P_IS_AI = 2
P_ELIMINATED = 3
P_HAS_PASSED = 4
P_SPECIES = 5                    # + SPECIES_COUNT
P_ACTIVATIONS = 12               # + 6
P_TURN_POS = 18                  # + MAX_SEATS
P_PASS_POS = 25                  # + MAX_SEATS + 1
P_VP_TOTAL = 32
P_VP_BREAKDOWN = 33              # + 9 scoring categories
P_VP_AT_ELIM = 42
P_GOLD = 44
P_SCIENCE = 45
P_MATERIALS = 46
P_PROD_INDEX = 47                # + 3, cubes REMAINING on the track
P_PRODUCTION = 50                # + 3, the ACTUAL production value
P_INCOME = 53
P_UPKEEP_COST = 54
P_NET_CASH = 55
P_SOLVENT = 56
P_DISKS_ON_SECTORS = 57
P_DISKS_AVAILABLE = 60
P_COLONY_TOTAL = 62
P_COLONY_AVAIL = 63
P_ORBITALS = 64
P_MONOLITHS = 65
P_TRADE_RATE = 66
P_GRAVEYARD = 67                 # + 3
P_TECH_BITS = 70                 # + TECH_BIT_COUNT
P_BLUEPRINTS = 116               # + PLAYER_SHIP_TYPES * BLUEPRINT_SIZE
P_PARTS_INV = 388                # + SHIP_PART_COUNT
P_REP_TRACK = 432                # + REP_SLOTS * REP_SLOT_SIZE
P_AMBASSADOR_HELD = 522
P_TRAITOR = 524
P_DISCOVERY_VP = 525
P_MINOR_SPECIES = 526            # + MINOR_SPECIES_COUNT
P_WARP_ELIGIBLE = 535
P_ARTIFACT_CHUNKS = 536
P_BUILD_COST = 537               # + BUILD_TYPE_COUNT

SHIP_STATS_SIZE = 6 + 2 * DIE_COLOR_COUNT                       # 16
BLUEPRINT_SIZE = SHIP_STATS_SIZE + SHIP_PART_COUNT + BLUEPRINT_SLOTS + 1   # 68
REP_SLOT_SIZE = REP_SLOT_KIND_COUNT + 1 + REL_SEAT_WIDTH + REP_TILE_VALUE_COUNT + 1  # 18
REP_SLOT_AMBASSADOR_FROM = REP_SLOT_KIND_COUNT + 1              # 4

# ── galaxy cell channels (must match observation.h's CellChannel enum) ────
C_PRESENT = 0
C_OWNER = 1                      # + REL_SEAT_WIDTH
C_POINTS = 9
C_WORMHOLE = 10                  # + HEX_DIRECTIONS, ROTATED
C_PLANET_PRINTED = 16            # + PLANET_TYPE_COUNT
C_PLANET_POPULATED = 24          # + PLANET_TYPE_COUNT
C_ORBITAL = 32
C_MONOLITH = 33
C_ORBITAL_POP_SLOT = 34
C_DISCOVERY_PRESENT = 35         # presence only -- identity is deliberately hidden
C_WARP_PORTAL_VP = 36
C_HAS_WARP_PORTAL = 37
C_HAS_ARTIFACT = 38
C_HAS_GUARDIAN = 39
C_IS_GCDS = 40
C_RING = 41                      # + RING_KINDS
C_MY_SHIPS = 45                  # + PLAYER_SHIP_TYPES
C_ENEMY_SHIPS = 49               # + PLAYER_SHIP_TYPES
C_NPC_SHIPS = 53                 # + NPC_SHIP_TYPES
C_DAMAGE = 56
C_MY_ANCHOR = 57
C_COMBAT_ACTIVE = 58
C_IN_BATTLE_QUEUE = 59
C_INFLUENCE_UNCONTROLLED = 60
C_MOVE_ACTIVE_UNIT = 61
C_EXPLORE_ZONE = 62
C_WARP_LINK = 63                 # + HEX_DIRECTIONS
C_LAYOUT_KIND = 69               # + SECTOR_TYPE_COUNT (warped only)
C_WARP_DEST_CELL = 76            # + HEX_DIRECTIONS (warped only)
C_WARP_DEST_DIR = 82             # + HEX_DIRECTIONS (warped only)


def player_block_start(slot):
  """Start index of a seat block. Slot 0 is always the viewing player."""
  return PLAYERS_START + slot * PLAYER_SIZE


def cell_start(cell):
  """Start index of a galaxy cell (cell = hex_to_index(q, r), 0..224)."""
  return GALAXY_START + cell * CELL_CHANNELS


def hex_to_index(q, r):
  """Mirror of galaxy.h:120 -- (q+7)*15 + (r+7)."""
  return (q + 7) * GALAXY_DIM + (r + 7)


def seat_for_slot(slot, viewer, num_players):
  """Absolute seat id occupying a block slot, for a given viewer."""
  return (viewer + slot) % num_players


def slot_for_seat(seat, viewer, num_players):
  """Block slot holding an absolute seat id, for a given viewer."""
  return (seat - viewer + num_players) % num_players


def galaxy_view(obs):
  """Reshape the flat galaxy block to (CELL_CHANNELS, 15, 15) for a conv tower.

  The C++ writer is cell-major / channel-minor, so the reshape order below is
  load-bearing: getting it wrong scrambles the grid silently rather than
  raising. Accepts a numpy array or torch tensor of shape (..., TOTAL).
  """
  block = obs[..., GALAXY_START:GALAXY_START + GALAXY_SIZE]
  shaped = block.reshape(*block.shape[:-1], GALAXY_DIM, GALAXY_DIM,
                         CELL_CHANNELS)
  ndim = len(shaped.shape)
  perm = tuple(range(ndim - 3)) + (ndim - 1, ndim - 3, ndim - 2)
  if hasattr(shaped, "permute"):          # torch
    return shaped.permute(*perm)
  return shaped.transpose(perm)           # numpy


def validate(game_or_size):
  """Assert this module matches the engine. Call once at startup.

  Accepts a pyspiel game or a raw observation size.
  """
  size = (game_or_size if isinstance(game_or_size, int)
          else game_or_size.observation_tensor_shape()[0])
  if size != TOTAL:
    raise ValueError(
        f"obs_layout.TOTAL={TOTAL} but the engine reports {size}. "
        "open_spiel/games/eclipse/observation.h changed -- update this module "
        "to match before running anything that reads observation offsets.")
  return size


def _self_check():
  """Internal consistency of the derived constants."""
  assert PLAYER_SIZE == 547, PLAYER_SIZE
  assert P_BLUEPRINTS == P_TECH_BITS + TECH_BIT_COUNT + 2 * TECH_TRACK_COUNT
  assert P_PARTS_INV == P_BLUEPRINTS + PLAYER_SHIP_TYPES * BLUEPRINT_SIZE
  assert P_REP_TRACK == P_PARTS_INV + SHIP_PART_COUNT + 1
  assert P_AMBASSADOR_HELD == P_REP_TRACK + REP_SLOTS * REP_SLOT_SIZE
  assert P_BUILD_COST + BUILD_TYPE_COUNT <= PLAYER_SIZE
  assert C_WARP_DEST_DIR + HEX_DIRECTIONS == CELL_CHANNELS
  assert TOTAL == 24714, TOTAL


_self_check()
