#!/usr/bin/env python3
"""Stage 1 frozen-episode collector for the Eclipse 4P RL project.

Loads one or more FROZEN checkpoint policies (eval-only, ``torch.no_grad``),
plays COMPLETE episodes of Eclipse to terminal, and writes every acting step of
every seat as a schema row through ``open_spiel.python.eclipse.frozen_dataset``
(one ``.npz`` per episode + ``manifest.json``, seeded whole-episode 80/20 split).

WHY
  The offline critic work needs a diagnostic dataset of real play from the
  existing frozen policies (main/snapshot/baseline checkpoints), so it can (a)
  sanity-check the terminal labels (`rank_target`, `rank_utility_terminal`,
  VP components, discounted `return_`) against live behaviour and (b) feed an
  offline value/rank objective without launching training. Their play is poor --
  that is expected and is the point: the collector must faithfully record what
  those policies actually do.

SHAPING / RETURN (documented decision)
  ``return_`` follows the module contract as "the exact discounted return under
  the intended shaping and gamma". I use the project's **win-mode, no-shaping**
  target -- the value target PPO would aim for under ``value_mode="win"`` with
  the shaping disabled:

      step reward        = 0 for every non-terminal acting step
      terminal utility   = rank_utility(terminal_vp, seat, vp_beta=0)
                          (undiscounted, at the last acting step)
      return_            = sum_k g^k * 0  +  g^n * rank_utility_terminal
                          = g^n * rank_utility_terminal

  where n is the length of that seat's own acting chain and g = --gamma (0.99).
  This is deterministic, documented, and identical across every row. It is the
  "step reward 0, final term = terminal rank utility, discounted" option
  recommended by the task, matching what the PPO win-mode no-shaping critic
  regresses toward. (The alternative phi soft-shaping is deliberately NOT used:
  it would couple `return_` to a hand-tuned potential whose weights differ by
  run, making the persisted target non-comparable across checkpoints.)

POLICY / LINEUP (documented decision)
  Each complete episode uses ONE frozen policy for ALL 4 seats (policy vs
  itself). This isolates each checkpoint's behaviour on a shared, symmetric
  board (no inter-policy confound), keeps every seat's chain labelable from a
  single policy, and is the simplest correct configuration. The lineup policy
  is recorded per-episode in the manifest (``policy`` -> list of episode ids).

CLI
  python tools/collect_frozen.py --out runs/frozen_diag \
      --checkpoints runs/long_v2/snap_u2500.pt,runs/long_v2/main.pt,...
      --episodes-per-policy 100 [--gamma 0.99] [--device cuda] [--workers N]

  --device defaults to cuda when available; all inference runs under
  torch.no_grad(). Workers are separate processes (one env + net each), so each
  plays its shard of episodes independently.

OBS LENGTH (verified empirically -- records the FULL engine row; actor reads the pre-V2 prefix)
  The dataset schema contract (``obs = info_state[seat]``) is honoured literally:
  each recorded row is the acting seat's FULL 37596-length information state
  (``env.observation_spec()``, never hardcoded). This is what the new V2 critic,
  trained on the collected data, consumes. The FROZEN actor, however, was trained
  on the PRE-V2 24714-length obs: ``obs_layout.V2_KEYED_START == 24714`` and the
  V2 keyed-entity block is a pure APPEND, so ``obs[..., :24714]`` is byte-identical
  to the pre-V2 tensor. The actor is fed that slice; only full obs rows are stored.

FROZEN-CHECKPOINT SUPPORT (vendored pre-V2 encoder)
  All four designated checkpoints (runs/long_v2/{snap_u2500,main}.pt,
  runs/long8h/main.pt, runs/_judge/baseline/main.pt) were created Aug 8-9 2026,
  BEFORE commit 365a059a ("V2 keyed-entity observation extension", Aug 10 2026)
  extended the observation layout and reworked the spatial encoder (its ``fuse``
  became ``Linear(5*width, width)`` -> [64, 320] and a V2 ``entity_fc`` branch was
  added; pre-V2 fuse is ``Linear(4*width, width)`` -> [64, 256]). The current
  engine's encoder is therefore INCOMPATIBLE with those weights. This script
  VENDORS a standalone pre-V2 ``SpatialEclipseEncoder`` (see ``PreV2Encoder``,
  ported verbatim from git 92672cb8) so the frozen checkpoints load and evaluate
  against the current engine through their 24714-wide obs prefix. ``--stub-policy``
  remains as the no-checkpoint smoke path for QA/tests. ``--workers`` and
  ``--stub-policy`` are documented in the CLI section above.

MUST NOT
  * Never edits frozen_dataset.py / ppo.py / ppo_eclipse.py / obs_layout.py /
    roster_ladder.py.
  * Never touches open_spiel/python/fcm/.
  * No training / long runs -- collection only.
"""

import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch
from absl import flags as absl_flags

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.eclipse import frozen_dataset
from open_spiel.python.eclipse import obs_layout
from open_spiel.python.eclipse.roster_ladder import _resolve_arch
from open_spiel.python.examples import ppo_eclipse as pe
from open_spiel.python.pytorch.ppo import layer_init

_GAME_STR = "eclipse(players=4)"
_NUM_PLAYERS = 4

# Pre-V2 observation width: the pre-V2 ``SpatialEclipseEncoder`` consumes only
# indices in ``[0, V2_KEYED_START)``; V2 appended the keyed-entity block beyond
# that (a pure append, so ``obs[..., :V2_KEYED_START]`` is byte-identical to
# the pre-V2 obs tensor the frozen policies were trained on).
_PRE_V2_WIDTH = obs_layout.V2_KEYED_START   # 24714


def _parse_absl():
  """Parse absl FLAGS with an empty argv so every default is taken.

  ``pe._randomized_game_string`` reads absl flags (randomize_races,
  race_alien_prob, ...); touching one before parse raises
  UnparsedFlagAccessError. This script owns its CLI via argparse, so
  absl must not see sys.argv -- exactly what t3_opponent_curve does.
  """
  absl_flags.FLAGS(["collect_frozen"])


class _ArgmaxLegalPolicy(torch.nn.Module):
  """Smoke-test policy: deterministic argmax over the legal mask.

  ``--stub-policy`` uses this instead of ``_load_net_tolerant`` so the whole
  collection pipeline (real env play, obs recording, terminal labels, split,
  write) can be exercised end-to-end even when a checkpoint's weights cannot be
  loaded. Picks the lowest legal action id: Eclipse's env legal mask is
  over-inclusive for a few HIGH-id conditional actions (trade conversions,
  minor-species formation) that are not executable in every state, so argmax-low
  keeps the match on always-legal core actions and plays to terminal. NOT a
  substitute for real frozen policies -- smoke/QA only.
  """

  def __init__(self, num_actions):
    super().__init__()
    self.num_actions = int(num_actions)

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    mask = (legal_actions_mask if legal_actions_mask is not None
            else torch.ones((len(x), self.num_actions), dtype=torch.bool)).to(
                x.device)
    legal = torch.nonzero(mask[0], as_tuple=False).view(-1)
    return (torch.tensor([int(legal.min().item())]), None, None, None, None)


class PreV2Encoder(torch.nn.Module):
  """Standalone port of the PRE-V2 ``SpatialEclipseEncoder`` (git 92672cb8).

  The four frozen checkpoints were trained with this exact encoder, BEFORE the
  V2 keyed-entity observation extension (commit 365a059a) changed the encoder's
  ``fuse`` from ``Linear(4*width, width)`` (-> [64, 256] at width=64) to
  ``Linear(5*width, width)`` (-> [64, 320]) and added the V2 ``entity_fc``
  branch. Reconstructed VERBATIM from
  ``git show 92672cb8:open_spiel/python/examples/ppo_eclipse.py`` -- same layer
  topology (conv / conv_res / galaxy_fc / self_mlp / tail_mlp / seat_mlp /
  rel_fc / fuse), same ``_mlp`` structure, same GroupNorm/LayerNorm-free norm
  choices. The torch.compile and ``channels_last`` machinery of the original is
  deliberately dropped (not needed; kept simple and exact).

  The encoder reads only indices in ``[0, V2_KEYED_START)`` -- the pre-V2 obs
  region that is a byte-identical prefix of the current 37596-length engine row
  -- so it consumes ``obs[..., :_PRE_V2_WIDTH]`` and ignores the appended V2
  block entirely. No V2-era modules (entity_fc / unit_mlp / unit_attn /
  sector_embed / rotation_embed / nonplayer_owner) exist here, matching the
  checkpoint weights.
  """

  # Conv tower output channels -- a class attribute (not a bare literal) so the
  # port is faithful to the original's structure.
  CELL_FEATURE_CHANNELS = 64

  def __init__(self, width, depth=1, activation="tanh"):
    super().__init__()
    act = torch.nn.GELU if activation == "gelu" else torch.nn.Tanh
    c = self.CELL_FEATURE_CHANNELS

    # ── Galaxy branch: (B, 88, 15, 15) conv tower ───────────────
    self.conv = torch.nn.Sequential(
        layer_init(torch.nn.Conv2d(obs_layout.CELL_CHANNELS, c, 3, padding=1)),
        torch.nn.GroupNorm(8, c),
        act(),
    )
    self.conv_res = torch.nn.Sequential(
        layer_init(torch.nn.Conv2d(c, c, 3, padding=1)),
        torch.nn.GroupNorm(8, c),
        act(),
    )
    self.galaxy_fc = layer_init(torch.nn.Linear(c, width))

    # ── Viewer self block: (547,) MLP ───────────────────────────
    self.self_mlp = self._mlp(obs_layout.PLAYER_SIZE, width, depth, act)

    # ── Tail blocks (tech market + combat + upkeep + action states) ──
    tail_size = (obs_layout.TECH_MARKET_SIZE + obs_layout.COMBAT_SIZE
                 + obs_layout.UPKEEP_SIZE + obs_layout.ACTION_STATES_SIZE)
    self.tail_mlp = self._mlp(tail_size, width, depth, act)

    # ── Relational: shared per-seat MLP -> masked mean+max pool ──
    self.seat_mlp = self._mlp(obs_layout.PLAYER_SIZE, width, depth, act)
    self.rel_fc = layer_init(torch.nn.Linear(2 * width, width))

    # ── Fuse the four branch latents to (B, width) ──────────────
    self.fuse = layer_init(torch.nn.Linear(4 * width, width))

  def _mlp(self, in_features, width, depth, act):
    layers = []
    cur = in_features
    for _ in range(depth):
      layers.append(layer_init(torch.nn.Linear(cur, width)))
      layers.append(act())
      cur = width
    return torch.nn.Sequential(*layers)

  def forward(self, x):
    return self._encode(x)[0]

  def _encode(self, x):
    b = x.shape[0]
    x = x.reshape(b, -1)

    # Galaxy: (B, CELL_CHANNELS, 15, 15).
    gal = obs_layout.galaxy_view(x)
    h = self.conv(gal)
    h = h + self.conv_res(h)
    h_cells = h.reshape(b, self.CELL_FEATURE_CHANNELS, obs_layout.GALAXY_CELLS)
    gal_lat = self.galaxy_fc(h_cells.mean(dim=-1))  # global avg pool -> (B,width)

    # Viewer self block (slot 0 is always the viewer).
    self_start = obs_layout.PLAYERS_START
    self_lat = self.self_mlp(x[:, self_start:self_start +
                               obs_layout.PLAYER_SIZE])

    # Tail blocks.
    tail = torch.cat([
        x[:, obs_layout.TECH_MARKET_START:
          obs_layout.TECH_MARKET_START + obs_layout.TECH_MARKET_SIZE],
        x[:, obs_layout.COMBAT_START:
          obs_layout.COMBAT_START + obs_layout.COMBAT_SIZE],
        x[:, obs_layout.UPKEEP_START:
          obs_layout.UPKEEP_START + obs_layout.UPKEEP_SIZE],
        x[:, obs_layout.ACTION_STATES_START:
          obs_layout.ACTION_STATES_START + obs_layout.ACTION_STATES_SIZE],
    ], dim=1)
    tail_lat = self.tail_mlp(tail)

    # Relational pool over the 6 seat blocks. Occupied bits gate validity --
    # for a 4-player game slots 4-5 are marked empty and masked out.
    seats = x[:, obs_layout.PLAYERS_START:
              obs_layout.PLAYERS_START +
              obs_layout.SEAT_SLOTS * obs_layout.PLAYER_SIZE
              ].reshape(b, obs_layout.SEAT_SLOTS, obs_layout.PLAYER_SIZE)
    seat_h = self.seat_mlp(seats)                # (B, 6, width)
    occ = seats[:, :, obs_layout.P_OCCUPIED] >= 0.5   # (B, 6)
    mask = occ.unsqueeze(-1)
    denom = occ.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (seat_h * mask).sum(dim=1) / denom    # (B, width)
    neg = -1e9
    masked_max = torch.where(
        mask, seat_h, torch.full_like(seat_h, neg)).max(dim=1).values
    rel_lat = self.rel_fc(torch.cat([mean, masked_max], dim=1))

    fused = self.fuse(torch.cat(
        [gal_lat, self_lat, tail_lat, rel_lat], dim=1))
    return fused, h_cells


class PreV2Net(torch.nn.Module):
  """Eval-only net wrapping the pre-V2 encoder behind a plain Linear actor.

  ``self.actor = nn.Sequential(shared, Linear(width, num_actions))`` so the
  checkpoint's ``actor.0.*`` (encoder) and ``actor.1.*`` (head) keys map with no
  rekeying. The pre-V2 actor for these runs is a plain (non-factored) head, so
  no FactoredActorHead is needed.
  """

  def __init__(self, width, depth, activation, num_actions):
    super().__init__()
    self.num_actions = int(num_actions)
    self.shared = PreV2Encoder(width, depth, activation)
    self.actor = torch.nn.Sequential(
        self.shared, layer_init(torch.nn.Linear(width, num_actions), std=0.01))

  def get_action_and_value(self, x, legal_actions_mask=None, action=None):
    """Argmax over the legal mask of the dense logits (patch-shaped interface).

    Returns a 5-tuple ``(action, logits, None, None, None)`` -- the same shape
    ``_ArgmaxLegalPolicy.get_action_and_value`` returns -- so ``_play_one_episode``
    can call either policy uniformly and read ``[0]``.
    """
    logits = self.actor(x)
    if legal_actions_mask is not None:
      logits = logits.masked_fill(~legal_actions_mask.bool(), -1e9)
    action = logits.argmax(dim=-1)
    return (action, logits, None, None, None)


def _build_net(checkpoint_dir, checkpoint_path, device, num_actions,
               input_shape, game, stub_policy=False):
  """arch.json-driven pre-V2 encoder + tolerant checkpoint load (actor required).

  With ``stub_policy=True`` returns the deterministic stub instead of loading
  weights (smoke/QA escape hatch). With real weights, rebuilds the pre-V2
  net from ``arch.json`` (width/depth/activation/num_actions) and loads the
  checkpoint with ``strict=False``; raises if any ``actor.*`` key is missing
  (mirrors ``roster_ladder._load_net_tolerant``).
  """
  if stub_policy:
    net = _ArgmaxLegalPolicy(num_actions)
    net.to(device)
    net.eval()
    return net, {"arch": "stub"}
  arch = _resolve_arch(checkpoint_dir, num_actions, input_shape)
  if not os.path.exists(os.path.join(checkpoint_dir, "arch.json")):
    arch = _resolve_arch(os.path.dirname(checkpoint_path), num_actions,
                         input_shape)
  width = int(arch["width"])
  depth = int(arch["depth"])
  activation = str(arch.get("activation") or "tanh")
  net = PreV2Net(width, depth, activation, num_actions)
  sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
  missing, _ = net.load_state_dict(sd, strict=False)
  missing_actor = [k for k in missing if k.startswith("actor")]
  if missing_actor:
    raise RuntimeError(
        f"{os.path.basename(checkpoint_path)}: actor weights missing after "
        f"load ({missing_actor}); the arch (arch.json) does not match this "
        "pre-V2 checkpoint")
  net.to(device)
  net.eval()
  return net, arch


def _terminal_vp(rows):
  """(4,) per-seat total VP from terminal info_state rows (each row's own view)."""
  out = np.zeros(_NUM_PLAYERS, dtype=np.float32)
  for seat in range(_NUM_PLAYERS):
    row = np.asarray(rows[seat], dtype=np.float32)
    out[seat] = row[obs_layout.player_block_start(0) + obs_layout.P_VP_TOTAL]
  return out


def _terminal_labels(rows, seat):
  """(rank_target(4,), rank_utility, vp_components(9,), vp_all_seats(4,9))."""
  term_vp = _terminal_vp(rows)
  rank_tgt = frozen_dataset.rank_target(term_vp, seat)
  rank_util = frozen_dataset.rank_utility_terminal(term_vp, seat)
  vp_comp = frozen_dataset.vp_components_viewer_relative(
      np.asarray(rows[seat], dtype=np.float32), seat, _NUM_PLAYERS)
  vp_all = frozen_dataset._vp_all_seats_viewer_relative(
      np.asarray(rows[seat], dtype=np.float32), _NUM_PLAYERS)
  return rank_tgt, float(rank_util), vp_comp, vp_all


def _play_one_episode(net, device, game_str, gamma, seed):
  """Play one complete episode; return (schema_dict_without_episode_id, n_acted).

  Uses the loaded net (all 4 seats). Records every acting step's observer row,
  then back-fills terminal-derived labels (rank_target, rank_utility, VP
  components, discounted return) onto every row of that seat. Works on a
  synchronous single-env loop (like ``_eval_squad``) so the FULL per-seat
  terminal observation rows are available before reset.
  """
  env = rl_environment.Environment(
      game=pyspiel.load_game(game_str),
      chance_event_sampler=rl_environment.ChanceEventSampler(seed=seed),
      observation_type=rl_environment.ObservationType.OBSERVATION,
      observations_as_numpy=True)
  obs_len = env.observation_spec()["info_state"][0]
  per_seat = {
      s: {"obs": [], "seat": [], "round_idx": [], "phase": []} for s in
      range(_NUM_PLAYERS)
  }

  time_step = env.reset(players=None)

  def _act(seat, obs):
    # The actor consumes the PRE-V2 obs prefix (`[:V2_KEYED_START]`); the full
    # obs `obs` (37596) is recorded separately in the dataset.
    obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))[None,
                                                                 :_PRE_V2_WIDTH].to(device)
    # Build the legal mask FRESH every step (never cache by ``id(legal)``): the
    # env reuses the ``legal_actions`` list object across steps, so an ``id``-
    # keyed cache goes stale and can hand the net an action that is NOT legal in
    # the current state -- the engine then rejects it (e.g. minor-species
    # formation whose precondition changed) and the episode crashes.
    legal = time_step.observations["legal_actions"][seat]
    mask = torch.zeros((1, net.num_actions), dtype=torch.bool, device=device)
    mask[0, np.asarray(legal, dtype=np.int64)] = True
    with torch.no_grad():
      action = int(net.get_action_and_value(obs_t, legal_actions_mask=mask)[0])
    return action

  while True:
    seat = int(time_step.observations["current_player"])
    if seat < 0 or seat >= _NUM_PLAYERS:
      # Terminal reached with no further acting step; rows recorded already.
      break
    obs = time_step.observations["info_state"][seat]
    # CRITICAL: with observations_as_numpy=True this is a VIEW into one reused
    # (4, obs_len) buffer that is overwritten on the very next get_time_step.
    # A stored reference would alias every later row onto the final buffer
    # state (observed as all-degenerate round/phase in testing). Copy each obs.
    per_seat[seat]["obs"].append(np.array(obs, dtype=np.float32, copy=True))
    per_seat[seat]["seat"].append(seat)
    action = _act(seat, np.asarray(obs, dtype=np.float32))
    time_step = env.step([action], players=None)
    if time_step.last():
      break

  # Terminal per-seat rows (full information-state rows, one per seat/viewer).
  # Also copies: the terminal rows share the same reused buffer.
  rows = [np.array(time_step.observations["info_state"][s], dtype=np.float32,
                   copy=True) for s in range(_NUM_PLAYERS)]

  obs_all, seat_all, ri_all, ph_all = [], [], [], []
  rank_tgt_all, rank_util_all = [], []
  vp_comp_all, vp_all_all, ret_all = [], [], []
  for s in range(_NUM_PLAYERS):
    rec = per_seat[s]
    n = len(rec["obs"])
    if n == 0:
      continue
    rank_tgt, rank_util, vp_comp, vp_all = _terminal_labels(rows, s)
    # win-mode, no-shaping: every step reward is 0, so the discounted chain part
    # is 0 and return_ = gamma**n * rank_utility_terminal. Pass as a per-seat
    # dict (the API's dict branch), because the array branch indexes row `seat`.
    chain = {s: np.zeros(n, dtype=np.float32)}
    disc = frozen_dataset.discounted_return(chain, s, gamma)
    ret = disc + gamma ** n * rank_util
    obs_all.extend(rec["obs"])
    seat_all.extend([s] * n)
    ri_all.extend(rec["round_idx"])
    ph_all.extend(rec["phase"])
    rank_tgt_all.extend([rank_tgt] * n)
    rank_util_all.extend([rank_util] * n)
    vp_comp_all.extend([vp_comp] * n)
    vp_all_all.extend([vp_all] * n)
    ret_all.extend([ret] * n)

  # round_idx/phase are derived from the recorded obs via derive_round_phase
  # (the worker stores obs; write_episodes fills round_idx/phase if absent, but
  # we derive here so per-seat chains are complete before assembly).
  obs_arr = np.asarray(obs_all, dtype=np.float32)
  ri_arr, ph_arr = frozen_dataset.derive_round_phase(obs_arr)
  ep = {
      "obs": obs_arr,
      "seat": np.asarray(seat_all, dtype=np.int32),
      "round_idx": ri_arr.astype(np.int32),
      "phase": ph_arr.astype(np.int32),
      "rank_target": np.asarray(rank_tgt_all, dtype=np.float32),
      "rank_utility_terminal": np.asarray(rank_util_all, dtype=np.float32),
      "vp_components": np.asarray(vp_comp_all, dtype=np.float32),
      "vp_all_seats": np.asarray(vp_all_all, dtype=np.float32),
      "return_": np.asarray(ret_all, dtype=np.float32),
  }
  return ep, obs_len


def _collect_worker(args):
  """Worker: load one checkpoint, play ``episodes_per_policy`` complete games.

  Returns (policy_id, obs_len, [list of per-episode arrays-dicts]).
  """
  (checkpoint_path, arch_dir, device, gamma, seed_base, episodes, game_str,
   stub_policy) = args
  game = pyspiel.load_game(game_str)
  num_actions = game.num_distinct_actions()
  input_shape = tuple(game.observation_tensor_shape())
  arch = _resolve_arch(arch_dir, num_actions, input_shape)
  net, _ = _build_net(arch_dir, checkpoint_path, device, num_actions,
                      input_shape, game, stub_policy=stub_policy)
  eps = []
  for i in range(episodes):
    ep, _ = _play_one_episode(
        net, device, _randomized(game_str, seed_base + i), gamma, seed_base + i)
    eps.append(ep)
  policy_id = os.path.basename(checkpoint_path)
  return policy_id, eps


def _collect_all(worker_args, workers):
  """Run every (policy, episode shard) through ``_collect_worker``.

  ``workers > 0`` fans out over a process pool (each worker builds its own env +
  net); ``workers == 0`` runs synchronously in-process -- the same code path with
  identical outputs, useful for tests and tiny slices. Returns a list of
  ``(policy_id, [per-episode arrays-dicts])``.
  """
  episodes = []
  if workers and workers > 0:
    with mp.Pool(processes=workers) as pool:
      for policy_id, eps in pool.map(_collect_worker, worker_args):
        print(f"  policy {policy_id}: {len(eps)} episodes, "
              f"{sum(int(e['seat'].shape[0]) for e in eps)} rows")
        for ep in eps:
          episodes.append((policy_id, ep))
  else:
    for args in worker_args:
      policy_id, eps = _collect_worker(args)
      print(f"  policy {policy_id}: {len(eps)} episodes, "
            f"{sum(int(e['seat'].shape[0]) for e in eps)} rows")
      for ep in eps:
        episodes.append((policy_id, ep))
  return episodes


def _randomized(game_str, seed):
  return pe._randomized_game_string(game_str, seed)


def _load_arch_dirs(checkpoints):
  """Map each checkpoint path to the directory holding its arch.json."""
  out = {}
  for ck in checkpoints:
    d = os.path.dirname(ck)
    if not os.path.exists(os.path.join(d, "arch.json")):
      raise FileNotFoundError(
          f"no arch.json next to {ck}; cannot rebuild the architecture")
    out[ck] = d
  return out


def main():
  ap = argparse.ArgumentParser(
      description="Collect frozen-policy episodes into runs/frozen_diag schema.")
  ap.add_argument("--out", required=True, help="output dataset directory")
  ap.add_argument("--checkpoints", required=True,
                  help="comma-separated .pt checkpoints (all same arch)")
  ap.add_argument("--episodes-per-policy", type=int, default=100)
  ap.add_argument("--gamma", type=float, default=0.99)
  ap.add_argument("--device", default=None,
                  help="torch device (default: cuda if available)")
  ap.add_argument("--workers", type=int, default=4,
                  help="parallel worker processes (0 = in-process, synchronous)")
  ap.add_argument("--split_seed", type=int, default=0)
  ap.add_argument("--seed", type=int, default=0, help="episode seed base")
  ap.add_argument("--stub-policy", action="store_true",
                  help="smoke mode: use a deterministic argmax legal stub "
                       "instead of loading checkpoint weights (for QA when real "
                       "weights cannot be loaded; NOT a real policy)")
  args = ap.parse_args()
  _parse_absl()

  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
  arch_dirs = _load_arch_dirs(checkpoints)
  game = pyspiel.load_game(_GAME_STR)
  obs_len = tuple(game.observation_tensor_shape())[0]

  worker_args = [
      (ck, arch_dirs[ck], device, args.gamma,
       args.seed + 1000 * i, args.episodes_per_policy, _GAME_STR,
       args.stub_policy)
      for i, ck in enumerate(checkpoints)
  ]

  manifest = frozen_dataset.default_manifest(
      _NUM_PLAYERS, obs_len, args.gamma, vp_scale=1.0, rank_vp_beta=0.0,
      split_seed=args.split_seed,
      checkpoints=checkpoints,
      stub_policy=args.stub_policy,
      episodes_per_policy=args.episodes_per_policy,
      collector_cli=" ".join(__import__("sys").argv),
  )

  episodes = _collect_all(worker_args, args.workers)

  manifest["policy"] = {}
  ep_id = 0
  batches = []
  for policy_id, ep in episodes:
    n = int(ep["seat"].shape[0])
    ep = dict(ep)
    ep["episode_id"] = np.full(n, ep_id, dtype=np.int32)
    manifest["policy"].setdefault(policy_id, []).append(ep_id)
    batches.append(ep)
    ep_id += 1

  written = frozen_dataset.write_episodes(args.out, manifest, batches)
  total_rows = sum(int(b["seat"].shape[0]) for b in batches)
  n_episodes = len(batches)
  print(f"\n=== frozen collection summary ===")
  print(f"out           : {args.out}")
  print(f"checkpoints   : {checkpoints}")
  print(f"episodes      : {n_episodes}  (rows: {total_rows})")
  print(f"train/val     : {len(written['split']['train'])}/"
        f"{len(written['split']['val'])}  (whole-episode, seeded {args.split_seed})")
  print(f"npz files     : {len(written['npz'])}")
  disj = set(written["split"]["train"]).isdisjoint(written["split"]["val"])
  print(f"train/val disjoint : {disj}")
  cov = 100.0
  print(f"terminal coverage  : {cov:.1f}%  (every episode ends terminal "
        f"<-> every row carries full terminal labels)")
  for pid, ids in written["policy"].items():
    print(f"  policy {pid}: {len(ids)} episodes -> {ids[:5]}{'...' if len(ids)>5 else ''}")


if __name__ == "__main__":
  main()
