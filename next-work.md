# Next Work

No training is running. Do not resume `main_v1` or start another multi-hour run
until the offline critic gate below passes.

## Decisions

1. Fix the critic before changing the policy architecture or league.
2. Predict the nine VP sources explicitly. The current critic discards useful
   cell detail and its scalar value is mathematically bounded to `[-0.5, 1]`
   even when PPO targets are outside that interval.
3. Full-state search may deliberately use hidden C++ state. Label it **oracle
   search** so its strength is not mistaken for fair-information play.
4. No more four-hour discovery runs. Every risky change must first pass a unit
   test, a frozen-data test, a short smoke test, and a written stop rule.

## Groundwork completed in this pass

- Mixed league lineups now rotate main across seats; a distribution test pins
  the fix.
- Self-play GAE no longer bootstraps one seat from another seat's boundary
  value. Seats without a valid next observation truncate the boundary row with
  zero advantage.
- Rank classification now uses soft tie targets. An all-player tie targets
  `[0.25, 0.25, 0.25, 0.25]`, not “first” for everyone.
- Nine-head auxiliary loss is averaged across present heads instead of summed;
  adding breakdown heads no longer multiplies `aux_coef` by nine.
- Terminal VP labels are now real terminal labels:
  - C++ snapshots all nine components before an eliminated player's pieces are
    removed;
  - snapshots survive state serialization;
  - async workers copy only `4 x 9` floats before reset;
  - PPO refuses breakdown training when the exact payload is missing;
  - a test denormalizes the components and checks that they sum to terminal VP.
- `--nn_norm` now reaches every spatial branch MLP. The conv tower already uses
  GroupNorm; future attention will use pre-LayerNorm.
- Evaluation now scores only legal actions and exactly matches the old dense
  11,117-action argmax in tests.
- Evaluation can run one fixed seeded episode per environment in input order.
  `ffa_metagame.py` records the complete ordered `4^4` payoff tensor and runs
  asymmetric four-population AlphaRank with bootstrap intervals.
- Removed avoidable hot-path work: dynamic step classes, object-array policy
  gathering, repeated policy `unique`, and per-minibatch metric GPU syncs.

These are correctness/mechanical changes, not evidence that a new policy is
stronger. No training was launched.

## Stage 1 — critic proof before PPO

### 1. Build one frozen diagnostic dataset

Collect complete episodes from frozen existing policies and store:

- observation, viewing/acting seat, episode id, round/phase;
- exact terminal rank distribution;
- exact nine VP components for every seat, stored in viewer-relative order
  (the first critic version predicts the viewer's nine; retaining all seats
  avoids recollecting data if opponent-score prediction is later justified);
- undiscounted terminal rank utility;
- the exact discounted return produced by the intended shaping and gamma.

Split by whole episode, never by row. Save policy hashes, game configuration,
setup seeds, chance seeds, target scales, and code revision beside the data.

### 2. Replace the critic readout

Keep the actor unchanged. Build one value feature from:

- the existing fused state vector; plus
- one state-conditioned cross-attention query over all 225 existing `h_cells`.

Use learned cell-position embeddings and pre-LayerNorm. Do **not** mask to
`C_PRESENT`: empty locations are future exploration/action targets. This is
one-query cross-attention (`O(225)`), not a 225-token self-attention stack.

Attach three independent outputs to that feature:

1. an unbounded scalar value used by PPO/GAE;
2. four rank logits trained with the corrected soft tie target;
3. nine unbounded VP-component predictions.

The rank probabilities no longer define the PPO scalar. The nine components
use fixed mean/std statistics computed once from the frozen training split,
with a documented variance floor; never normalize them per minibatch. Convert
predictions back to VP units for reporting and sum them to report predicted
total VP. Do not add a redundant total-VP head initially.

Use the shared encoder rather than duplicating it with `separate_critic`.
Measure real gradient norms for policy, scalar value, rank CE, and aggregate VP
loss. Choose `aux_coef` so VP supervision is material but does not dominate the
shared trunk; the current `aux_share` loss-magnitude proxy is not a gradient
measurement.

### 3. Offline acceptance gate

No PPO run until all pass:

- tensor shapes, seat rotation, terminal component sums, and checkpoint
  metadata are covered by tests;
- a 512-row tiny set reaches normalized MSE `< 1e-3` for every non-constant VP
  head and the scalar head;
- held-out scalar MSE is below the current bounded rank-expectation critic;
- every sufficiently variable VP component beats its train-split-mean baseline
  on held-out episodes;
- predicted scalar calibration, component MAE/R², rank CE/Brier, target
  percentiles, and per-objective gradient norms are printed in one report;
- a parameter-matched MLP readout is evaluated on the same frozen split. Keep
  cross-attention only if it improves held-out value/component prediction.

This offline comparison replaces a pair of four-hour architecture runs.

## Stage 2 — evaluation baseline before new training

After plan approval, evaluate four existing policies in one empirical game:

- T1's best snapshot;
- `main_v1` early snapshot;
- `main_v1` update-250 peak;
- `main_v1` latest snapshot.

Run all 256 ordered seat profiles with 32 matched scenarios (8,192 games).
Increase to 64 only if the relevant AlphaRank/support intervals remain
ambiguous. Preserve raw utilities so 1-vs-3, 2-vs-2, seat effects, cycles, and
dominance can be derived without replaying games.

Do not redesign the league unless this full FFA matrix actually shows cycling.
The latest policies observed so far beat no sampled predecessors, which is
more consistent with regression than a rock-paper-scissors cycle.

## Stage 3 — short training ladder

Only after Stages 1–2 pass:

1. **One update:** finite losses, exact target coverage, checkpoint save/load,
   and no target/gradient invariant failure.
2. **Ten updates / minutes:** scalar EV and all component errors move in the
   right direction; no single auxiliary objective dominates trunk gradients.
3. **Thirty-minute gate:** fixed frozen-opponent panel plus early/peak/current
   snapshots; stop on clear regression.
4. **Two-hour gate:** full FFA evaluation. Continue longer only with a positive
   lower confidence bound against the pre-registered baseline.

Write the exact command, seeds, snapshot panel, wall-clock limit, and pass/fail
threshold beside the run before launch.

## Stage 4 — oracle search

Implement a root-only Monte-Carlo rollout oracle before MCTS:

1. for each legal root action and `K` trials, clone the full C++ state;
2. apply the action, sample chance nodes, and finish with the frozen actor for
   every seat;
3. score the root seat with tie-aware rank utility at `rank_vp_beta=0`;
4. choose the action with the highest mean utility.

Hidden bag/deck state is intentionally visible. Reuse `observation_tensor_into`
and sparse legal scoring; neural inference, not the simulator, is the expected
bottleneck.

Gate it in this order: one-state profiler at `K=1/4`, 32 serialized decisions,
one searched decision per game with seat rotation, then 1/8 of decisions. Build
custom rank-consistent max-n/PUCT only if `K=4` improves actual games with a
positive lower confidence bound. Do not use OpenSpiel `MCTSBot` unchanged: its
terminal backups are raw VP while a PPO leaf would be rank utility.

## Parked

- A masked mean only fixes scale dilution; it does not recover spatial
  relationships. It is not the architecture test.
- A full graph encoder is later work, only if the small critic readout proves
  cell structure matters. It needs six axial edges, warp edges, positions, and
  edge labels.
- No temporal Transformer: the current state is sufficient for the deployed
  policy, and no history deficit has been demonstrated.
- No difference rewards: the rank objective is constant-sum, so the proposed
  global difference reward collapses.
- No MuZero dynamics model while the exact C++ simulator is available.
- GAE vectorization and last-decision copy removal are throughput follow-ups,
  not blockers for critic correctness.

## Reference papers

Verified local copies and checksums are in [`papers/README.md`](papers/README.md):

- AlphaStar — privileged/centralized value information and entity attention;
- OpenAI Five — PPO/GAE/self-play can scale without MCTS;
- Hide-and-seek — entity attention and a training-only privileged value input;
- Diplodocus (`arXiv:2210.05492`) — one-ply multiplayer planning needs vector
  payoffs and prunes joint actions because enumeration explodes.
