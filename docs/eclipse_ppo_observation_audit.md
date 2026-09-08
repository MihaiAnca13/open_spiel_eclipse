# Eclipse PPO observation and encoder audit

Date: 2026-09-06. Scope: game state → observation tensor → spatial encoder → actor → PPO legal-action masking. This records an inspection and diagnostic checks, not implemented fixes or demonstrated win-rate improvements.

## Summary

The current policy loses useful information at three different stages:

1. The observation omits or clips decision-relevant state.
2. The spatial encoder ignores entire observation blocks and discards unit registry identity.
3. The factored actor cannot express certain state-dependent interactions between action arguments.

Increasing model width alone cannot fix these structural losses. The observation also leaks some information hidden by the board-game rules, so connecting every existing field without a visibility audit would be incorrect.

Eclipse's strategic tradeoff is spending actions and influence discs to grow production, territory, technology, and military power before the eight-round deadline. Expansion increases upkeep; population changes production nonlinearly; upgrades affect every ship of a class; movement depends on wormholes, pinning, and warped connections. Combat needs exact damage, initiative, retreat timing, and reputation choices.

## Findings ledger

| ID | Area | Finding | Evidence / status |
| --- | --- | --- | --- |
| OBS-01 | Encoder inputs | Original global block was never consumed | Fixed: public global block now feeds the tail MLP |
| OBS-02 | Unit identity | Attention and pooling discard registry row identity needed by action IDs | Swapping two rows changes the encoder output by only approximately 1.8e-7 |
| OBS-03 | Movement | Six destination routes per unit are written but never read | Runtime perturbation: exactly zero encoder-output change |
| OBS-04 | Planet slots | Exact slot type/occupancy rows are written but never read | Runtime perturbation: exactly zero encoder-output change |
| OBS-05 | Blueprints | Part counts and occupied slots omitted the mapping from slot to part | Fixed: each blueprint slot now carries its part ID |
| OBS-06 | Resources | Balances above 40 were clipped | Fixed: balances use the `uint8_t` range |
| OBS-07 | Action head | Additive factors lack state-dependent interactions between arguments | Follows directly from the actor's scoring equation |
| OBS-08 | Combat timing | Retreat start rounds were omitted | Fixed: keyed retreat records include the start round |
| OBS-09 | Combat encoding | Reputation draw target count was encoded as a player identity | Fixed: normalized tile count |
| OBS-10 | Empty units | Masked maximum returned -1e9 when no units were valid | Fixed: empty max pooling is neutral zero |
| VIS-01 | Reputation privacy | Opponents' face-down reputation values and exact reputation VP are exposed | Fixed: live observations hide retained values, private draws, derived scores, and the bag histogram; terminal scoring reveals them |
| VIS-02 | Sector privacy | Exact randomly selected outer-sector supply is exposed | Fixed: observations retain its public count but zero the secret bitmask |
| RULE-01 | Game model | Discarded sectors are returned when their stack is depleted | Fixed: per-ring discard piles preserve the setup-limited tile pool |

These statuses distinguish observed numerical failures, structural code findings, and issues still needing targeted gameplay reproduction. The ledger is not proof that every other state field is sufficient.

## Encoder and action losses

### Ignored blocks: OBS-01, OBS-03, OBS-04

[`SpatialEclipseEncoder._encode_impl`](../open_spiel/python/examples/ppo_eclipse.py) does not read these ranges from [`obs_layout.py`](../open_spiel/python/eclipse/obs_layout.py):

| Block | Entries |
| --- | ---: |
| Original global block | 146 |
| V2 unit routes | 768 |
| V2 planet slots | 7,200 |
| Total ignored | 8,114 of 37,596 (21.6%) |

The percentage includes padding and reserved entries; it is not a percentage of meaningful game information. Independently replacing each block with random values left the encoder output exactly unchanged in an eager CPU check on an opening observation, using width 16 and depth 1.

The original global block carries the round, phase, NPC difficulty and combat profiles, sector supply, bag sizes, available minor species, and other context. Some fields can be partly inferred elsewhere. The actual round cannot safely be replaced by board size or resource levels. Likewise, learned generic NPC owner embeddings do not expose the episode's selected NPC combat profiles.

Planet aggregates and sector identities provide partial alternatives to exact slot rows, and ordinary routes can be derived from coordinates. However, the network has no direct consumption of the exact rows already supplied for these decisions.

### Registry identity: OBS-02

Unit rows use shared processing, self-attention without registry-position embeddings, and masked mean/max pooling. Their row order is therefore discarded, up to numerical noise. Unit coordinates and ownership survive as features of the set; the mapping from registry index to unit does not.

Actions such as `MOVE_UNIT_7_*` and `COMBAT_TARGET_UNIT_7` require that mapping. A global fleet summary can say that a damaged cruiser exists without telling the actor that it is unit 7. The factored actor receives only the fused vector, with no unit-feature lookup by action target.

Unit coordinates and destination routes already exist in V2. Adding another engine-side unit-location table would duplicate data; the missing link is its network consumption.

### Additive action factors: OBS-07

[`FactoredActorHead`](../open_spiel/python/examples/ppo_eclipse.py) sums embeddings selected by [`action_factors.py`](../open_spiel/python/eclipse/action_factors.py), then takes a dot product with the fused state vector. Movement scores have the form:

```text
logit(s, u, d) = family(s) + unit(s, u) + direction(s, d) + bias(u, d)
```

For two units and two directions, the difference of direction preferences is therefore independent of state:

```text
[logit(s, u, E) - logit(s, u, NE)]
  - [logit(s, v, E) - logit(s, v, NE)]
  = bias(u, E) - bias(u, NE) - bias(v, E) + bias(v, NE)
```

When all four choices are legal, masking does not remove this restriction. The policy cannot freely adapt each unit's direction preference to its own changing surroundings. Similar restrictions affect build-type × cell and upgrade-slot × part combinations. More encoder width does not change this algebra.

The previous pointer experiment's null result is evidence about that experiment, not proof that these structural losses are harmless.

## Observation writer and edge cases

### Blueprint slot contents: OBS-05

[`WriteBlueprint`](../open_spiel/games/eclipse/observation.cpp) now emits each slot's compact part ID in addition to derived statistics, a histogram, and occupancy flags. The following collision was the pre-fix evidence.

Controlled engine checks using deserialized states found:

- Swapping an Ion Cannon and Hull on a cruiser leaves observations identical.
- `UPGRADE_CRUISER_SLOT0_REMOVE` can be legal in both states.
- Executing that common action produces different successor observations.

[`can_upgrade` and `execute_upgrade`](../open_spiel/games/eclipse/systems/actions/upgrade.cpp) read the actual part in the addressed slot. That information belongs in the observation. The legal-action sets differed in the checked pair, so this is an observation collision, not a claim that the complete observation-plus-mask inputs were identical. Masks still do not directly provide the missing part identities to the encoder.

### Resource clipping: OBS-06

The writer's `Frac` clamps to [-1, 1]. Gold, science, and materials now use their `uint8_t` maximum (255); the following collision was the pre-fix evidence.

In constructed state pairs, independently changing each resource from 70 to 90 left the complete observation identical. The extra gold cash-flow feature also saturated in these examples. These checks demonstrate aliasing; they do not measure how often those balances occur during training.

Use a representation that preserves the supported range. Audit other clamped fields against actual engine bounds rather than assuming every divisor is a valid maximum.

### Combat fields and empty-set handling: OBS-08–OBS-10

- `CombatState::retreating_rounds` now appears in each keyed retreat record, using the combat-round scale.
- `CombatState::rep_draw_target` now appears as a normalized tile count, not a seat code.
- Empty unit sets now use a zero max-pool result, matching their zero mean.

## Visibility and rule fidelity

The reference is the supplied [Second Dawn rulebook](../07-eclipse-second-dawn-for-the-galaxy-rulebook.pdf).

### Reputation: VIS-01

The rulebook places retained reputation tiles face down. The writer previously exposed every player's exact reputation values and exact reputation VP. Exact total VP was another disclosure channel for that hidden contribution.

Live observations now expose retained reputation values and current draws only to their owner, and hide opponents' reputation VP from both the breakdown and derived total fields. Hidden occupied slots use an all-zero value one-hot, distinct from the explicit `NONE` value. The reputation-bag size remains public, while its per-value histogram is withheld until terminal scoring. Terminal observations reveal the exact values so existing final score-breakdown targets remain valid.

### Outer-sector supply: VIS-02

[`setup.cpp`](../open_spiel/games/eclipse/systems/setup.cpp) randomly chooses the outer-sector subset. Public placements cannot identify which unseen tiles were initially included. The observation previously wrote the exact remaining bitmask.

The raw tensor now retains the public remaining count but leaves the outer-sector bitmask zero. Placed tiles remain available through the galaxy representation; inner- and middle-sector masks remain public. This preserves the tensor layout while preventing the secret setup subset from reaching a future global encoder.

### Discoveries and discards

Face-down discovery identities are deliberately withheld; revealed discoveries have a ledger. This is the correct distinction between hidden outcomes and remembered public information.

[`explore.cpp`](../open_spiel/games/eclipse/systems/actions/explore.cpp) now puts discarded sectors, including the Descendants of Draco's unchosen tile, in the corresponding faceup discard pile and reuses that pile when the live stack empties. Sector III refills only reuse tiles selected for the player-count-limited setup pool; excluded tiles cannot enter play. Because chance outcomes sample uniformly from a bitmask, refilling needs no hidden draw order. The PPO observation does not expose discard-pile contents.

## Research and implementation comparisons

- [OpenAI Five, Figures 17–18](https://cdn.openai.com/dota-2.pdf): pools entity features for global processing while retaining unit embeddings for attention-based target selection. The applicable lesson is to keep action-addressable entities alongside pooled summaries.
- [AlphaStar paper](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf): combines scalar, entity, and spatial streams with structured action heads and entity targeting. This supports explicit global inputs and preserving entity connections; it does not establish that Eclipse needs the entire architecture.
- [DeepMind's AlphaStar implementation](https://github.com/google-deepmind/alphastar/blob/main/alphastar/architectures/standard/heads.py): reference for separate structured action-head components.
- [Meta's Diplomacy implementation](https://github.com/facebookresearch/diplomacy_cicero/blob/main/fairdiplomacy/models/base_strategy_model/base_strategy_model.py#L1274): its optional relational output features gather candidate orders' source and destination board features. This is a relevant pattern for Eclipse movement scoring.

These are design references, not evidence of an Eclipse performance gain. Memory may help retain legitimate public history, but cannot repair information discarded before the policy sees it or justify access to hidden future outcomes.

## Recommended order and acceptance checks

1. Preserve exact unit/slot identity through action selection, using existing position/route data and allowing state-dependent interactions between action arguments.
2. Compare width and topology handling only after those correctness checks pass.

At inspection time, `runs/roster/arch.json` specified spatial encoding, width 16, depth 1, and a factored actor. This is checkpoint metadata, not confirmation of the settings of any currently running process. Compare wider models under controlled training and wall-clock budgets; do not assume width is the primary defect.

Acceptance checks should cover:

- Identical board/economy with different rounds retains distinct time context.
- Supported resource balances remain distinguishable above 40.
- Blueprint part swaps change slot representations; replacement consequences match the addressed part.
- Reindexing units and corresponding actions permutes target scores consistently.
- Different units can reverse their direction preferences independently as destinations change.
- Colony/population choices access the correct slot; warped moves access the correct destination.
- Retreat timing and reputation draw counts match their actual semantics.
- Empty entity sets produce ordinary finite representations.
- Changing opponents' hidden tiles or secret outer-sector selection does not leak into policy observations or derived score features.
- Sparse training and dense evaluation score the same policy.

Then evaluate seeded matches against fixed opponents and the existing ladder. Numerical sensitivity tests establish information access, not that the trained policy uses the information well.

## Documentation and validation limits

[`eclipse_observation_v2.md`](eclipse_observation_v2.md) claims all V2 information is consumed and describes a planet-type embedding absent from the current encoder. [`eclipse_rl_todo.md`](eclipse_rl_todo.md) contains historical pointer and unit-location plans. These notes need reconciliation with current code before implementation; they were not edited during this audit.

Checks were small CPU diagnostics and constructed engine-state comparisons, not a full training run or exhaustive reachability audit. During inspection the shared build changed: the Python extension temporarily disappeared, then returned with bindings that allowed engine checks but failed trainer import because `pyspiel.PlayerId` was unavailable. Earlier encoder checks completed before that change. No build-system changes were made as part of this audit.

## Bonus: shared part embeddings

Store a part ID in each blueprint slot, then let the network look up a shared vector for that ID. Keep the vector attached to its player, ship class, and slot. Reusing the same lookup for an upgrade's proposed part connects installed parts with candidate actions.

| Variation | Idea / tradeoff |
| --- | --- |
| Fixed perpendicular vectors | One-hot encoding is the simplest example. For 43 nonempty parts, exact orthogonality requires at least 43 dimensions; clear identities, but no built-in similarity. |
| Small learned vectors | Start with an 8–16-dimensional lookup and an explicit empty-slot ID. Training can learn useful similarities between parts; orthogonality is unnecessary. |
| Learned vectors plus part statistics | Add known energy, hull, movement, and weapon properties. Gives the network the effects directly instead of requiring it to learn every rule from IDs. |
| Action-conditioned attention | Let an upgrade candidate consult the addressed slot and ship's other parts. Preserves replacement context, but adds machinery; test direct slot lookup and a small scorer first. |

Suggested starting point: per-slot IDs, a shared learned lookup, and direct access to the current and proposed parts when scoring upgrades. IDs can reduce observation storage compared with repeated one-hot vectors; expanding them inside the model still costs computation, and adding missing slot IDs alone increases today's tensor size. Embeddings may improve sharing, but do not repair discarded slot identity, resource clipping, or ignored global inputs by themselves.
