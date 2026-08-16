# Eclipse observation V2 audit

> **2026-08: the typed/spatial pointer head that consumed the "pointer keys"
> below was REMOVED as a null result.** There is no `TypedPointerActorHead`,
> `SpatialFactoredActorHead`, `logits_for`, `forward_with_context`, or
> `PointerContext` anymore — the actor is always a `FactoredActorHead`. The
> pointer-keys/pointers language in this doc is historical. The cell/unit/slot
> rows are still decoded and embedded in the encoder (`SpatialEclipseEncoder`),
> but no head gathers them as per-action pointer terms. If you are porting these
> design notes, ignore the "action consumer ... pointers" column and the pointer
> sections; localize details to the removed head in `docs/eclipse_rl_todo.md`
> (Section 7).

`observation.h` is the tensor authority and `obs_layout.py` mirrors it. V2 is
checkpoint-incompatible (`37,596` floats) and appends keyed public entities to
the V1 blocks. `obs_layout._self_check` pins the total and every sub-block
offset, so a C++ change fails loudly in Python instead of mis-reshaping.

| Public state | Encoding | Action consumer |
| --- | --- | --- |
| Units | 128 registry rows: owner (8-wide rel-seat one-hot), type, cell/coords, damage, arrival RECENCY, movement and die-target flags; six resolved routes | move and combat target pointers |
| Planet slots | 225 x 8 exact rows: valid, type, occupied, orbital | colony and population-combat pointers |
| Players | V1 relative blocks plus V2 absolute seat key and independent military/grid/nano bitmaps | diplomacy pointer |
| Galaxy | V1 semantic channels plus sector-definition id and rotation | cell pointers |
| Combat | ordered battle participant/arrival, destruction/killer, firing queue, dice, retreats, population target cell | combat decisions |
| Discoveries | current revealed identity and a 30-kind public ledger | reward-versus-VP decision and history |
| Tech bag | exact 40-kind histogram | research evaluation |

Deliberately hidden: face-down `Sector::discovery_tile`, discovery-bag
composition/order, and RNG state. `RevealDiscovery` is the only reveal path;
it updates the ledger once before Explore or Combat offers the decision.

## Written is not read

Writing a field into the tensor does not mean the network sees it. As first
committed, **1,835 of the 12,882 V2 floats (14.2%) were written every step and
read by nothing** — the encoder never referenced `V2_GLOBAL_START` or
`V2_CELLS_START` at all, and touched only 6 of 732 seat floats and 1 of 553
combat floats. That included the tech-bag histogram, the discovery ledger and
the entire keyed combat queue: precisely the features this block exists to
expose. All of it is now consumed (`SpatialEclipseEncoder._encode_impl`).

If you add a V2 field, grep the encoder for its offset constant before claiming
the agent can use it.

## Scalars that are really categoricals

Several V2 fields are normalised integer ids, not magnitudes: `sector_id`
(396-way), `rotation` (6-way), planet `type` (8-way). Fed straight into a
`Linear` they force the net to learn a lookup along one dimension — strictly
worse than V1, which one-hots the same planet types.

They stay narrow in the tensor and are **decoded back to integers and embedded
in Python** (`sector_embed`, `rotation_embed`, `planet_type_embed`). That is the
OpenAI Five / AlphaStar treatment of a categorical, and it keeps the tensor at
37,596 instead of widening it for one-hots.

Cell ids (`U_CELL`, unit routes, `pop_attack` cell) are different: those are
**pointer keys**, decoded to indices rather than learned as features. Note the
two normalisations are not the same — `U_CELL` is `cell / 224`, while route and
pop-attack cells are `(cell + 1) / 225` so that 0 can mean "none". Decode each
with the divisor its writer used.

## Ranges

`Frac` clamps to [-1, 1], so a bad divisor saturates rather than overflowing —
which hides itself. `arrival_order` was divided by 400 while the counter never
exceeded 12 in a full game, spending 97% of the range on values that never
occur and leaving every unit's arrival indistinguishable. It is now relative to
`next_arrival_order`, which is scale-free and is what combat's arrival tiebreak
actually needs.

## The tensor is 97.5% zeros, and the writer used to walk capacity anyway

Measured over random 4p games (sampled every 25 moves, warped and unwarped):

| | occupied | capacity | share |
|---|---|---|---|
| nonzero floats | 877–1,250 | 37,596 | **2.5%** |
| galaxy cells present | 7–11 | 225 | 3–5% |
| valid planet-slot rows | 24–38 | 1,800 | ~2% |
| valid unit rows | 6–10 | 128 | ~7% |

Occupancy has a hard ceiling, not just an empirical one: `sectors.h` holds the
entire tile supply — 11 INNER + 14 MIDDLE + 23 OUTER + 4 STARTING (4p) + 1
CENTER = **53 placeable tiles**. The 15×15 grid can never exceed ~24% occupancy
*by construction*, so any fixed-capacity cell list only needs 64 rows.

Until 2026-08-13 the writer walked all 225 cells and all 1,800 slot rows,
writing zeros into a span that had *already* been zeroed — twice, since
`eclipse.cc` and `observation.cpp` both filled it. `observation_tensor_into` cost
13.0 µs, of which only ~3.5 µs was the unavoidable fill and the rest was the
capacity scan producing ~950 useful values.

Three skips (empty cells write no unit channels; unplaced cells write no slot
rows; slot rows above the orbital row are not walked) plus dropping the duplicate
fill took it to **5.0 µs, bitwise identical** across 4 game configs × 6 seeds ×
every seat. If you add a block here, write it occupancy-first: the padding
convention is already "all-zero row, validity bit at offset 0", so a skip is
free and a capacity walk is not.

## Shrinking the tensor: what it would and would not buy

A recurring plan is to entity-list the two dense capacity blocks (galaxy
19,800 floats, planet slots 7,200). Two corrections before anyone builds it.

**You cannot convolve a list.** The proposal in the old planning notes was that
entity-listing the galaxy shrinks the conv tower's input "from 225 cells to ~96".
It does not: `obs_layout.galaxy_view` requires the dense 15×15, and the encoder
addresses `h_cells` by cell id. The implementable form is AlphaStar's *scatter
connection* — emit ≤64 rows of `(cell_id, 88 channels)`, scatter them into a
zeroed `(B, 88, 15, 15)` in Python, then run the existing conv tower unchanged.
Conv cost is therefore **unchanged** (the value of a pointer head was irrelevant
here — it has been removed, Item 7 in `docs/eclipse_rl_todo.md`).

**The slot block buys no learn memory either.** The raw slot rows are a *view*
into the observation (zero allocation) and only the rows the action targets need
an embedding. Scattering a compacted slot list back to `(B, 1800, 4)` would
*create* a 59 MB device allocation at minibatch 4,096 where today there is none —
so if slots are compacted, give the consumer a per-cell row-base map instead of
scattering.

What the shrink genuinely buys, after the writer fix took the env-step half:
rollout **buffer bytes**, **H2D bytes** per act step, and the fp32 minibatch
materialization in the learn input term (`ppo.py:365`/`:1740`, which upcasts
regardless of `--obs_buffer_dtype`). Roughly 37,596 → ~17,000 floats. On a 12 GB
card that is the difference between 256–512 envs and about double that.

Cost is unchanged and still the blocker: it invalidates every checkpoint, and
there is no observation-version parameter — `ObservationTensorShape`
(`eclipse.cc`) reads no game parameter, and `obs_layout` is module-level
constants, so a runtime switch means restructuring `validate()` and every
`obs_layout.X_START` reference.

## Planet slots: 8 wide, 16 offered

The engine offers `COMBAT_POP_TARGET_0..15` (it scans a `uint16`
`occupied_slots_mask`), but the V2 table is 8 slots per cell and the Orbital
occupies row `printed`. Widest real tile has 6 slots, so slots >= 8 are
currently unreachable — but `cell*8 + slot` would silently alias into the *next*
cell's rows. The writer asserts `printed < kPlanetSlotsPerCell`; the 
slot>=8 rejection that used to live in the (now-removed) pointer head is gone.
Do not remove the remaining writer guard on the grounds that the case cannot
happen.
