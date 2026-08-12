# Eclipse observation V2 audit

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
expose. All of it is now consumed (`SpatialEclipseEncoder._encode_context`).

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

## Planet slots: 8 wide, 16 offered

The engine offers `COMBAT_POP_TARGET_0..15` (it scans a `uint16`
`occupied_slots_mask`), but the V2 table is 8 slots per cell and the Orbital
occupies row `printed`. Widest real tile has 6 slots, so slots >= 8 are
currently unreachable — but `cell*8 + slot` would silently alias into the *next*
cell's rows, so `TypedPointerActorHead` rejects `slot >= 8` explicitly and the
writer asserts `printed < kPlanetSlotsPerCell`. Do not remove either guard on
the grounds that the case cannot happen.
