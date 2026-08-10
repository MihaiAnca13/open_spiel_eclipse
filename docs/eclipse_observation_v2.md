# Eclipse observation V2 audit

`observation.h` is the tensor authority and `obs_layout.py` mirrors it. V2 is
checkpoint-incompatible (`37,596` floats) and appends keyed public entities to
the V1 blocks.

| Public state | Encoding | Action consumer |
| --- | --- | --- |
| Units | 128 registry rows: owner, type, cell/coords, damage, arrival, movement and die-target flags; six resolved routes | move and combat target pointers |
| Planet slots | 225 x 8 exact rows: valid, type, occupied, orbital | colony and population-combat pointers |
| Players | V1 relative blocks plus V2 absolute seat key and independent military/grid/nano bitmaps | diplomacy pointer |
| Galaxy | V1 semantic channels plus sector-definition id and rotation | cell pointers |
| Combat | ordered battle participant/arrival, destruction/killer, firing queue, dice, retreats, population target cell | combat decisions |
| Discoveries | current revealed identity and a 30-kind public ledger | reward-versus-VP decision and history |
| Tech bag | exact 40-kind histogram | research evaluation |

Deliberately hidden: face-down `Sector::discovery_tile`, discovery-bag
composition/order, and RNG state. `RevealDiscovery` is the only reveal path;
it updates the ledger once before Explore or Combat offers the decision.
