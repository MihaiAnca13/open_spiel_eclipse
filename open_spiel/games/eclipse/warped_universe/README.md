The 3 layout files
Identical grid (127 cells, grid_radius: 8, hex_radius: 28.0), only warp count differs:
File	Warp cells	Components (each = canonical 18 hex rotation)
layout_tool_map_3p.json	54	3
layout_tool_map_4p.json	36	2
layout_tool_map_5p.json	18	1
Per-cell schema
{ "q": int, "r": int, "kind": "warp" | "sector2" | "sector3" | "inner" | "starting_player" | "gcs" }
q, r are axial coordinates (Red Blob convention, pointy-top hex).
Cell kinds:
- warp — warp-portal hex (the only ones that participate in links)
- sector2 — outer-ring sector slot
- sector3 — interior sector slot
- inner — empty interior (no sector placed)
- starting_player — player's home sector (one per player in scenario)
- gcs — Galactic Center Section (the single center hex)
The file does not say which sector tile goes where — it only marks slot positions. Tile assignment is separate (sector catalog, deterministic selection).
Pairings file (warp_pairings_v1.json)
- touchpoints[] (108 entries) — every hex-edge of the 18-hex canonical warp region. id = tp_<q>_<r>_<edge>, warp_q/warp_r = the hex, edge_index = 0..5, perimeter_index = flat 0..107 ordering.
- enabled_ids[] (24 entries) — which 24 of the 108 touchpoints actually form the 12 portal pairs.
- assignments[] — groups ids by integer pair. Two ids with the same pair are portals that connect.
What the engine does with them
1. Build a Map<axial, cell> from the layout's cells[].
2. Find connected components of kind=="warp" cells (4-neighbor adjacency via the 6 axial directions).
3. For each component, find the (rotation, translation) that maps the canonical 18-hex warp region (derived from the unique (q, r) of enabled touchpoints) onto that component. Selection rule: minimize mean distance from world origin after mapping. See warp_universe_links.nim:171-238.
4. For each assignments pair (a, b) where both are enabled:
- Map a and b through the transform.
- edge_rotated = (edge_index + rotation) mod 6 → which neighbor hex the warp exit points to.
- Look up the layout cell at that neighbor: if it's a sector* or starting_player, the warp links that sector's matching edge (the edge that points back to the warp) to the other end's sector/edge.
5. Result: map<"sectorId:edgeIdx", WarpExitRef> — the runtime edge-connection table.
Assumptions baked in (your C++ port must replicate)
Hex math
- Axial coords (q, r). Cube s = -q - r.
- Pointy-top: x = sqrt(3) * (q + r/2), y = 1.5 * r, hex width = sqrt(3) * hex_radius, hex height = 2 * hex_radius.
- Edge directions (the 6 neighbors): [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)] indexed 0..5.
- edge_index rotates with the transform: (original + rot) mod 6.
Player starting positions (6 fixed slots around the board)
0:(0,-2)  1:(2,-2)  2:(2,0)
3:(0,2)   4:(-2,2) 5:(-2,0)
Use 2, 3, 4, or 5 of them (3p→positions 0,2,4 / 1,3,5; 4p→0,1,3,4; 5p→0..4) — your C++ project decides the actual position→sector mapping.
The 18-hex canonical shape is fixed — it's hard-coded in the pairings file (18 unique (warp_q, warp_r) among enabled touchpoints). The layout files only place rotated/reflected copies of it. Layouts are validated against this shape at load time by validateWarpLayout in the loader — if a layout's connected component isn't a 6-fold rotation of the 18-hex shape, the engine can't match it and that component is skipped.
One direction only: each enabled touchpoint appears in exactly one pair (validated). No chaining. No portals between two warp hexes inside the same component — portals always exit to a sector hex outside the warp region.
Minimal C++ recipe
struct Axial  { int q, r; };
struct Cell   { Axial a; enum Kind { Warp, Sector2, Sector3, Inner, Start, Gcs } kind; };
struct Touch  { Axial warp; int edge; bool enabled; int pair; };
struct Link   { std::string fromSector; int fromEdge;
                std::string toSector;   int toEdge; };

const Axial DIR[6] = {{1,0},{1,-1},{0,-1},{-1,0},{-1,1},{0,1}};

std::pair<double,double> axialToWorld(Axial c, double R) {
    return { sqrt(3.0)*(c.q + c.r/2.0), 1.5*c.r };
}

// 1. Parse layout → unordered_map<Axial,Cell>
// 2. Parse pairings → touch[]; canonical_18 = unique(warp) for enabled touchpoints
// 3. Find connected components of Warp cells (BFS over axial neighbors)
// 4. For each component: try rot=0..5, anchor=any cell, check
//    (rot(canonical_i) + anchor) ⊆ component for all i → that's the transform.
// 5. For each pair {a,b} of enabled touchpoints:
//      warpA = rot(touch[a].warp, rot) + anchor;  edgeA = (touch[a].edge + rot) % 6;
//      neighborA = warpA + DIR[edgeA];
//      cellA = layout[neighborA];  if sector, sectorA = its tile id
//      do same for b
//      link[sectorA][edgeA_opposite] = sectorB, edgeB_opposite
//      (opposite = (edge+3)%6 — the edge the sector uses to point at the warp)
// 6. Result: edge_map[Axial(sectorId, edgeIdx)] → (sectorId, edgeIdx)
Things you can drop
- The gcs, inner, sector2, sector3 annotations — useful only for placing actual sector tiles from your catalog; the warp logic itself only cares about warp cells and what sits at the neighbor coord.
- perimeter_index in pairings — purely informational, the engine never reads it.
- grid_radius / hex_radius / version — bookkeeping for the editor; not used by the link engine.
Things you cannot drop
- The 18-hex canonical shape. It's the contract between layout files and the pairings file. If you author new layouts, each connected warp region must be a rigid rotation of these 18 hexes.
- The enabled-touchpoint subset (24 of 108). Disable a touchpoint → its pair goes silent (no warp from that edge).
- The 6-direction axial convention and pointy-top math. The exit-edge rotation only works if you use this convention.
