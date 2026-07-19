#!/usr/bin/env python3
"""
Generate sector_layouts.json: pixel positions of planet slots per sector.

Run from apps/eclipse_ui/:
  python tools/gen_sector_layouts.py [--debug] [--sector 101]

Existing layouts are preserved by default. Use --regenerate-existing to
re-detect planet and structure positions from the sector images.

Outputs data/sector_layouts.json with (dx, dy) offsets from image center (512, 512).

UI renders overlays at: svgX = cx + dx * scale, svgY = cy + dy * scale
where scale = (2 * hexSize) / 1024, wrapped in the same rotation transform
as the sector image so overlays rotate correctly with the tile.

Influence space is assumed to be at the hex center (dx=0, dy=0) for all sectors.
"""

import cv2
import numpy as np
import json
import re
import argparse
import sys
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent.parent
TEMPLATES_DIR = BASE / "tools/templates"
SECTORS_DIR = BASE / "data/sectors"
OUTPUT_FILE = BASE / "data/sector_layouts.json"
DEBUG_DIR = BASE / "tools/debug_layouts"

IMAGE_CENTER = 512      # All images are 1024x1024
HEX_CIRCUMRADIUS = 450  # Estimated pixel radius of hex in 1024px image space

# Hardcoded from open_spiel/games/eclipse/sectors.h, SECTOR_TABLE
# Each list = slot types in slot-index order (bit i of occupied_slots_mask = slot i)
SECTOR_SLOTS: dict[int, list[str]] = {
    1:   ["ADV_MATERIALS", "MATERIALS", "ADV_SCIENCE", "SCIENCE", "ADV_MONEY", "MONEY"],
    101: ["ADV_MATERIALS", "MATERIALS", "MONEY"],
    102: ["SCIENCE", "SCIENCE"],
    103: ["SCIENCE", "MONEY"],
    104: ["ADV_MONEY", "MONEY", "ADV_SCIENCE", "SCIENCE"],
    105: ["MONEY", "SCIENCE", "ADV_MATERIALS"],
    106: ["SCIENCE", "MATERIALS"],
    107: ["ADV_SCIENCE", "MONEY", "ADV_MATERIALS"],
    108: ["ADV_MONEY", "SCIENCE", "ANY"],
    109: ["MATERIALS", "MONEY"],
    110: ["ADV_MONEY", "ADV_ANY"],
    201: ["MATERIALS", "MONEY"],
    202: ["ADV_SCIENCE", "SCIENCE"],
    203: ["MATERIALS", "MONEY", "SCIENCE"],
    204: ["ADV_MONEY", "ADV_MATERIALS", "ANY"],
    205: ["ADV_MONEY", "MONEY", "ADV_SCIENCE"],
    206: ["MATERIALS"],
    207: [],
    208: [],
    209: ["MONEY", "SCIENCE"],
    210: ["MONEY", "MATERIALS"],
    211: ["MONEY", "MATERIALS", "ANY"],
    214: ["ADV_ANY", "ADV_MATERIALS", "SCIENCE"],
    221: ["ADV_MONEY", "MONEY", "ADV_SCIENCE", "SCIENCE", "MATERIALS"],
    222: ["ADV_MONEY", "MONEY", "ADV_SCIENCE", "SCIENCE"],
    223: ["ADV_MONEY", "MONEY", "ADV_SCIENCE", "SCIENCE", "MATERIALS"],
    224: ["MONEY", "ADV_SCIENCE", "ADV_MATERIALS"],
    225: ["ADV_MONEY", "MONEY", "ADV_SCIENCE", "SCIENCE", "MATERIALS"],
    226: ["SCIENCE", "MATERIALS"],
    227: ["ADV_MONEY", "MONEY", "ADV_SCIENCE", "SCIENCE", "MATERIALS"],
    228: ["MONEY", "SCIENCE", "ADV_MATERIALS"],
    229: ["ADV_MONEY", "MONEY", "ADV_SCIENCE", "SCIENCE", "MATERIALS"],
    230: ["ADV_MONEY", "MONEY", "SCIENCE", "ADV_MATERIALS"],
    231: ["ADV_MONEY", "MONEY", "ADV_SCIENCE", "SCIENCE", "MATERIALS"],
    232: ["ADV_MONEY", "SCIENCE", "ADV_MATERIALS", "MATERIALS"],
    271: ["ADV_MATERIALS", "MONEY", "SCIENCE"],
    272: ["MONEY", "ADV_SCIENCE", "MATERIALS"],
    273: ["ADV_MATERIALS", "MATERIALS", "MONEY"],
    274: ["ADV_SCIENCE", "SCIENCE", "MONEY"],
    281: ["SCIENCE", "MONEY"],
    301: ["SCIENCE", "MONEY", "ADV_MATERIALS"],
    302: ["ADV_SCIENCE", "ADV_MONEY", "MATERIALS"],
    303: ["ADV_SCIENCE", "ADV_MONEY", "ANY"],
    304: ["ADV_MONEY", "MATERIALS"],
    305: ["SCIENCE", "MATERIALS"],
    306: ["MONEY", "MATERIALS"],
    307: ["MONEY", "ADV_SCIENCE"],
    308: ["SCIENCE", "ADV_MATERIALS"],
    309: ["MONEY", "ADV_SCIENCE"],
    310: ["SCIENCE", "MATERIALS"],
    311: ["MATERIALS"],
    312: ["MATERIALS"],
    313: ["ANY"],
    314: ["ANY"],
    315: [],
    316: [],
    317: ["ADV_MONEY", "MONEY"],
    318: ["ADV_MATERIALS", "ANY"],
    381: ["MATERIALS", "MONEY"],
    382: ["SCIENCE", "MATERIALS"],
}


def load_templates() -> dict[str, np.ndarray]:
    templates = {}
    for name in ["MONEY", "SCIENCE", "MATERIALS", "ADV_MONEY", "ADV_SCIENCE",
                 "ADV_MATERIALS", "ANY", "ADV_ANY"]:
        path = TEMPLATES_DIR / f"{name}.png"
        if path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                templates[name] = img
            else:
                print(f"WARNING: Could not load template {path}", file=sys.stderr)
        else:
            print(f"WARNING: Template not found: {path}", file=sys.stderr)
    return templates


def build_sector_manifest(sectors_dir: Path) -> dict[int, Path]:
    """Map sector_id -> image path by extracting 3-digit ID from filename."""
    manifest = {}
    for p in sectors_dir.glob("*.png"):
        m = re.search(r'(\d{3})', p.name)
        if m:
            sid = int(m.group(1))
            manifest[sid] = p
    return manifest


# Planet icons live in the inner ~88% of the hex; wormhole rings live at the edge.
INNER_HEX_FRACTION = 0.88

# Hue ranges in OpenCV HSV (H: 0-179) for each planet type's icon background colour.
# MONEY icons: orange/gold.  SCIENCE icons: pink/magenta.  MATERIALS: brown/dark.
# These are used to reject candidates whose detected-pixel colour doesn't match.
HUE_RANGES: dict[str, tuple | None] = {
    "MONEY":         (3,  40),   # orange (wider: some icons render darker/more yellow)
    "ADV_MONEY":     (3,  40),   # orange
    "SCIENCE":       (130, 179), # pink/magenta (wraps at 0, but 130-179 covers it)
    "ADV_SCIENCE":   (130, 179),
    "MATERIALS":     None,       # brown/gray: no hue check (saturation check instead)
    "ADV_MATERIALS": None,
    "ANY":           None,
    "ADV_ANY":       None,
}
COLOR_CHECK_RADIUS = 28   # pixels around detected centre to sample
COLOR_MIN_SAT    = 70     # ignore near-gray pixels
COLOR_MIN_VAL    = 70     # ignore near-black pixels
COLOR_MIN_PIXELS = 8      # need at least this many coloured pixels


def _hue_ok(img_hsv: np.ndarray, cx: int, cy: int, ptype: str) -> bool:
    """Return True if the dominant hue at (cx,cy) is compatible with ptype."""
    hue_range = HUE_RANGES.get(ptype)
    if hue_range is None:
        return True  # no constraint
    r = COLOR_CHECK_RADIUS
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(img_hsv.shape[1], cx + r), min(img_hsv.shape[0], cy + r)
    roi = img_hsv[y1:y2, x1:x2]
    mask = (roi[:, :, 1] > COLOR_MIN_SAT) & (roi[:, :, 2] > COLOR_MIN_VAL)
    if mask.sum() < COLOR_MIN_PIXELS:
        return True  # insufficient colour data — don't reject
    avg_hue = float(roi[:, :, 0][mask].mean())
    lo, hi = hue_range
    return lo <= avg_hue <= hi


def collect_candidates(img: np.ndarray, templates: dict[str, np.ndarray],
                       type_counts: dict[str, int],
                       min_dist: int = 90, n_per_type: int = 8
                       ) -> list[tuple]:
    """
    For each required planet type, collect up to n_per_type candidate positions.
    Two filters applied to each candidate:
      1. Must fall inside the inner hex (rejects wormhole-ring false positives).
      2. Dominant hue at the peak must match the planet type's expected colour
         (rejects ADV_SCIENCE matching ADV_MONEY icons and vice-versa).
    Returns list of (confidence, ptype, center_x, center_y) sorted desc by conf.
    """
    inner_R = HEX_CIRCUMRADIUS * INNER_HEX_FRACTION
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    all_candidates = []
    for ptype, count in type_counts.items():
        if ptype not in templates:
            continue
        tmpl = templates[ptype]
        th, tw = tmpl.shape[:2]
        match = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        result = match.copy()
        for _ in range(count * n_per_type):
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < CONFIDENCE_THRESHOLD:
                break
            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            dx, dy = cx - IMAGE_CENTER, cy - IMAGE_CENTER
            x, y = max_loc
            x1, y1 = max(0, x - min_dist), max(0, y - min_dist)
            x2, y2 = min(result.shape[1], x + min_dist), min(result.shape[0], y + min_dist)
            result[y1:y2, x1:x2] = -2.0
            if (in_hex_flat_top(dx, dy, inner_R)
                    and _hue_ok(img_hsv, cx, cy, ptype)):
                # Boost MONEY/ADV_MONEY confidence at orange pixels so they
                # always win over MATERIALS in the greedy assignment.
                conf = float(max_val)
                if ptype in ("MONEY", "ADV_MONEY"):
                    roi = img_hsv[max(0, cy-COLOR_CHECK_RADIUS):cy+COLOR_CHECK_RADIUS,
                                  max(0, cx-COLOR_CHECK_RADIUS):cx+COLOR_CHECK_RADIUS]
                    sat_mask = (roi[:, :, 1] > 120) & (roi[:, :, 2] > 80)
                    if sat_mask.sum() > COLOR_MIN_PIXELS:
                        conf += 0.15  # orange = very likely MONEY, give it priority
                all_candidates.append((conf, ptype, cx, cy))
    all_candidates.sort(key=lambda c: -c[0])
    return all_candidates


def greedy_assign(candidates: list[tuple], type_counts: dict[str, int],
                  min_dist: int = 90) -> dict[str, list[tuple]]:
    """
    Greedy bipartite assignment with global spatial NMS.
    High-confidence matches win; once a position is taken no other type
    can claim within min_dist, preventing cross-type position conflicts.
    Returns {ptype: [(cx, cy, conf), ...]} in detection order.
    """
    remaining = dict(type_counts)
    occupied: list[tuple] = []   # (cx, cy) of assigned positions
    assigned: dict[str, list] = {t: [] for t in type_counts}

    for (conf, ptype, cx, cy) in candidates:
        if remaining.get(ptype, 0) <= 0:
            continue
        too_close = any(
            ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < min_dist
            for ox, oy in occupied
        )
        if too_close:
            continue
        assigned[ptype].append((cx, cy, conf))
        occupied.append((cx, cy))
        remaining[ptype] -= 1

    return assigned


def in_hex_flat_top(dx: float, dy: float, R: float = HEX_CIRCUMRADIUS) -> bool:
    """Is (dx,dy) from center inside a flat-top regular hexagon of circumradius R?"""
    r_in = R * 3**0.5 / 2  # inradius
    return (abs(dy) <= r_in
            and abs(3**0.5 * dx + dy) <= 2 * r_in
            and abs(3**0.5 * dx - dy) <= 2 * r_in)


def find_free_anchors(occupied_dxdy: list[tuple], n: int = 2,
                      R: float = HEX_CIRCUMRADIUS, step: int = 40,
                      inner_fraction: float = 0.82) -> list[dict]:
    """
    Find n positions inside the hex that maximise min-distance to occupied positions.
    The returned anchors use dx/dy offsets from the image center.
    """
    candidates = []
    r_inner = int(R * inner_fraction)
    for dy in range(-r_inner, r_inner + 1, step):
        for dx in range(-r_inner, r_inner + 1, step):
            if in_hex_flat_top(dx, dy, R * 0.85):
                candidates.append((dx, dy))

    if not candidates:
        defaults = [{"dx": 0, "dy": int(R * 0.3 * i)} for i in range(n)]
        return defaults

    anchors = []
    taken: list[tuple] = []
    for _ in range(n):
        best: tuple | None = None
        best_score = -1.0
        for (dx, dy) in candidates:
            all_others = occupied_dxdy + taken
            if all_others:
                dists = [((dx - px)**2 + (dy - py)**2)**0.5
                         for px, py in all_others]
                score = min(dists)
            else:
                score = (dx**2 + dy**2)**0.5  # furthest from center if no planets
            if score > best_score:
                best_score = score
                best = (dx, dy)
        if best:
            anchors.append({"dx": int(best[0]), "dy": int(best[1])})
            taken.append(best)
    return anchors


CONFIDENCE_THRESHOLD = 0.25
SHIP_ANCHOR_COUNT = 16


def add_ship_anchors(layout: dict) -> dict:
    """Add or refresh ship anchors without changing manual layout fields."""
    planets = layout.get("planets", [])
    planet_dxdy = [
        (planet["dx"], planet["dy"])
        for planet in planets
        if planet.get("confidence", 0) > 0
    ]
    structure_dxdy = [
        (layout[key]["dx"], layout[key]["dy"])
        for key in ("monolith_anchor", "orbital_anchor")
        if key in layout
    ]
    occupied_for_ships = planet_dxdy + [(0, 0)] + structure_dxdy
    updated_layout = dict(layout)
    updated_layout["ship_anchors"] = find_free_anchors(
        occupied_for_ships,
        n=SHIP_ANCHOR_COUNT,
        step=55,
        inner_fraction=0.74,
    )
    return updated_layout


def process_sector(
    sector_id: int,
    img_path: Path,
    templates: dict[str, np.ndarray],
    slots: list[str],
    debug: bool = False,
) -> dict:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot load {img_path}")
    if img.shape[:2] != (1024, 1024):
        img = cv2.resize(img, (1024, 1024))

    warnings = []
    type_counts = Counter(slots)

    # Check for missing templates
    for ptype in type_counts:
        if ptype not in templates:
            warnings.append(f"no template for {ptype}")

    # Global greedy assignment: collect all candidates across all types,
    # then assign greedily by confidence with cross-type spatial NMS.
    # This prevents ADV_X templates from stealing positions from other ADV_ types.
    candidates = collect_candidates(img, templates, type_counts, min_dist=90)
    assigned = greedy_assign(candidates, type_counts, min_dist=90)

    # Convert absolute pixel centres → (dx, dy) from image centre
    # Sort each type's positions top-to-bottom for stable slot index assignment
    found_by_type: dict[str, list[tuple]] = {}
    for ptype, positions in assigned.items():
        offsets = [(cx - IMAGE_CENTER, cy - IMAGE_CENTER, conf)
                   for (cx, cy, conf) in positions]
        offsets.sort(key=lambda p: (p[1], p[0]))  # top-to-bottom, left-to-right
        found_by_type[ptype] = offsets
        count = type_counts.get(ptype, 0)
        if len(offsets) < count:
            warnings.append(f"expected {count}x {ptype}, found {len(offsets)}")

    # Assign found positions to slot indices in SECTOR_SLOTS order
    type_used: dict[str, int] = {t: 0 for t in found_by_type}
    planets = []
    for slot_idx, ptype in enumerate(slots):
        positions = found_by_type.get(ptype, [])
        used = type_used.get(ptype, 0)
        if used < len(positions):
            dx, dy, conf = positions[used]
            type_used[ptype] = used + 1
            planets.append({
                "slot": slot_idx,
                "type": ptype,
                "dx": int(dx),
                "dy": int(dy),
                "confidence": round(conf, 3),
            })
        else:
            planets.append({
                "slot": slot_idx,
                "type": ptype,
                "dx": 0,
                "dy": 0,
                "confidence": 0.0,
            })

    planet_dxdy = [(p["dx"], p["dy"]) for p in planets if p["confidence"] > 0]
    anchors = find_free_anchors(planet_dxdy, n=2)
    result: dict = {
        "influence_space": {"dx": 0, "dy": 0},
        "planets": planets,
        "monolith_anchor": anchors[0] if anchors else {"dx": -200, "dy": 0},
        "orbital_anchor": anchors[1] if len(anchors) > 1 else {"dx": -200, "dy": 80},
    }
    result = add_ship_anchors(result)
    if warnings:
        result["warnings"] = warnings

    if debug:
        _save_debug_image(sector_id, img, planets, anchors)

    return result


def _save_debug_image(sector_id: int, img: np.ndarray,
                       planets: list[dict], anchors: list[dict]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out = img.copy()
    TYPE_COLORS = {
        "MONEY":         (0, 200, 255),   # bright cyan-yellow (visible on dark)
        "SCIENCE":       (255, 60, 220),  # magenta
        "MATERIALS":     (200, 200, 200), # white-gray
        "ADV_MONEY":     (0, 255, 255),   # yellow (BGR)
        "ADV_SCIENCE":   (255, 0, 255),   # bright magenta
        "ADV_MATERIALS": (180, 255, 180), # bright green
        "ANY":           (255, 255, 255), # white
        "ADV_ANY":       (200, 200, 255), # light purple
    }
    cx0, cy0 = IMAGE_CENTER, IMAGE_CENTER
    for p in planets:
        px, py = cx0 + p["dx"], cy0 + p["dy"]
        color = TYPE_COLORS.get(p["type"], (255, 255, 255))
        conf = p["confidence"]
        cv2.circle(out, (px, py), 30, color, 3)
        cv2.putText(out, f"{p['type'][:3]}{p['slot']}({conf:.2f})",
                    (px - 30, py - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for i, a in enumerate(anchors):
        ax, ay = cx0 + a["dx"], cy0 + a["dy"]
        label = "MON" if i == 0 else "ORB"
        cv2.circle(out, (ax, ay), 20, (0, 255, 0), 2)
        cv2.putText(out, label, (ax - 15, ay - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Mark center
    cv2.circle(out, (cx0, cy0), 10, (255, 255, 0), -1)
    out_path = DEBUG_DIR / f"sector_{sector_id:03d}.jpg"
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true",
                        help="Save annotated debug images to tools/debug_layouts/")
    parser.add_argument("--sector", type=int, default=None,
                        help="Process only this sector ID")
    parser.add_argument("--regenerate-existing", action="store_true",
                        help="Re-detect existing sectors instead of preserving them")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Minimum template match confidence (default 0.25)")
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.threshold

    templates = load_templates()
    print(f"Loaded {len(templates)} templates: {sorted(templates)}")

    manifest = build_sector_manifest(SECTORS_DIR)
    print(f"Found {len(manifest)} sector images")

    existing: dict = {}
    if OUTPUT_FILE.exists():
        existing = json.loads(OUTPUT_FILE.read_text())

    layouts: dict = dict(existing)
    issues: list[tuple[int, list[str]]] = []

    target_ids = [args.sector] if args.sector else sorted(SECTOR_SLOTS.keys())

    for sid in target_ids:
        if sid not in manifest:
            print(f"  SKIP {sid}: no image found")
            continue

        slots = SECTOR_SLOTS.get(sid, [])
        label = f"sector {sid:3d} ({len(slots)} slots)"

        if str(sid) in existing and not args.regenerate_existing:
            layouts[str(sid)] = add_ship_anchors(existing[str(sid)])
            print(f"  {label}  → preserved existing layout")
            continue

        if not slots:
            # Sectors with no planets still need influence/anchor data
            structure_anchors = [{"dx": -200, "dy": 0}, {"dx": -200, "dy": 80}]
            layouts[str(sid)] = add_ship_anchors({
                "influence_space": {"dx": 0, "dy": 0},
                "planets": [],
                "monolith_anchor": structure_anchors[0],
                "orbital_anchor": structure_anchors[1],
            })
            print(f"  {label}  → no planets, skipping match")
            continue

        print(f"  {label}  ...", end=" ", flush=True)
        try:
            layout = process_sector(sid, manifest[sid], templates, slots, args.debug)
            layouts[str(sid)] = layout

            warns = layout.get("warnings", [])
            if warns:
                print(f"WARN: {'; '.join(warns)}")
                issues.append((sid, warns))
            else:
                confs = [p["confidence"] for p in layout["planets"]]
                avg = sum(confs) / len(confs) if confs else 0.0
                print(f"OK  avg_conf={avg:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")
            issues.append((sid, [str(e)]))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(layouts, indent=2))
    print(f"\nWrote {OUTPUT_FILE}  ({len(layouts)} sectors)")

    if issues:
        print(f"\n=== {len(issues)} sectors with match issues ===")
        for sid, warns in sorted(issues):
            print(f"  Sector {sid:3d}: {'; '.join(warns)}")

    # Sectors where influence space is NOT simply the hex center:
    # In Eclipse, the influence disc goes into the printed "Influence Space"
    # icon on each tile, which is a small circular region near the top-left
    # inside the hex. ALL sectors have this icon in roughly the same position.
    # Assuming center (dx=0, dy=0) is a simplification — after visual review,
    # the actual influence icon is typically at approximately (-250, -310)
    # from center (upper-left). Mark here if any exceptions found:
    print("\nNote: influence_space set to center (0,0) for all sectors.")
    print("After visual review, override specific sectors in the JSON if needed.")


if __name__ == "__main__":
    main()
