#!/usr/bin/env python3
"""
Interactive layout review + editor — PyQt6.

Left/Right arrows (or A/D) to cycle sectors.  Q / Escape to quit.
Click-and-drag any marker to reposition it; position is written to
sector_layouts.json on mouse-release.

Dots:    basic type  = solid colour fill
         advanced    = white fill + coloured outline
Squares: purple = monolith anchor, green = orbital anchor
Selected marker is highlighted with a bright ring.
"""

import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import (QBrush, QColor, QImage, QKeyEvent,
                          QMouseEvent, QPainter, QPen, QPixmap)
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar

BASE         = Path(__file__).parent.parent
LAYOUTS_FILE = BASE / "data/sector_layouts.json"
SECTORS_DIR  = BASE / "data/sectors"

IMAGE_SIZE   = 1024
IMAGE_CENTER = IMAGE_SIZE // 2
DISPLAY_SIZE = 900        # square window / pixmap display size
DOT_R        = 18         # planet dot radius (image px)
SQ_HALF      = 10         # half-side of structure squares (image px)
HIT_RADIUS   = 28         # click detection radius (image px)

# colours
GOLD   = QColor(255, 200,  20)
PINK   = QColor(255, 100, 180)
BROWN  = QColor(160,  80,  20)
GRAY   = QColor(160, 160, 160)
WHITE  = QColor(255, 255, 255)
BLACK  = QColor(  0,   0,   0)
PURPLE = QColor(160,   0, 200)
GREEN  = QColor( 40, 210,  60)
SEL    = QColor(255, 255,   0)  # selection highlight

TYPE_STYLE: dict[str, tuple[QColor, bool]] = {
    "MONEY":         (GOLD,  False),
    "ADV_MONEY":     (GOLD,  True),
    "SCIENCE":       (PINK,  False),
    "ADV_SCIENCE":   (PINK,  True),
    "MATERIALS":     (BROWN, False),
    "ADV_MATERIALS": (BROWN, True),
    "ANY":           (GRAY,  False),
    "ADV_ANY":       (GRAY,  True),
}


def build_manifest() -> dict[int, Path]:
    m: dict[int, Path] = {}
    for p in SECTORS_DIR.glob("*.png"):
        hit = re.search(r"(\d{3})", p.name)
        if hit:
            m[int(hit.group(1))] = p
    return m


# ─── rendering ────────────────────────────────────────────────────────────────

def load_base_image(path: Path) -> QImage:
    img = QImage(str(path))
    if img.isNull():
        img = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format.Format_RGB888)
        img.fill(QColor(30, 30, 30))
    return img.convertToFormat(QImage.Format.Format_ARGB32)


def render(base: QImage, layout: dict, selected=None) -> QPixmap:
    """Draw all markers on a copy of base; highlight selected marker."""
    px_img = base.copy()
    p = QPainter(px_img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # ── planets ──────────────────────────────────────────────────────────
    for i, planet in enumerate(layout.get("planets", [])):
        cx = IMAGE_CENTER + planet["dx"]
        cy = IMAGE_CENTER + planet["dy"]
        colour, advanced = TYPE_STYLE.get(planet["type"], (WHITE, False))
        is_sel = selected == ("planet", i)

        if advanced:
            p.setPen(QPen(SEL if is_sel else colour, 5 if is_sel else 4))
            p.setBrush(QBrush(WHITE))
        else:
            p.setPen(QPen(SEL if is_sel else BLACK, 3 if is_sel else 1))
            p.setBrush(QBrush(colour))

        p.drawEllipse(cx - DOT_R, cy - DOT_R, DOT_R * 2, DOT_R * 2)

        if is_sel:
            p.setPen(QPen(SEL, 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(cx - DOT_R - 5, cy - DOT_R - 5,
                          (DOT_R + 5) * 2, (DOT_R + 5) * 2)

    # ── monolith ─────────────────────────────────────────────────────────
    ma = layout.get("monolith_anchor")
    if ma:
        mx, my = IMAGE_CENTER + ma["dx"], IMAGE_CENTER + ma["dy"]
        is_sel = selected == ("monolith",)
        p.setPen(QPen(SEL if is_sel else BLACK, 3 if is_sel else 1))
        p.setBrush(QBrush(PURPLE))
        p.drawRect(mx - SQ_HALF, my - SQ_HALF, SQ_HALF * 2, SQ_HALF * 2)

    # ── orbital ──────────────────────────────────────────────────────────
    oa = layout.get("orbital_anchor")
    if oa:
        ox, oy = IMAGE_CENTER + oa["dx"], IMAGE_CENTER + oa["dy"]
        is_sel = selected == ("orbital",)
        p.setPen(QPen(SEL if is_sel else BLACK, 3 if is_sel else 1))
        p.setBrush(QBrush(GREEN))
        p.drawRect(ox - SQ_HALF, oy - SQ_HALF, SQ_HALF * 2, SQ_HALF * 2)

    p.end()
    return QPixmap.fromImage(px_img)


# ─── draggable label ──────────────────────────────────────────────────────────

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__(alignment=Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(DISPLAY_SIZE, DISPLAY_SIZE)
        self.setMouseTracking(True)

        self._base: QImage | None = None
        self._layout: dict = {}
        self._selected = None   # ('planet', i) | ('monolith',) | ('orbital',)
        self._dragging = False
        self._on_save = None    # callback(layout) called on drop

    def set_sector(self, base: QImage, layout: dict, on_save):
        self._base     = base
        self._layout   = deepcopy(layout)
        self._selected = None
        self._dragging = False
        self._on_save  = on_save
        self._refresh()

    def _refresh(self):
        pm = render(self._base, self._layout, self._selected)
        self.setPixmap(
            pm.scaled(DISPLAY_SIZE, DISPLAY_SIZE,
                      Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)
        )

    # ── coordinate helpers ────────────────────────────────────────────────

    def _to_image(self, qp: QPoint) -> tuple[int, int]:
        """Label pixel → 1024-px image pixel."""
        pm = self.pixmap()
        if pm is None:
            return qp.x(), qp.y()
        ox = (self.width()  - pm.width())  // 2
        oy = (self.height() - pm.height()) // 2
        scale = IMAGE_SIZE / pm.width()      # pm is square → one scale
        return int((qp.x() - ox) * scale), int((qp.y() - oy) * scale)

    def _hit(self, ix: int, iy: int) -> tuple | None:
        """Return the marker id closest to image-space (ix, iy), or None."""
        best, best_d = None, HIT_RADIUS

        for i, planet in enumerate(self._layout.get("planets", [])):
            d = _dist(ix, iy, IMAGE_CENTER + planet["dx"],
                               IMAGE_CENTER + planet["dy"])
            if d < best_d:
                best, best_d = ("planet", i), d

        for key, anchor_key in (("monolith", "monolith_anchor"),
                                 ("orbital",  "orbital_anchor")):
            a = self._layout.get(anchor_key)
            if a:
                d = _dist(ix, iy, IMAGE_CENTER + a["dx"],
                                   IMAGE_CENTER + a["dy"])
                if d < best_d:
                    best, best_d = (key,), d

        return best

    # ── mouse events ──────────────────────────────────────────────────────

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        ix, iy = self._to_image(ev.pos())
        hit = self._hit(ix, iy)
        if hit:
            self._selected = hit
            self._dragging = True
            self._refresh()

    def mouseMoveEvent(self, ev: QMouseEvent):
        if not self._dragging or self._selected is None:
            return
        ix, iy = self._to_image(ev.pos())
        # clamp to image bounds
        ix = max(0, min(IMAGE_SIZE, ix))
        iy = max(0, min(IMAGE_SIZE, iy))
        dx, dy = ix - IMAGE_CENTER, iy - IMAGE_CENTER
        self._set_selected_pos(dx, dy)
        self._refresh()

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        if self._on_save and self._selected:
            self._on_save(self._layout)

    # ── internal position setter ──────────────────────────────────────────

    def _set_selected_pos(self, dx: int, dy: int):
        s = self._selected
        if s is None:
            return
        if s[0] == "planet":
            self._layout["planets"][s[1]]["dx"] = dx
            self._layout["planets"][s[1]]["dy"] = dy
        elif s[0] == "monolith":
            self._layout["monolith_anchor"] = {"dx": dx, "dy": dy}
        elif s[0] == "orbital":
            self._layout["orbital_anchor"] = {"dx": dx, "dy": dy}


def _dist(ax, ay, bx, by) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ─── main window ─────────────────────────────────────────────────────────────

class Viewer(QMainWindow):
    def __init__(self, sector_ids: list[int], layouts: dict,
                 manifest: dict[int, Path]):
        super().__init__()
        self.sector_ids = sector_ids
        self.layouts    = layouts       # full dict — mutated in place on save
        self.manifest   = manifest
        self.idx        = 0

        self.image_label = ImageLabel()
        self.setCentralWidget(self.image_label)
        self.setStatusBar(QStatusBar())
        self.setFixedSize(DISPLAY_SIZE, DISPLAY_SIZE + self.statusBar().height())

        self._load_current()

    # ── sector navigation ─────────────────────────────────────────────────

    def _load_current(self):
        sid    = self.sector_ids[self.idx]
        layout = self.layouts[str(sid)]
        base   = load_base_image(self.manifest[sid])
        n_pl   = len(layout.get("planets", []))

        self.image_label.set_sector(base, layout, self._on_save)
        self.setWindowTitle(
            f"Sector {sid}  [{self.idx + 1}/{len(self.sector_ids)}]"
            f"  —  {n_pl} planet{'s' if n_pl != 1 else ''}"
            f"    ◀ A/←  D/→ ▶    Q quit"
        )
        self.statusBar().showMessage(
            "Click-drag any dot/square to move it — saved automatically on release."
        )

    def _on_save(self, layout: dict):
        sid = self.sector_ids[self.idx]
        self.layouts[str(sid)] = layout
        LAYOUTS_FILE.write_text(json.dumps(self.layouts, indent=2))
        self.statusBar().showMessage(f"Saved sector {sid}.", 2000)

    # ── keyboard navigation ───────────────────────────────────────────────

    def keyPressEvent(self, ev: QKeyEvent):
        key = ev.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_A):
            self.idx = (self.idx - 1) % len(self.sector_ids)
            self._load_current()
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_D):
            self.idx = (self.idx + 1) % len(self.sector_ids)
            self._load_current()
        elif key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()


# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    if not LAYOUTS_FILE.exists():
        print(f"Missing {LAYOUTS_FILE} — run gen_sector_layouts.py first",
              file=sys.stderr)
        sys.exit(1)

    layouts  = json.loads(LAYOUTS_FILE.read_text())
    manifest = build_manifest()

    sector_ids = sorted(int(k) for k in layouts if int(k) in manifest)
    if not sector_ids:
        print("No matching sector images found.", file=sys.stderr)
        sys.exit(1)

    app    = QApplication(sys.argv)
    viewer = Viewer(sector_ids, layouts, manifest)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
