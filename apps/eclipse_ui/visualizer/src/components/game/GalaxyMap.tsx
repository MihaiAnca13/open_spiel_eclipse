import { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import type { MouseEvent } from 'react';
import type { GameState, LayoutAnchor, Sector, SectorLayout, Unit } from '../../types/game';
import { ACTION } from '../../actionTypes';
import { NPC_COLOR_GCDS, NPC_COLOR_GUARDIAN, UNOWNED_COLOR } from '../../theme';
import { getPlayerHexColor } from '../../utils/game';
import { axialToPixel, getHexPoints, hexSize, IMAGE_ROTATION_OFFSET, axialNeighbor } from '../../utils/hex';
import { getPlayerColor } from '../../theme';
import { shipImageUrl } from '../../types/lobby';
import { NPC_PLAYER_ID, GALAXY_MAP_SIZE, GALAXY_OFFSET } from '../../constants';


const INITIAL_SCALE = 1;
const WARP_LAYOUT_KIND = 6;
const NO_WARP_LINK = 255;
const EDGE_ANGLES_DEG = [30, -30, -90, -150, 150, 90] as const;
const HEX_EDGE_RADIUS = (hexSize - 1.5) * Math.sqrt(3) / 2;
const FALLBACK_SHIP_ANCHORS: LayoutAnchor[] = [
  { dx: -165, dy: 145 },
  { dx: -85, dy: 165 },
  { dx: -5, dy: 145 },
  { dx: 75, dy: 165 },
  { dx: 155, dy: 145 },
  { dx: -125, dy: 85 },
  { dx: -45, dy: 105 },
  { dx: 35, dy: 85 },
  { dx: 115, dy: 105 },
  { dx: -85, dy: 25 },
  { dx: -5, dy: 45 },
  { dx: 75, dy: 25 },
];
const SHIP_SIZE_BY_TYPE: Record<string, number> = {
  ancient: 17,
  cruiser: 18,
  dreadnought: 21,
  gcds: 22,
  guardian: 19,
  interceptor: 15,
  starbase: 18,
};
const DEFAULT_SHIP_SIZE = 16;
const SHIP_OUTLINE = 2.2;

interface UnitWithIndex {
  unit: Unit;
  globalIdx: number;
}

interface ShipRenderItem {
  key: string;
  unit: Unit;
  globalIdx: number;
  members: UnitWithIndex[];
  sectorId: number;
  x: number;
  y: number;
  size: number;
  color: string;
}

interface WarpCell {
  cell: number;
  q: number;
  r: number;
  cx: number;
  cy: number;
}

interface WarpLink {
  id: string;
  fromCell: number;
  fromDir: number;
  toCell: number;
  toDir: number;
}

function cellToAxial(cell: number) {
  return {
    q: Math.floor(cell / GALAXY_MAP_SIZE) - GALAXY_OFFSET,
    r: (cell % GALAXY_MAP_SIZE) - GALAXY_OFFSET,
  };
}

function axialToCell(q: number, r: number) {
  return (q + GALAXY_OFFSET) * GALAXY_MAP_SIZE + (r + GALAXY_OFFSET);
}

function cellCenter(cell: number) {
  const { q, r } = cellToAxial(cell);
  const { cx, cy } = axialToPixel(q, r);
  return { x: cx, y: cy };
}

function warpEdgePoint(cell: number, dir: number) {
  const { q, r } = cellToAxial(cell);
  const { cx, cy } = axialToPixel(q, r);
  const angle = (Math.PI / 180) * (EDGE_ANGLES_DEG[dir as 0 | 1 | 2 | 3 | 4 | 5] ?? 0);
  return {
    x: cx + Math.cos(angle) * HEX_EDGE_RADIUS,
    y: cy + Math.sin(angle) * HEX_EDGE_RADIUS,
  };
}

function warpGateLine(cell: number, dir: number) {
  const center = warpEdgePoint(cell, dir);
  const angle = (Math.PI / 180) * ((EDGE_ANGLES_DEG[dir as 0 | 1 | 2 | 3 | 4 | 5] ?? 0) + 90);
  const half = 7;
  return {
    x1: center.x - Math.cos(angle) * half,
    y1: center.y - Math.sin(angle) * half,
    x2: center.x + Math.cos(angle) * half,
    y2: center.y + Math.sin(angle) * half,
  };
}

function warpLinkPath(link: WarpLink) {
  const start = warpEdgePoint(link.fromCell, link.fromDir);
  const end = warpEdgePoint(link.toCell, link.toDir);
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const distance = Math.hypot(dx, dy);
  if (distance < 1) return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
  const fromCenter = cellCenter(link.fromCell);
  const toCenter = cellCenter(link.toCell);
  const cp = {
    x: (fromCenter.x + toCenter.x) / 2,
    y: (fromCenter.y + toCenter.y) / 2,
  };
  return `M ${start.x} ${start.y} Q ${cp.x} ${cp.y} ${end.x} ${end.y}`;
}

function normalizeShipType(type: string) {
  return type.trim().toLowerCase();
}

function getShipColor(unit: Unit) {
  if (unit.player_id !== NPC_PLAYER_ID) return getPlayerColor(unit.player_id);

  const type = normalizeShipType(String(unit.type));
  if (type === 'gcds') return NPC_COLOR_GCDS;
  if (type === 'guardian') return NPC_COLOR_GUARDIAN;
  return '#94a3b8';
}

function getShipSize(type: string) {
  return SHIP_SIZE_BY_TYPE[normalizeShipType(type)] ?? DEFAULT_SHIP_SIZE;
}

function isCenteredShip(type: string) {
  const normalizedType = normalizeShipType(type);
  return normalizedType === 'ancient' || normalizedType === 'guardian' || normalizedType === 'gcds';
}

function rotateAnchor(cx: number, cy: number, anchor: LayoutAnchor, scale: number, degrees: number) {
  const radians = (Math.PI / 180) * degrees;
  const x = anchor.dx * scale;
  const y = anchor.dy * scale;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return {
    x: cx + x * cos - y * sin,
    y: cy + x * sin + y * cos,
  };
}

interface GalaxyMapProps {
  gameState: GameState | null;
  sectorImages: Record<number, string>;
  sectorLayouts: Record<number, SectorLayout>;
  legalActions: number[];
  isMyTurn: boolean;
  explorePhase: string;
  selectedSectorId: number;
  previewRotation: number | null;
  setPreviewRotation: (rot: number | null) => void;
  previewingDrawnTile: number | null;
  previewingDrawnTileIndex: number | null;
  inSelectDrawnTile: boolean;
  submitAction: (actionId: number) => void;
  playerLabel: (pid: number) => string;
  beginPreview: (preview: { src: string; label: string }) => void;
  clearPreview: () => void;
  influenceToCellActions?: number[];
  reclaimFromCellActions?: number[];
  buildTargetActions?: number[];
  selectedBuildType?: number | null;
  moveUnitSectors?: Set<number>;
  moveUnitIndicesBySector?: Map<number, number[]>;
  moveTargetCells?: Map<number, number>;
  selectedMoveUnitIdx?: number | null;
  onSelectMoveUnit?: (idx: number | null) => void;
  onSelectMoveSector?: (sectorId: number | null) => void;
  showMapDebug: boolean;
}

export default function GalaxyMap({
  gameState,
  sectorImages,
  sectorLayouts,
  legalActions,
  isMyTurn,
  explorePhase,
  selectedSectorId,
  previewRotation,
  setPreviewRotation,
  previewingDrawnTile,
  previewingDrawnTileIndex,
  inSelectDrawnTile,
  submitAction,
  playerLabel,
  beginPreview,
  clearPreview,
  influenceToCellActions = [],
  reclaimFromCellActions = [],
  buildTargetActions = [],
  selectedBuildType,
  moveUnitSectors,
  moveUnitIndicesBySector,
  moveTargetCells,
  selectedMoveUnitIdx,
  onSelectMoveUnit,
  onSelectMoveSector,
  showMapDebug,
}: GalaxyMapProps) {
  const [hoveredSector, setHoveredSector] = useState<Sector | null>(null);
  const [brokenImages, setBrokenImages] = useState<Set<number>>(new Set());
  const [hoveredWarpLinkId, setHoveredWarpLinkId] = useState<string | null>(null);
  const [selectedWarpLinkId, setSelectedWarpLinkId] = useState<string | null>(null);
  const [isPanActive, setIsPanActive] = useState(false);

  // Influence targets: cell index -> actionId for quick lookup.
  const influencePlaceCells = useMemo(() => {
    const map = new Map<number, number>();
    for (const a of influenceToCellActions) {
      map.set(a - ACTION.INFLUENCE_TO_CELL_START, a);
    }
    return map;
  }, [influenceToCellActions]);
  const influenceReclaimCells = useMemo(() => {
    const map = new Map<number, number>();
    for (const a of reclaimFromCellActions) {
      map.set(a - ACTION.RECLAIM_FROM_CELL_START, a);
    }
    return map;
  }, [reclaimFromCellActions]);
  const inInfluencePhase = influenceToCellActions.length > 0 || reclaimFromCellActions.length > 0;

  // Build targets: cell index -> actionId for quick lookup.
  const buildTargetCells = useMemo(() => {
    const map = new Map<number, number>();
    for (const a of buildTargetActions) {
      map.set(a - ACTION.BUILD_CHOICE_START, a);
    }
    return map;
  }, [buildTargetActions]);
  const inBuildTargetPhase = selectedBuildType !== null && buildTargetActions.length > 0;

  // Move phase detection & target cells for highlighting
  const movePhase = gameState?.move_state?.phase ?? 'inactive';
  const inMovePhase = movePhase === 'choose_move' || movePhase === 'choose_warp_destination';

  // Pan/zoom state
  const [viewTransform, setViewTransform] = useState({ x: 0, y: 0, scale: INITIAL_SCALE });
  const isPanning = useRef(false);
  const panStart = useRef({ x: 0, y: 0, tx: 0, ty: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const centeredOnce = useRef(false);

  // ── Zoom limits ──
  const ZOOM_MIN = 0.3;
  const ZOOM_MAX = 3;
  // ─────────────────

  // Convert CSS pixel coords to SVG user-space using getScreenCTM
  const screenToSVG = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    return pt.matrixTransform(ctm.inverse());
  }, []);

  const zoomIn = useCallback(() => {
    setViewTransform(prev => ({ ...prev, scale: Math.min(ZOOM_MAX, prev.scale * 1.3) }));
  }, []);

  const zoomOut = useCallback(() => {
    setViewTransform(prev => ({ ...prev, scale: Math.max(ZOOM_MIN, prev.scale / 1.3) }));
  }, []);

  const resetView = useCallback(() => {
    setViewTransform({ x: 0, y: 0, scale: INITIAL_SCALE });
  }, []);

  // Zoom toward mouse cursor position
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const svg = svgRef.current;
    if (!svg) return;
    const cursorSVG = screenToSVG(e.clientX, e.clientY);
    setViewTransform(prev => {
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, prev.scale * delta));
      const ratio = newScale / prev.scale;
      return {
        x: cursorSVG.x - (cursorSVG.x - prev.x) * ratio,
        y: cursorSVG.y - (cursorSVG.y - prev.y) * ratio,
        scale: newScale,
      };
    });
  }, [screenToSVG]);

  // Pan: user-space deltas from screen coords
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    isPanning.current = true;
    setIsPanActive(true);
    const pt = screenToSVG(e.clientX, e.clientY);
    panStart.current = { x: pt.x, y: pt.y, tx: viewTransform.x, ty: viewTransform.y };
  }, [viewTransform, screenToSVG]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current) return;
    const pt = screenToSVG(e.clientX, e.clientY);
    const dx = pt.x - panStart.current.x;
    const dy = pt.y - panStart.current.y;
    setViewTransform(prev => ({
      ...prev,
      x: panStart.current.tx + dx,
      y: panStart.current.ty + dy,
    }));
  }, [screenToSVG]);

  const handleMouseUp = useCallback(() => {
    isPanning.current = false;
    setIsPanActive(false);
  }, []);

  // Non-passive wheel listener
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => { e.preventDefault(); };
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, []);

  const sectorImageUrl = (id: number): string | null =>
    (!brokenImages.has(id) && sectorImages[id]) || null;

  const unitsBySector = useMemo(() => {
    const map = new Map<number, UnitWithIndex[]>();
    gameState?.unit_registry.forEach((unit, globalIdx) => {
      const units = map.get(unit.sector_id) ?? [];
      units.push({ unit, globalIdx });
      map.set(unit.sector_id, units);
    });
    return map;
  }, [gameState]);

  const getUnitsInSector = useCallback((sectorId: number) => {
    return (unitsBySector.get(sectorId) ?? []).map(({ unit }) => unit);
  }, [unitsBySector]);

  const activeSectors: { sector: Sector; cx: number; cy: number }[] = useMemo(() => {
    const list: { sector: Sector; cx: number; cy: number }[] = [];
    if (gameState?.galaxy) {
      const mapSize = gameState.galaxy.length;
      for (let qIdx = 0; qIdx < mapSize; qIdx++) {
        for (let rIdx = 0; rIdx < mapSize; rIdx++) {
          const sector = gameState.galaxy[qIdx][rIdx];
          if (sector && sector.sector_id > 0) {
            const q = qIdx - GALAXY_OFFSET;
            const r = rIdx - GALAXY_OFFSET;
            const { cx, cy } = axialToPixel(q, r);
            list.push({ sector, cx, cy });
          }
        }
      }
    }
    return list;
  }, [gameState]);

  const shipRenderItems = useMemo<ShipRenderItem[]>(() => {
    const items: ShipRenderItem[] = [];
    const scale = (2 * (hexSize - 1.5)) / 1024;

    for (const { sector, cx, cy } of activeSectors) {
      const units = unitsBySector.get(sector.sector_id) ?? [];
      if (units.length === 0) continue;

      const layout = sectorLayouts[sector.sector_id];
      const anchors = layout?.ship_anchors?.length ? layout.ship_anchors : FALLBACK_SHIP_ANCHORS;
      const imgDeg = IMAGE_ROTATION_OFFSET - 60 * (sector.rotation ?? 0);

      const stacks = new Map<string, UnitWithIndex[]>();
      for (const entry of units) {
        const type = normalizeShipType(String(entry.unit.type));
        const key = `${entry.unit.player_id}:${type}`;
        const members = stacks.get(key) ?? [];
        members.push(entry);
        stacks.set(key, members);
      }

      let anchorIndex = 0;
      for (const [stackKey, members] of stacks) {
        const [entry] = members;
        const type = String(entry.unit.type);
        const centered = isCenteredShip(type);
        const index = anchorIndex;
        if (!centered) anchorIndex++;
        const anchor = anchors[index % anchors.length];
        const wrapOffset = Math.floor(index / anchors.length) * 4;
        const point = centered
          ? { x: cx, y: cy }
          : rotateAnchor(
            cx,
            cy,
            { dx: anchor.dx + wrapOffset, dy: anchor.dy + wrapOffset },
            scale,
            imgDeg
          );
        items.push({
          key: `${sector.sector_id}:${stackKey}`,
          unit: entry.unit,
          globalIdx: entry.globalIdx,
          members,
          sectorId: sector.sector_id,
          x: point.x,
          y: point.y,
          size: getShipSize(type),
          color: getShipColor(entry.unit),
        });
      }
    }

    return items;
  }, [activeSectors, sectorLayouts, unitsBySector]);

  const warpCells: WarpCell[] = useMemo(() => {
    if (!gameState?.warped_universe || !Array.isArray(gameState.layout_kinds)) return [];
    return gameState.layout_kinds
      .map((kind, cell) => {
        if (kind !== WARP_LAYOUT_KIND) return null;
        const { q, r } = cellToAxial(cell);
        return { cell, q, r, ...axialToPixel(q, r) };
      })
      .filter((cell): cell is WarpCell => cell !== null);
  }, [gameState]);

  const warpLinks: WarpLink[] = useMemo(() => {
    const destCells = gameState?.warp_link_dest_cell;
    const destDirs = gameState?.warp_link_dest_dir;
    if (!gameState?.warped_universe || !Array.isArray(destCells) || !Array.isArray(destDirs)) {
      return [];
    }

    const links: WarpLink[] = [];
    const seen = new Set<string>();
    for (let cell = 0; cell < GALAXY_MAP_SIZE * GALAXY_MAP_SIZE; cell++) {
      for (let dir = 0; dir < 6; dir++) {
        const idx = cell * 6 + dir;
        const toCell = destCells[idx];
        const toDir = destDirs[idx];
        if (toCell === undefined || toDir === undefined || toCell === NO_WARP_LINK || toDir === NO_WARP_LINK) {
          continue;
        }
        const a = `${cell}:${dir}`;
        const b = `${toCell}:${toDir}`;
        const key = a < b ? `${a}|${b}` : `${b}|${a}`;
        if (seen.has(key)) continue;
        seen.add(key);
        links.push({
          id: key,
          fromCell: cell,
          fromDir: dir,
          toCell,
          toDir,
        });
      }
    }
    return links;
  }, [gameState]);

  const warpDestByCellDir = useMemo(() => {
    const map = new Map<string, WarpLink>();
    for (const link of warpLinks) {
      map.set(`${link.fromCell}:${link.fromDir}`, link);
      map.set(`${link.toCell}:${link.toDir}`, {
        id: link.id,
        fromCell: link.toCell,
        fromDir: link.toDir,
        toCell: link.fromCell,
        toDir: link.fromDir,
      });
    }
    return map;
  }, [warpLinks]);

  // For move targets: compute destination cell for each direction
  const moveDestCells = useMemo(() => {
    const map = new Map<string, { actionId: number; isWarp: boolean; linkId?: string }>();
    if (!moveTargetCells || selectedMoveUnitIdx == null || !gameState) return map;
    const idx = selectedMoveUnitIdx;
    const unit = gameState.unit_registry[idx];
    if (!unit) return map;
    const sectorEntry = activeSectors.find(s => s.sector.sector_id === unit.sector_id);
    if (!sectorEntry) return map;
    const { q, r } = sectorEntry.sector.coords;
    const sourceCell = axialToCell(q, r);
    for (const [dir, actionId] of moveTargetCells) {
      if (dir < 6) {
        const warpLink = warpDestByCellDir.get(`${sourceCell}:${dir}`);
        if (warpLink) {
          const dest = cellToAxial(warpLink.toCell);
          map.set(`${dest.q},${dest.r}`, { actionId, isWarp: true, linkId: warpLink.id });
        } else {
          const [nq, nr] = axialNeighbor(q, r, dir);
          map.set(`${nq},${nr}`, { actionId, isWarp: false });
        }
      }
    }
    return map;
  }, [moveTargetCells, selectedMoveUnitIdx, gameState, activeSectors, warpDestByCellDir]);

  // Center view on sector 1 (0,0) on first render with sectors
  useEffect(() => {
    if (centeredOnce.current || activeSectors.length === 0) return;
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    centeredOnce.current = true;
    const centerScreen = screenToSVG(rect.width / 2, rect.height / 2);
    const sector1 = activeSectors.find(s => s.sector.sector_id === 1);
    if (sector1) {
      setViewTransform({
        x: centerScreen.x - sector1.cx,
        y: centerScreen.y - sector1.cy,
        scale: 1,
      });
    }
  }, [activeSectors, screenToSVG]);

  const inZoneSelect = isMyTurn && explorePhase === 'choose_zone';

  const legalZones = useMemo(() => {
    return inZoneSelect
      ? legalActions
          .filter((a) => a >= ACTION.EXPLORE_ZONE_START)
          .map((a) => {
            const idx = a - ACTION.EXPLORE_ZONE_START;
            const q = Math.floor(idx / GALAXY_MAP_SIZE) - GALAXY_OFFSET;
            const r = (idx % GALAXY_MAP_SIZE) - GALAXY_OFFSET;
            return { action: a, q, r, ...axialToPixel(q, r) };
          })
      : [];
  }, [inZoneSelect, legalActions]);

  const legalRotations = useMemo(
    () =>
      legalActions
        .filter((action) => action >= ACTION.EXPLORE_ROT_START && action < ACTION.EXPLORE_ROT_START + 6)
        .map((action) => action - ACTION.EXPLORE_ROT_START)
        .sort((a, b) => a - b),
    [legalActions]
  );

  const inRotationPreview =
    isMyTurn &&
    explorePhase === 'choose_rotation' &&
    selectedSectorId > 0 &&
    legalRotations.length > 0;

  const currentPreviewRotation =
    inRotationPreview && previewRotation !== null && legalRotations.includes(previewRotation)
      ? previewRotation
      : inRotationPreview
        ? legalRotations[0]
        : null;

  const cyclePreviewRotation = (direction: -1 | 1) => {
    if (!legalRotations.length) return;
    const currentIndex = Math.max(0, legalRotations.indexOf(currentPreviewRotation ?? legalRotations[0]));
    const nextIndex = (currentIndex + direction + legalRotations.length) % legalRotations.length;
    setPreviewRotation(legalRotations[nextIndex]);
  };

  const exploreState = gameState?.explore_state;

  const previewZone = useMemo(() => (
    inRotationPreview && exploreState
      ? {
          q: exploreState.zone_q,
          r: exploreState.zone_r,
          ...axialToPixel(exploreState.zone_q, exploreState.zone_r),
        }
      : null
  ), [exploreState, inRotationPreview]);

  const drawnTilePreviewZone = useMemo(() => (
    inSelectDrawnTile && exploreState && previewingDrawnTile !== null && previewingDrawnTileIndex !== null
      ? {
          q: exploreState.zone_q,
          r: exploreState.zone_r,
          ...axialToPixel(exploreState.zone_q, exploreState.zone_r),
          previewSectorId: previewingDrawnTile,
          previewTileIndex: previewingDrawnTileIndex,
        }
      : null
  ), [exploreState, inSelectDrawnTile, previewingDrawnTile, previewingDrawnTileIndex]);

  const emphasizedWarpLink = useMemo(() => {
    const linkId = hoveredWarpLinkId ?? selectedWarpLinkId;
    return linkId ? warpLinks.find((link) => link.id === linkId) ?? null : null;
  }, [hoveredWarpLinkId, selectedWarpLinkId, warpLinks]);

  const transformStr = `translate(${viewTransform.x},${viewTransform.y}) scale(${viewTransform.scale})`;

  const viewBox = useMemo(() => {
    const activeWarpCells = warpCells.map(({ cx, cy }) => ({ cx, cy }));
    const fitCells = previewZone
      ? [...activeWarpCells, ...activeSectors, ...legalZones, previewZone]
      : drawnTilePreviewZone
        ? [...activeWarpCells, ...activeSectors, ...legalZones, drawnTilePreviewZone]
        : [...activeWarpCells, ...activeSectors, ...legalZones];

    if (fitCells.length === 0) return '0 0 600 520';
    const xs = fitCells.map((c) => c.cx);
    const ys = fitCells.map((c) => c.cy);
    const pad = hexSize * 1.4;
    const minX = Math.min(...xs) - hexSize - pad;
    const minY = Math.min(...ys) - hexSize - pad;
    const w = Math.max(...xs) - Math.min(...xs) + 2 * (hexSize + pad);
    const h = Math.max(...ys) - Math.min(...ys) + 2 * (hexSize + pad);
    return `${minX} ${minY} ${w} ${h}`;
  }, [activeSectors, legalZones, previewZone, drawnTilePreviewZone, warpCells]);

  if (!gameState) {
    return (
      <div className="text-center text-[#64748b]">
        <svg className="w-16 h-16 mb-4 text-[#334155]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
        </svg>
        <h3>Galaxy Map Uninitialized</h3>
        <p className="text-sm">Send a Stage 1 setup config to spin up the shared C++ core</p>
      </div>
    );
  }

  return (
    <div className="map-viewport">
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox={viewBox}
        preserveAspectRatio="xMidYMid meet"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isPanActive ? 'grabbing' : 'grab' }}
      >
        <g transform={transformStr}>
          {warpCells.map(({ cell, cx, cy }) => (
            <polygon
              key={`warp-cell-${cell}`}
              points={getHexPoints(cx, cy, hexSize - 1.5)}
              className="warp-region-cell"
            />
          ))}

          {activeSectors.map(({ sector, cx, cy }) => {
          const fillColor = getPlayerHexColor(sector.owner_id, sector.sector_id);
          const isCenter = sector.sector_id === 1;
          const units = getUnitsInSector(sector.sector_id);
          const owned = sector.owner_id !== NPC_PLAYER_ID;
          const hasHostiles = units.some((u) => u.player_id === NPC_PLAYER_ID);
          const stroke = isCenter
            ? '#eab308'
            : owned
              ? getPlayerColor(sector.owner_id)
              : hasHostiles
                ? '#ef4444'
                : '#475569';
          const strokeWidth = isCenter || hasHostiles || owned ? '2.5' : '1.5';

          const imgUrl = sectorImageUrl(sector.sector_id);
          const r = hexSize - 1.5;
          const clipId = `hexclip-${sector.sector_id}-${sector.coords.q}-${sector.coords.r}`;
          const imgSize = 2 * r;
          const imgDeg = IMAGE_ROTATION_OFFSET - 60 * (sector.rotation ?? 0);

          return (
            <g
              key={`${sector.sector_id}-${sector.coords.q}-${sector.coords.r}`}
              onMouseEnter={() => {
                setHoveredSector(sector);
                if (imgUrl) {
                  beginPreview({
                    src: imgUrl,
                    label: sector.sector_id === 1 ? '001 (Center)' : `Sector ${sector.sector_id}`,
                  });
                }
              }}
              onMouseLeave={() => {
                setHoveredSector(null);
                clearPreview();
              }}
            >
              {imgUrl && (
                <>
                  <clipPath id={clipId}>
                    <polygon points={getHexPoints(cx, cy, r)} />
                  </clipPath>
                  <g clipPath={`url(#${clipId})`}>
                    <image
                      href={imgUrl}
                      x={cx - imgSize / 2}
                      y={cy - imgSize / 2}
                      width={imgSize}
                      height={imgSize}
                      preserveAspectRatio="xMidYMid slice"
                      transform={`rotate(${imgDeg} ${cx} ${cy})`}
                      onError={() =>
                        setBrokenImages((prev) => new Set(prev).add(sector.sector_id))
                      }
                      style={{ pointerEvents: 'none' }}
                    />
                  </g>
                </>
              )}

              <polygon
                points={getHexPoints(cx, cy, r)}
                fill={imgUrl ? 'none' : fillColor}
                fillOpacity={imgUrl ? 0 : owned || hasHostiles ? 1 : 0.65}
                stroke={stroke}
                strokeWidth={strokeWidth}
                className="hex-polygon"
                style={{ pointerEvents: 'all' }}
              />

              {showMapDebug && (
                <>
                  <text
                    x={cx}
                    y={cy - 6}
                    textAnchor="middle"
                    fill="#f8fafc"
                    fontSize="11px"
                    fontWeight="bold"
                    style={{ pointerEvents: 'none' }}
                  >
                    {sector.sector_id === 1 ? 'GCDS' : sector.sector_id}
                  </text>

                  <text
                    x={cx}
                    y={cy + 8}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize="8px"
                    style={{ pointerEvents: 'none' }}
                  >
                    ({sector.coords.q},{sector.coords.r})
                  </text>
                </>
              )}

              {/* ── sector overlay: planet cubes, influence disk, structures ── */}
              {(() => {
                const layout = sectorLayouts[sector.sector_id];
                if (!layout || !imgUrl) return null;
                const scale = imgSize / 1024;
                const pColor = owned ? getPlayerColor(sector.owner_id) : '#ffffff';
                return (
                  <g transform={`rotate(${imgDeg} ${cx} ${cy})`} style={{ pointerEvents: 'none' }}>
                    {/* Influence disk */}
                    {owned && (
                      <circle
                        cx={cx + layout.influence_space.dx * scale}
                        cy={cy + layout.influence_space.dy * scale}
                        r={5}
                        fill={pColor}
                        fillOpacity={0.9}
                        stroke="#000"
                        strokeWidth={0.6}
                      />
                    )}

                    {/* Planet population cubes */}
                    {layout.planets.map((planet) => {
                      const occupied = Boolean((sector.occupied_slots_mask >> planet.slot) & 1);
                      const px = cx + planet.dx * scale;
                      const py = cy + planet.dy * scale;
                      return occupied ? (
                        <rect
                          key={planet.slot}
                          x={px - 4}
                          y={py - 4}
                          width={8}
                          height={8}
                          rx={1}
                          fill={pColor}
                          fillOpacity={0.9}
                          stroke="#000"
                          strokeWidth={0.5}
                        />
                      ) : (
                        <rect
                          key={planet.slot}
                          x={px - 3.5}
                          y={py - 3.5}
                          width={7}
                          height={7}
                          rx={1}
                          fill="none"
                          stroke="#ffffff"
                          strokeWidth={0.6}
                          strokeOpacity={0.35}
                        />
                      );
                    })}

                    {/* Monolith */}
                    {sector.monolith_built && (
                      <rect
                        x={cx + layout.monolith_anchor.dx * scale - 4}
                        y={cy + layout.monolith_anchor.dy * scale - 6}
                        width={8}
                        height={12}
                        rx={1}
                        fill="#a855f7"
                        fillOpacity={0.9}
                        stroke="#000"
                        strokeWidth={0.5}
                      />
                    )}

                    {/* Orbital */}
                    {sector.orbital_built && (
                      <circle
                        cx={cx + layout.orbital_anchor.dx * scale}
                        cy={cy + layout.orbital_anchor.dy * scale}
                        r={4}
                        fill="#22d3ee"
                        fillOpacity={0.9}
                        stroke="#000"
                        strokeWidth={0.5}
                      />
                    )}
                  </g>
                );
              })()}

              {/* ── Influence overlay: clickable sectors for place/reclaim ── */}
              {inInfluencePhase && (() => {
                const cellIdx = (sector.coords.q + GALAXY_OFFSET) * GALAXY_MAP_SIZE + (sector.coords.r + GALAXY_OFFSET);
                const placeAction = influencePlaceCells.get(cellIdx);
                const reclaimAction = influenceReclaimCells.get(cellIdx);
                if (!placeAction && !reclaimAction) return null;
                const isPlace = !!placeAction;
                const actionId = placeAction ?? reclaimAction!;
                return (
                  <polygon
                    points={getHexPoints(cx, cy, r)}
                    fill={isPlace ? '#38bdf8' : '#4ade80'}
                    fillOpacity={0.3}
                    stroke={isPlace ? '#38bdf8' : '#4ade80'}
                    strokeWidth={2}
                    strokeDasharray="4 2"
                    className="influence-target"
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      submitAction(actionId);
                    }}
                    onMouseEnter={() => {
                      setHoveredSector(sector);
                    }}
                    onMouseLeave={() => {
                      setHoveredSector(null);
                    }}
                  />
                );
              })()}

              {/* ── Build target overlay: clickable sectors ── */}
              {inBuildTargetPhase && (() => {
                const cellIdx = (sector.coords.q + GALAXY_OFFSET) * GALAXY_MAP_SIZE + (sector.coords.r + GALAXY_OFFSET);
                const encodedIdx = (selectedBuildType ?? 0) * ACTION.GALAXY_CELL_COUNT + cellIdx;
                if (!buildTargetCells.has(encodedIdx)) return null;
                const actionId = ACTION.BUILD_CHOICE_START + encodedIdx;
                return (
                  <polygon
                    points={getHexPoints(cx, cy, r)}
                    fill="#f59e0b"
                    fillOpacity={0.35}
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    className="build-target"
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      submitAction(actionId);
                    }}
                    onMouseEnter={() => setHoveredSector(sector)}
                    onMouseLeave={() => setHoveredSector(null)}
                  />
                );
              })()}

              {/* ── Move unit sector highlight: sectors with movable units ── */}
              {inMovePhase && movePhase === 'choose_move' && moveUnitSectors?.has(sector.sector_id) && (
                <polygon
                  points={getHexPoints(cx, cy, r)}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth={2.5}
                  strokeDasharray="5 3"
                  className="move-source"
                  style={{ cursor: 'pointer' }}
                  onClick={(e) => {
                    e.stopPropagation();
                    const movableUnitIndices = moveUnitIndicesBySector?.get(sector.sector_id) ?? [];
                    if (movableUnitIndices.length === 1 && onSelectMoveUnit) {
                      onSelectMoveSector?.(null);
                      onSelectMoveUnit(movableUnitIndices[0]);
                    } else if (movableUnitIndices.length > 1) {
                      onSelectMoveUnit?.(null);
                      onSelectMoveSector?.(sector.sector_id);
                    }
                  }}
                  onMouseEnter={() => setHoveredSector(sector)}
                  onMouseLeave={() => setHoveredSector(null)}
                />
              )}

              {/* ── Move destination highlight: hexes adjacent to selected unit ── */}
              {inMovePhase && movePhase === 'choose_move' && selectedMoveUnitIdx !== null && moveDestCells.size > 0 && (() => {
                const cellKey = `${sector.coords.q},${sector.coords.r}`;
                const target = moveDestCells.get(cellKey);
                if (!target) return null;
                return (
                  <polygon
                    points={getHexPoints(cx, cy, r)}
                    fill={target.isWarp ? '#f59e0b' : '#22c55e'}
                    fillOpacity={0.25}
                    stroke={target.isWarp ? '#f59e0b' : '#22c55e'}
                    strokeWidth={2.5}
                    strokeDasharray={target.isWarp ? '4 3' : undefined}
                    className={`move-target ${target.isWarp ? 'warp-route-target' : ''}`}
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      submitAction(target.actionId);
                    }}
                    onMouseEnter={() => setHoveredSector(sector)}
                    onMouseLeave={() => setHoveredSector(null)}
                  />
                );
              })()}

              {/* ── Warp destination highlight ── */}
              {inMovePhase && movePhase === 'choose_warp_destination' && (() => {
                const cellIdx = (sector.coords.q + GALAXY_OFFSET) * GALAXY_MAP_SIZE + (sector.coords.r + GALAXY_OFFSET);
                const base = ACTION.MOVE_WARP_DESTINATION_START;
                if (!legalActions.includes(base + cellIdx)) return null;
                return (
                  <polygon
                    points={getHexPoints(cx, cy, r)}
                    fill="#f59e0b"
                    fillOpacity={0.3}
                    stroke="#f59e0b"
                    strokeWidth={2.5}
                    strokeDasharray="4 3"
                    className="warp-target"
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      submitAction(base + cellIdx);
                    }}
                    onMouseEnter={() => setHoveredSector(sector)}
                    onMouseLeave={() => setHoveredSector(null)}
                  />
                );
              })()}
            </g>
          );
        })}

        {warpLinks.map((link) => (
          <path
            key={`warp-link-${link.id}`}
            d={warpLinkPath(link)}
            className={`warp-link-path ${emphasizedWarpLink?.id === link.id ? 'selected' : ''}`}
          />
        ))}

        {warpLinks.flatMap((link) => {
          const sourceGate = warpGateLine(link.fromCell, link.fromDir);
          const destGate = warpGateLine(link.toCell, link.toDir);
          const isActive = emphasizedWarpLink?.id === link.id;
          const events = {
            onMouseEnter: () => setHoveredWarpLinkId(link.id),
            onMouseLeave: () => setHoveredWarpLinkId(null),
            onMouseDown: (event: MouseEvent<SVGLineElement>) => {
              event.stopPropagation();
            },
            onClick: (event: MouseEvent<SVGLineElement>) => {
              event.stopPropagation();
              setSelectedWarpLinkId((current) => current === link.id ? null : link.id);
            },
          };
          return [
            <line
              key={`warp-gate-a-${link.id}`}
              {...sourceGate}
              className={`warp-gate ${isActive ? 'selected' : ''}`}
              {...events}
            />,
            <line
              key={`warp-gate-b-${link.id}`}
              {...destGate}
              className={`warp-gate ${isActive ? 'selected' : ''}`}
              {...events}
            />,
          ];
        })}

        {inRotationPreview && previewZone && currentPreviewRotation !== null && (() => {
          const imgUrl = sectorImageUrl(selectedSectorId);
          const r = hexSize - 1.5;
          const imgSize = 2 * r;
          const clipId = `explore-preview-clip-${selectedSectorId}-${previewZone.q}-${previewZone.r}`;
          const imgDeg = IMAGE_ROTATION_OFFSET - 60 * currentPreviewRotation;
          const leftControlX = previewZone.cx - hexSize - 18;
          const rightControlX = previewZone.cx + hexSize + 18;

          return (
            <g className="explore-preview">
              {imgUrl && (
                <>
                  <clipPath id={clipId}>
                    <polygon points={getHexPoints(previewZone.cx, previewZone.cy, r)} />
                  </clipPath>
                  <g clipPath={`url(#${clipId})`}>
                    <image
                      href={imgUrl}
                      x={previewZone.cx - imgSize / 2}
                      y={previewZone.cy - imgSize / 2}
                      width={imgSize}
                      height={imgSize}
                      preserveAspectRatio="xMidYMid slice"
                      transform={`rotate(${imgDeg} ${previewZone.cx} ${previewZone.cy})`}
                      onError={() =>
                        setBrokenImages((prev) => new Set(prev).add(selectedSectorId))
                      }
                      style={{ pointerEvents: 'none' }}
                    />
                  </g>
                </>
              )}

              <polygon
                points={getHexPoints(previewZone.cx, previewZone.cy, r)}
                fill={imgUrl ? 'none' : UNOWNED_COLOR}
                fillOpacity={imgUrl ? 0 : 0.75}
                className="explore-preview-hex"
              />
              <text
                x={previewZone.cx}
                y={previewZone.cy - 6}
                textAnchor="middle"
                className="explore-preview-id"
              >
                {selectedSectorId}
              </text>
              <text
                x={previewZone.cx}
                y={previewZone.cy + 9}
                textAnchor="middle"
                className="explore-preview-rotation"
              >
                rot {currentPreviewRotation}
              </text>

              <g
                className="explore-preview-control"
                onClick={(event) => {
                  event.stopPropagation();
                  cyclePreviewRotation(1);
                }}
              >
                <circle cx={leftControlX} cy={previewZone.cy} r="14" />
                <text x={leftControlX} y={previewZone.cy + 5} textAnchor="middle">&lt;</text>
              </g>
              <g
                className="explore-preview-control"
                onClick={(event) => {
                  event.stopPropagation();
                  cyclePreviewRotation(-1);
                }}
              >
                <circle cx={rightControlX} cy={previewZone.cy} r="14" />
                <text x={rightControlX} y={previewZone.cy + 5} textAnchor="middle">&gt;</text>
              </g>
            </g>
          );
        })()}

        {drawnTilePreviewZone && (() => {
          const sectorId = drawnTilePreviewZone.previewSectorId!;
          const tileIndex = drawnTilePreviewZone.previewTileIndex!;
          const imgUrl = sectorImageUrl(sectorId);
          const r = hexSize - 1.5;
          const imgSize = 2 * r;
          const clipId = `drawn-preview-clip-${sectorId}-${tileIndex}-${drawnTilePreviewZone.q}-${drawnTilePreviewZone.r}`;

          return (
            <g className="drawn-tile-preview">
              {imgUrl && (
                <>
                  <clipPath id={clipId}>
                    <polygon points={getHexPoints(drawnTilePreviewZone.cx, drawnTilePreviewZone.cy, r)} />
                  </clipPath>
                  <g clipPath={`url(#${clipId})`}>
                    <image
                      href={imgUrl}
                      x={drawnTilePreviewZone.cx - imgSize / 2}
                      y={drawnTilePreviewZone.cy - imgSize / 2}
                      width={imgSize}
                      height={imgSize}
                      preserveAspectRatio="xMidYMid slice"
                      style={{ pointerEvents: 'none' }}
                    />
                  </g>
                </>
              )}

              <polygon
                points={getHexPoints(drawnTilePreviewZone.cx, drawnTilePreviewZone.cy, r)}
                fill={imgUrl ? 'none' : UNOWNED_COLOR}
                fillOpacity={imgUrl ? 0 : 0.75}
                className="drawn-tile-preview-hex"
              />
              <text
                x={drawnTilePreviewZone.cx}
                y={drawnTilePreviewZone.cy - 6}
                textAnchor="middle"
                className="drawn-tile-preview-id"
              >
                {sectorId}
              </text>
              <text
                x={drawnTilePreviewZone.cx}
                y={drawnTilePreviewZone.cy + 9}
                textAnchor="middle"
                className="drawn-tile-preview-label"
              >
                Tile {tileIndex + 1}
              </text>
            </g>
          );
        })()}

        {legalZones.map((zone) => (
          <g
            key={`zone-${zone.action}`}
            className="explore-zone"
            onClick={() => submitAction(zone.action)}
          >
            <polygon
              points={getHexPoints(zone.cx, zone.cy, hexSize - 1.5)}
              className="explore-zone-hex"
            />
            <text
              x={zone.cx}
              y={zone.cy + 4}
              textAnchor="middle"
              fill="#a7f3d0"
              fontSize="14px"
              fontWeight="bold"
              style={{ pointerEvents: 'none' }}
            >
              +
            </text>
          </g>
        ))}

        {shipRenderItems.map((ship) => {
          const imageUrl = shipImageUrl(String(ship.unit.type));
          const isStack = ship.members.length > 1;
          const half = ship.size / 2;
          const outlineSize = ship.size + SHIP_OUTLINE * 2;
          const outlineHalf = outlineSize / 2;
          const shipMaskId = `ship-mask-${ship.sectorId}-${ship.globalIdx}`;
          const outlineMaskId = `ship-outline-mask-${ship.sectorId}-${ship.globalIdx}`;
          const damage = ship.unit.damage ?? 0;
          const badgeText = String(ship.members.length);
          const badgeWidth = Math.max(10, 5 + badgeText.length * 5);
          const badgeX = ship.x + half - 2;
          const badgeY = ship.y - half - 8;

          return (
            <g
              key={ship.key}
              className="map-ship"
              style={{ pointerEvents: 'none' }}
            >
              <mask
                id={outlineMaskId}
                x={ship.x - outlineHalf}
                y={ship.y - outlineHalf}
                width={outlineSize}
                height={outlineSize}
                maskUnits="userSpaceOnUse"
              >
                <image
                  href={imageUrl}
                  x={ship.x - outlineHalf}
                  y={ship.y - outlineHalf}
                  width={outlineSize}
                  height={outlineSize}
                  preserveAspectRatio="xMidYMid meet"
                />
              </mask>
              <rect
                x={ship.x - outlineHalf}
                y={ship.y - outlineHalf}
                width={outlineSize}
                height={outlineSize}
                fill="#020617"
                fillOpacity={0.95}
                mask={`url(#${outlineMaskId})`}
              />

              <mask
                id={shipMaskId}
                x={ship.x - half}
                y={ship.y - half}
                width={ship.size}
                height={ship.size}
                maskUnits="userSpaceOnUse"
              >
                <image
                  href={imageUrl}
                  x={ship.x - half}
                  y={ship.y - half}
                  width={ship.size}
                  height={ship.size}
                  preserveAspectRatio="xMidYMid meet"
                />
              </mask>
              <rect
                x={ship.x - half}
                y={ship.y - half}
                width={ship.size}
                height={ship.size}
                fill={ship.color}
                mask={`url(#${shipMaskId})`}
              />

              {!isStack && damage > 0 && Array.from({ length: Math.min(damage, 4) }, (_, damageIdx) => (
                <rect
                  key={damageIdx}
                  x={ship.x - half + damageIdx * 3.5}
                  y={ship.y - half - 4.5}
                  width={3}
                  height={3}
                  rx={0.5}
                  fill="#ef4444"
                  stroke="#020617"
                  strokeWidth={0.4}
                />
              ))}

              {isStack && (
                <g style={{ pointerEvents: 'none' }}>
                  <rect
                    x={badgeX}
                    y={badgeY}
                    width={badgeWidth}
                    height={11}
                    rx={2}
                    fill="#020617"
                    stroke="#f8fafc"
                    strokeWidth={0.75}
                  />
                  <text
                    x={badgeX + badgeWidth / 2}
                    y={badgeY + 8}
                    textAnchor="middle"
                    fill="#f8fafc"
                    fontSize="9px"
                    fontWeight="bold"
                  >
                    {badgeText}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        </g>
      </svg>

      {/* Zoom controls */}
      <div className="zoom-controls">
        <button className="zoom-btn" onClick={zoomIn} title="Zoom in">+</button>
        <button className="zoom-btn" onClick={zoomOut} title="Zoom out">-</button>
        <button className="zoom-btn" onClick={resetView} title="Reset view" style={{ fontSize: '11px' }}>⟲</button>
      </div>

      {/* Hover info overlay */}
      {hoveredSector && (
        <div className="galaxy-info-overlay">
          <div className="galaxy-info-item">
            <span>Sector ID</span>
            <span>{hoveredSector.sector_id === 1 ? '001 (Center)' : hoveredSector.sector_id}</span>
          </div>
          <div className="galaxy-info-item">
            <span>Coordinates</span>
            <span>({hoveredSector.coords.q}, {hoveredSector.coords.r})</span>
          </div>
          <div className="galaxy-info-item">
            <span>Owner</span>
            <span>
              {hoveredSector.owner_id === NPC_PLAYER_ID
                ? 'Unowned (Neutral)'
                : `${playerLabel(hoveredSector.owner_id)} (${gameState.players[hoveredSector.owner_id]?.species_id})`}
            </span>
          </div>
          <div className="galaxy-info-item">
            <span>Sector Points</span>
            <span>⭐️ {hoveredSector.points}</span>
          </div>
          {getUnitsInSector(hoveredSector.sector_id).length > 0 && (
            <div className="galaxy-info-item">
              <span>Active Units</span>
              <span>
                {getUnitsInSector(hoveredSector.sector_id)
                  .map((unit) => `${unit.type} (P${unit.player_id})`)
                  .join(', ')}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
