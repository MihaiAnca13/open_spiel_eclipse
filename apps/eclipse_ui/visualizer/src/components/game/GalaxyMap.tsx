import { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import type { GameState, Sector, SectorLayout } from '../../types/game';
import { ACTION } from '../../actionTypes';
import { UNOWNED_COLOR } from '../../theme';
import { getPlayerHexColor } from '../../utils/game';
import { axialToPixel, getHexPoints, hexSize, IMAGE_ROTATION_OFFSET, axialNeighbor } from '../../utils/hex';
import { getPlayerColor } from '../../theme';
import { shipImageUrl } from '../../types/lobby';


const INITIAL_SCALE = 1;

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
  combatActiveSectorId?: number;
  combatRetreatActions?: number[];
  combatTargetActions?: number[];
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
  combatActiveSectorId = 0,
  combatRetreatActions = [],
  combatTargetActions = [],
}: GalaxyMapProps) {
  const [hoveredSector, setHoveredSector] = useState<Sector | null>(null);
  const [brokenImages, setBrokenImages] = useState<Set<number>>(new Set());

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

  // Combat: retreat-destination cells (cell index -> actionId) and targetable
  // ships (global unit_registry index -> actionId).
  const combatRetreatCells = useMemo(() => {
    const map = new Map<number, number>();
    for (const a of combatRetreatActions) map.set(a - ACTION.COMBAT_RETREAT_TO_CELL_START, a);
    return map;
  }, [combatRetreatActions]);
  const combatTargetUnits = useMemo(() => {
    const map = new Map<number, number>();
    for (const a of combatTargetActions) map.set(a - ACTION.COMBAT_DICE_TARGET_START, a);
    return map;
  }, [combatTargetActions]);
  const inCombat = combatActiveSectorId > 0;

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

  const handleMouseUp = useCallback(() => { isPanning.current = false; }, []);

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

  const getUnitsInSector = (sectorId: number) => {
    if (!gameState) return [];
    return gameState.unit_registry.filter((unit) => unit.sector_id === sectorId);
  };

  const activeSectors: { sector: Sector; cx: number; cy: number }[] = useMemo(() => {
    const list: { sector: Sector; cx: number; cy: number }[] = [];
    if (gameState?.galaxy) {
      const mapSize = gameState.galaxy.length;
      const offset = 7;
      for (let qIdx = 0; qIdx < mapSize; qIdx++) {
        for (let rIdx = 0; rIdx < mapSize; rIdx++) {
          const sector = gameState.galaxy[qIdx][rIdx];
          if (sector && sector.sector_id > 0) {
            const q = qIdx - offset;
            const r = rIdx - offset;
            const { cx, cy } = axialToPixel(q, r);
            list.push({ sector, cx, cy });
          }
        }
      }
    }
    return list;
  }, [gameState]);

  // For move targets: compute destination cell for each direction
  const moveDestCells = useMemo(() => {
    const map = new Map<string, number>();
    if (!moveTargetCells || selectedMoveUnitIdx == null || !gameState) return map;
    const idx = selectedMoveUnitIdx;
    const unit = gameState.unit_registry[idx];
    if (!unit) return map;
    const sectorEntry = activeSectors.find(s => s.sector.sector_id === unit.sector_id);
    if (!sectorEntry) return map;
    const { q, r } = sectorEntry.sector.coords;
    for (const [dir, actionId] of moveTargetCells) {
      if (dir < 6) {
        const [nq, nr] = axialNeighbor(q, r, dir);
        map.set(`${nq},${nr}`, actionId);
      }
    }
    return map;
  }, [moveTargetCells, selectedMoveUnitIdx, gameState, activeSectors]);

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
            const q = Math.floor(idx / 15) - 7;
            const r = (idx % 15) - 7;
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

  const previewZone =
    inRotationPreview && exploreState
      ? {
          q: exploreState.zone_q,
          r: exploreState.zone_r,
          ...axialToPixel(exploreState.zone_q, exploreState.zone_r),
        }
      : null;

  const drawnTilePreviewZone = inSelectDrawnTile && exploreState && previewingDrawnTile !== null && previewingDrawnTileIndex !== null
    ? {
        q: exploreState.zone_q,
        r: exploreState.zone_r,
        ...axialToPixel(exploreState.zone_q, exploreState.zone_r),
        previewSectorId: previewingDrawnTile,
        previewTileIndex: previewingDrawnTileIndex,
      }
    : null;

  const transformStr = `translate(${viewTransform.x},${viewTransform.y}) scale(${viewTransform.scale})`;

  const viewBox = useMemo(() => {
    const fitCells = previewZone
      ? [...activeSectors, ...legalZones, previewZone]
      : drawnTilePreviewZone
        ? [...activeSectors, ...legalZones, drawnTilePreviewZone]
        : [...activeSectors, ...legalZones];

    if (fitCells.length === 0) return '0 0 600 520';
    const xs = fitCells.map((c) => c.cx);
    const ys = fitCells.map((c) => c.cy);
    const pad = hexSize * 1.4;
    const minX = Math.min(...xs) - hexSize - pad;
    const minY = Math.min(...ys) - hexSize - pad;
    const w = Math.max(...xs) - Math.min(...xs) + 2 * (hexSize + pad);
    const h = Math.max(...ys) - Math.min(...ys) + 2 * (hexSize + pad);
    return `${minX} ${minY} ${w} ${h}`;
  }, [activeSectors, legalZones, previewZone, drawnTilePreviewZone]);

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
        style={{ cursor: isPanning.current ? 'grabbing' : 'grab' }}
      >
        <g transform={transformStr}>
          {activeSectors.map(({ sector, cx, cy }) => {
          const fillColor = getPlayerHexColor(sector.owner_id, sector.sector_id);
          const isCenter = sector.sector_id === 1;
          const units = getUnitsInSector(sector.sector_id);
          const owned = sector.owner_id !== 255;
          const hasHostiles = units.some((u) => u.player_id === 255);
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

              {units.length > 0 && (
  <g transform={`translate(${cx}, ${cy + 18})`}>
    {units.map((unit, ui) => {
      const uColor = getPlayerColor(unit.player_id);
      const globalIdx = gameState!.unit_registry.indexOf(unit);
      const targetAction = combatTargetUnits.get(globalIdx);
      const isTargetable = targetAction !== undefined;
      const damage = unit.damage ?? 0;
      return (
        <g
          key={ui}
          transform={`translate(${ui * 12}, 0)`}
          style={{ cursor: isTargetable ? 'pointer' : 'default' }}
          onClick={isTargetable ? (e) => { e.stopPropagation(); submitAction(targetAction!); } : undefined}
        >
          {isTargetable && (
            <circle cx={0} cy={0} r={9} fill="#ef4444" fillOpacity={0.25} stroke="#ef4444" strokeWidth={1.2}>
              <animate attributeName="r" values="8;10;8" dur="1s" repeatCount="indefinite" />
            </circle>
          )}
          <image
            href={shipImageUrl(String(unit.type))}
            width={14}
            height={14}
            x={-7}
            y={-7}
            style={{ opacity: 0.9, pointerEvents: isTargetable ? 'all' : 'none' }}
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
          />
          <circle cx={5} cy={5} r={3} fill={uColor} stroke="#fff" strokeWidth={0.5} />
          {/* Damage cubes */}
          {damage > 0 && Array.from({ length: Math.min(damage, 4) }, (_, di) => (
            <rect key={di} x={-7 + di * 3.5} y={-11} width={3} height={3} rx={0.5} fill="#ef4444" stroke="#000" strokeWidth={0.3} />
          ))}
        </g>
      );
    })}
  </g>
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
                const mapSize = gameState?.galaxy.length ?? 15;
                const cellIdx = (sector.coords.q + 7) * mapSize + (sector.coords.r + 7);
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

              {/* ── Combat: battle sector outline ── */}
              {inCombat && sector.sector_id === combatActiveSectorId && (
                <polygon
                  points={getHexPoints(cx, cy, r)}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth={3}
                  className="combat-battle"
                  style={{ pointerEvents: 'none' }}
                >
                  <animate attributeName="stroke-opacity" values="1;0.4;1" dur="1.2s" repeatCount="indefinite" />
                </polygon>
              )}

              {/* ── Combat: retreat-destination hexes (clickable) ── */}
              {inCombat && combatRetreatCells.size > 0 && (() => {
                const mapSize = gameState?.galaxy.length ?? 15;
                const cellIdx = (sector.coords.q + 7) * mapSize + (sector.coords.r + 7);
                const action = combatRetreatCells.get(cellIdx);
                if (action === undefined) return null;
                return (
                  <polygon
                    points={getHexPoints(cx, cy, r)}
                    fill="#22c55e"
                    fillOpacity={0.25}
                    stroke="#22c55e"
                    strokeWidth={2.5}
                    strokeDasharray="5 3"
                    className="combat-retreat"
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => { e.stopPropagation(); submitAction(action); }}
                    onMouseEnter={() => setHoveredSector(sector)}
                    onMouseLeave={() => setHoveredSector(null)}
                  />
                );
              })()}

              {/* ── Build target overlay: clickable sectors ── */}
              {inBuildTargetPhase && (() => {
                const mapSize = gameState?.galaxy.length ?? 15;
                const cellIdx = (sector.coords.q + 7) * mapSize + (sector.coords.r + 7);
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
                const targetAction = moveDestCells.get(cellKey);
                if (!targetAction) return null;
                return (
                  <polygon
                    points={getHexPoints(cx, cy, r)}
                    fill="#22c55e"
                    fillOpacity={0.25}
                    stroke="#22c55e"
                    strokeWidth={2.5}
                    className="move-target"
                    style={{ cursor: 'pointer' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      submitAction(targetAction);
                    }}
                    onMouseEnter={() => setHoveredSector(sector)}
                    onMouseLeave={() => setHoveredSector(null)}
                  />
                );
              })()}

              {/* ── Warp destination highlight ── */}
              {inMovePhase && movePhase === 'choose_warp_destination' && (() => {
                const mapSize = gameState?.galaxy.length ?? 15;
                const cellIdx = (sector.coords.q + 7) * mapSize + (sector.coords.r + 7);
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
              {hoveredSector.owner_id === 255
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
