import { useMemo, useState } from 'react';
import type { GameState, Sector, SectorLayout } from '../../types/game';
import { ACTION } from '../../actionTypes';
import { UNOWNED_COLOR } from '../../theme';
import { getPlayerHexColor } from '../../utils/game';
import { axialToPixel, getHexPoints, hexSize, IMAGE_ROTATION_OFFSET } from '../../utils/hex';
import { getPlayerColor } from '../../theme';

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
}: GalaxyMapProps) {
  const [hoveredSector, setHoveredSector] = useState<Sector | null>(null);
  const [brokenImages, setBrokenImages] = useState<Set<number>>(new Set());

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

  const inZoneSelect = isMyTurn && explorePhase === 'choose_zone';

  // Legal explore zones → clickable hexes (action id 35 + hex_to_index(q,r)).
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

  // Floating preview for drawn tiles during select_drawn_tile phase.
  const drawnTilePreviewZone = inSelectDrawnTile && exploreState && previewingDrawnTile !== null && previewingDrawnTileIndex !== null
    ? {
        q: exploreState.zone_q,
        r: exploreState.zone_r,
        ...axialToPixel(exploreState.zone_q, exploreState.zone_r),
        previewSectorId: previewingDrawnTile,
        previewTileIndex: previewingDrawnTileIndex,
      }
    : null;

  // Auto-fit the SVG viewBox to the populated region (sectors + explore zones)
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
      <svg width="100%" height="100%" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
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
                  <circle r="7" fill="#dc2626" stroke="#ffffff" strokeWidth="1" />
                  <text y="2.5" textAnchor="middle" fill="#ffffff" fontSize="8px" fontWeight="bold">
                    {units.length}
                  </text>
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
      </svg>

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
