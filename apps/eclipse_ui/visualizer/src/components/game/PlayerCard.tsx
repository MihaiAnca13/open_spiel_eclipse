import type { GameMetadata, Player, PlayerScoreBreakdown } from '../../types/game';
import { getPlayerColor } from '../../theme';
import { POPULATION_PRODUCTION_TABLE, POP_TRACK_MAX, INFLUENCE_TOTAL } from '../../utils/game';
import PopulationTracks from '../ui/PopulationTracks';
import InfluenceTrack from '../ui/InfluenceTrack';
import ColonyShips from '../ui/ColonyShips';
import TradePanel from '../ui/TradePanel';
import ResearchTracks from '../../ResearchTracks';

interface PlayerCardProps {
  player: Player;
  playerId: number;
  currentPlayerId: number;
  mySeatIdx: number;
  playerLabel: (pid: number) => string;
  colonyShipPlacements: { actionId: number; sectorId: number; slotIdx: number; track: number }[];
  legalTradeActions: number[];
  submitAction: (actionId: number) => void;
  gameMetadata: GameMetadata;
  scoreBreakdown?: PlayerScoreBreakdown;
}

export default function PlayerCard({
  player,
  playerId,
  currentPlayerId,
  mySeatIdx,
  playerLabel,
  colonyShipPlacements,
  legalTradeActions,
  submitAction,
  gameMetadata,
  scoreBreakdown,
}: PlayerCardProps) {
  const onSectors = player.disks_on_sectors ?? 0;
  const onActions = player.disks_on_actions ?? 0;
  const discsLeft = Math.max(0, INFLUENCE_TOTAL - onSectors - onActions);
  const isMine = playerId === mySeatIdx;
  const isActive = playerId === currentPlayerId;

  return (
    <div className={`economy-card ${isMine ? 'mine' : ''} ${isActive ? 'active' : ''}`}>
      <div className="economy-name">
        <span style={{ color: getPlayerColor(playerId) }}>{playerLabel(playerId)}</span>
        <span className="economy-score" title={scoreBreakdown ? "Hover for VP Breakdown" : undefined}>
          ⭐ {player.score}
          {scoreBreakdown && (
            <div className="score-tooltip">
              <div className="score-tooltip-row"><span>Reputation:</span> <span>+{scoreBreakdown.reputation_vp}</span></div>
              <div className="score-tooltip-row"><span>Ambassadors:</span> <span>+{scoreBreakdown.ambassador_vp}</span></div>
              <div className="score-tooltip-row"><span>Sectors:</span> <span>+{scoreBreakdown.sector_vp}</span></div>
              <div className="score-tooltip-row"><span>Monoliths:</span> <span>+{scoreBreakdown.monolith_vp}</span></div>
              <div className="score-tooltip-row"><span>Discoveries:</span> <span>+{scoreBreakdown.discovery_vp}</span></div>
              <div className="score-tooltip-row"><span>Tech tracks:</span> <span>+{scoreBreakdown.tech_track_vp}</span></div>
              <div className="score-tooltip-row"><span>Traitor:</span> <span>{scoreBreakdown.traitor_vp}</span></div>
              <div className="score-tooltip-row"><span>Species:</span> <span>+{scoreBreakdown.species_vp}</span></div>
              <div className="score-tooltip-row score-tooltip-total"><span>Total VP:</span> <span>{scoreBreakdown.total_vp}</span></div>
            </div>
          )}
        </span>
      </div>
      <div className="economy-resources">
        <span className="res gold" title="Money">
          💰 {player.resources.gold}
          <em> +{POPULATION_PRODUCTION_TABLE[Math.min(player.resources.gold_prod, POP_TRACK_MAX)]}</em>
        </span>
        <span className="res science" title="Science">
          🔬 {player.resources.science}
          <em> +{POPULATION_PRODUCTION_TABLE[Math.min(player.resources.science_prod, POP_TRACK_MAX)]}</em>
        </span>
        <span className="res materials" title="Materials">
          ⚙️ {player.resources.materials}
          <em> +{POPULATION_PRODUCTION_TABLE[Math.min(player.resources.materials_prod, POP_TRACK_MAX)]}</em>
        </span>
      </div>
      <PopulationTracks resources={player.resources} playerColor={getPlayerColor(playerId)} />
      <ResearchTracks
        militaryMask={player.researched_techs_military}
        gridMask={player.researched_techs_grid}
        nanoMask={player.researched_techs_nano}
        techCatalog={gameMetadata.tech_catalog ?? {}}
      />
      {isMine ? (
        <>
          <InfluenceTrack onSectors={onSectors} onActions={onActions} />
          <ColonyShips
            total={player.colony_ships_total}
            available={player.colony_ships_available}
            legalPlacements={colonyShipPlacements}
            onPlace={submitAction}
          />
          <TradePanel
            tradeRate={player.trade_rate}
            legalTradeActions={legalTradeActions}
            onTrade={submitAction}
            resources={player.resources}
          />
        </>
      ) : (
        <div className="economy-meta">
          <span title="Influence discs available">🔵 {discsLeft} discs</span>
          {player.colony_ships_total > 0 && (
            <span title="Colony ships">
              ◎ {player.colony_ships_available}/{player.colony_ships_total}
            </span>
          )}
        </div>
      )}
      {player.has_passed && <div className="economy-meta"><span className="economy-passed">passed</span></div>}
    </div>
  );
}
