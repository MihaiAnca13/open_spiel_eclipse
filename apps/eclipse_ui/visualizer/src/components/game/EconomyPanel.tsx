import type { GameMetadata, GameState } from '../../types/game';
import PlayerCard from './PlayerCard';
import { NPC_PLAYER_ID } from '../../constants';

interface EconomyPanelProps {
  gameState: GameState;
  mySeatIdx: number;
  playerLabel: (pid: number) => string;
  colonyShipPlacements: { actionId: number; sectorId: number; slotIdx: number; track: number }[];
  legalTradeActions: number[];
  submitAction: (actionId: number) => void;
  gameMetadata: GameMetadata;
}

export default function EconomyPanel({
  gameState,
  mySeatIdx,
  playerLabel,
  colonyShipPlacements,
  legalTradeActions,
  submitAction,
  gameMetadata,
}: EconomyPanelProps) {
  return (
    <div className="panel economy-panel">
      <h3 className="panel-title">Players</h3>
      <div className="economy-grid">
        {gameState.turn_order
          .filter((pid) => pid !== NPC_PLAYER_ID)
          .map((pid) => {
            const player = gameState.players[pid];
            if (!player) return null;
            return (
              <PlayerCard
                key={pid}
                player={player}
                playerId={pid}
                currentPlayerId={gameState.current_player}
                mySeatIdx={mySeatIdx}
                playerLabel={playerLabel}
                colonyShipPlacements={colonyShipPlacements}
                legalTradeActions={legalTradeActions}
                submitAction={submitAction}
                gameMetadata={gameMetadata}
                scoreBreakdown={gameState.scores?.[pid]}
              />
            );
          })}
      </div>
    </div>
  );
}
