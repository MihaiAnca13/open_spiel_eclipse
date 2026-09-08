import type { GameState } from '../../types/game';
import { getPlayerColor } from '../../theme';
import { finalStandings } from '../../utils/standings';
import ScoreBreakdown from '../ui/ScoreBreakdown';

interface EndgameScreenProps {
  gameState: GameState;
  playerLabel: (playerId: number) => string;
}

export default function EndgameScreen({ gameState, playerLabel }: EndgameScreenProps) {
  const standings = finalStandings(gameState.players, gameState.scores ?? {});
  const winner = standings[0];
  const sharedWin = standings.length > 1 &&
    winner.score.total_vp === standings[1].score.total_vp &&
    winner.resourceTotal === standings[1].resourceTotal;

  return (
    <div className="endgame-overlay">
      <section className="endgame-screen" aria-label="Final standings">
        <header className="endgame-header">
          <p>Game Over</p>
          <h1>{sharedWin ? 'Co-victory' : `${playerLabel(winner.player.id)} wins`}</h1>
          <span>
            {sharedWin
              ? 'Players are tied on Victory Points and stored resources.'
              : `${winner.score.total_vp} VP · ${winner.resourceTotal} stored resources`}
          </span>
        </header>

        <div className="endgame-standings">
          {standings.map((standing, index) => {
            const next = standings[index + 1];
            const sharesRank = standing.isTied || (
              next !== undefined &&
              standing.score.total_vp === next.score.total_vp &&
              standing.resourceTotal === next.resourceTotal
            );
            return (
              <article className="endgame-player" key={standing.player.id}>
                <header>
                  <span className="endgame-rank">{sharesRank ? `T-${standing.rank}` : `#${standing.rank}`}</span>
                  <div>
                    <h2 style={{ color: getPlayerColor(standing.player.id) }}>{playerLabel(standing.player.id)}</h2>
                    <p>{standing.player.species_id}</p>
                  </div>
                  <strong>{standing.score.total_vp} VP</strong>
                </header>
                <p className="endgame-tiebreak">
                  Resources: {standing.player.resources.gold} 💰 · {standing.player.resources.science} 🔬 · {standing.player.resources.materials} ⚙️ = {standing.resourceTotal}
                </p>
                <ScoreBreakdown score={standing.score} className="endgame-score-breakdown" />
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
