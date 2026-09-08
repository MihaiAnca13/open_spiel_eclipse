import type { Player, PlayerScoreBreakdown } from '../types/game';

export interface FinalStanding {
  player: Player;
  score: PlayerScoreBreakdown;
  resourceTotal: number;
  rank: number;
  isTied: boolean;
}

export function finalStandings(
  players: Player[],
  scores: Record<string, PlayerScoreBreakdown>,
): FinalStanding[] {
  const sorted = players
    .map((player) => ({
      player,
      score: scores[String(player.id)],
      resourceTotal: player.resources.gold + player.resources.science + player.resources.materials,
    }))
    .sort((left, right) =>
      right.score.total_vp - left.score.total_vp ||
      right.resourceTotal - left.resourceTotal ||
      left.player.id - right.player.id,
    );

  let currentRank = 0;
  return sorted.map((standing, index) => {
    const previous = index > 0 ? sorted[index - 1] : undefined;
    const isTied = previous !== undefined &&
      standing.score.total_vp === previous.score.total_vp &&
      standing.resourceTotal === previous.resourceTotal;
    if (!isTied) currentRank = index + 1;
    return {
      ...standing,
      rank: currentRank,
      isTied,
    };
  });
}
