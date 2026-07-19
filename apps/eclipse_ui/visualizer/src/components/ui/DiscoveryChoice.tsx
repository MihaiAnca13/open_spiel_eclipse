import { useEffect, useMemo, useState } from 'react';
import { discoveryRewardImageUrl } from '../../types/lobby';
import type { DiscoveryTileDefinition } from '../../types/game';

interface Props {
  tile?: DiscoveryTileDefinition;
  legalActions: number[];
  busy: boolean;
  rewardActionId: number;
  vpActionId: number;
  onAction: (actionId: number) => void;
}

export default function DiscoveryChoice({
  tile,
  legalActions,
  busy,
  rewardActionId,
  vpActionId,
  onAction,
}: Props) {
  const legal = useMemo(() => new Set(legalActions), [legalActions]);
  const [rewardImageError, setRewardImageError] = useState(false);

  useEffect(() => {
    setRewardImageError(false);
  }, [tile?.slug]);

  const canTakeReward = legal.has(rewardActionId);
  const canTakeVp = legal.has(vpActionId);

  return (
    <div className="discovery-choice-grid">
      {canTakeReward && (
        <button
          type="button"
          className="discovery-choice-card reward"
          disabled={busy}
          onClick={() => onAction(rewardActionId)}
          title={`Take ${tile?.name ?? 'reward'}`}
        >
          <span className="discovery-choice-frame">
            {tile && !rewardImageError ? (
              <img
                className="discovery-choice-img"
                src={discoveryRewardImageUrl(tile.slug)}
                alt={tile.name}
                onError={() => setRewardImageError(true)}
              />
            ) : (
              <span className="discovery-choice-placeholder" aria-hidden="true">?</span>
            )}
          </span>
          <span className="discovery-choice-label">{tile?.name ?? 'Take reward'}</span>
        </button>
      )}
      {canTakeVp && (
        <button
          type="button"
          className="discovery-choice-card vp"
          disabled={busy}
          onClick={() => onAction(vpActionId)}
          title="Keep the Discovery tile for 2 VP"
        >
          <span className="discovery-choice-frame vp-frame" aria-hidden="true">
            <span className="discovery-choice-vp-mark">
              <span className="discovery-choice-vp-number">2</span>
              <span className="discovery-choice-vp-text">VP</span>
            </span>
          </span>
          <span className="discovery-choice-label">Keep 2 VP</span>
        </button>
      )}
    </div>
  );
}
