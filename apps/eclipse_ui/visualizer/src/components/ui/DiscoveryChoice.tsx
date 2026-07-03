import { useMemo, useState } from 'react';
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
  const [brokenRewardSlug, setBrokenRewardSlug] = useState<string | null>(null);

  const canTakeReward = legal.has(rewardActionId);
  const canTakeVp = legal.has(vpActionId);
  const hasBrokenRewardImage = !!tile && brokenRewardSlug === tile.slug;

  if (!tile || hasBrokenRewardImage) {
    return (
      <div className="action-row">
        {canTakeReward && (
          <button className="action-btn primary" disabled={busy} onClick={() => onAction(rewardActionId)}>
            Take reward
          </button>
        )}
        {canTakeVp && (
          <button className="action-btn secondary" disabled={busy} onClick={() => onAction(vpActionId)}>
            Take 2 VP
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="discovery-choice-grid">
      {canTakeReward && (
        <button
          type="button"
          className="discovery-choice-card reward"
          disabled={busy}
          onClick={() => onAction(rewardActionId)}
          title={`Take ${tile.name}`}
        >
          <span className="discovery-choice-frame">
            <img
              className="discovery-choice-img"
              src={discoveryRewardImageUrl(tile.slug)}
              alt={tile.name}
              onError={() => setBrokenRewardSlug(tile.slug)}
            />
          </span>
          <span className="discovery-choice-label">{tile.name}</span>
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
