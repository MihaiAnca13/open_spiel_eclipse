import { ACTION } from '../../actionTypes';
import { shipImageUrl } from '../../types/lobby';
import type { BuildCosts } from '../../types/game';

interface Props {
  legalActions: number[];
  busy: boolean;
  onAction: (actionId: number) => void;
  activationsRemaining: number;
  selectedBuildType: number | null;
  onSelectBuildType: (type: number | null) => void;
  builtShipsCount: number;
  buildCosts?: BuildCosts | null;
}

const BUILD_TYPE_NAMES = ['Interceptor', 'Cruiser', 'Dreadnought', 'Starbase', 'Orbital', 'Monolith'] as const;

const BUILD_TYPE_ICONS: Record<string, string> = {
  Interceptor: '🚀',
  Cruiser: '🛡️',
  Dreadnought: '💥',
  Starbase: '🏰',
  Orbital: '🛰️',
  Monolith: '🗿',
};

const DEFAULT_BUILD_COST: Record<string, number> = {
  Interceptor: 3,
  Cruiser: 5,
  Dreadnought: 8,
  Starbase: 3,
  Orbital: 4,
  Monolith: 10,
};

export default function BuildPanel({
  legalActions,
  busy,
  onAction,
  activationsRemaining,
  selectedBuildType,
  onSelectBuildType,
  builtShipsCount,
  buildCosts,
}: Props) {
  const legal = new Set(legalActions);

  const canStop = legal.has(ACTION.BUILD_STOP);

  // Compute which build types have at least one valid target
  const availableTypes = new Set<number>();
  for (const actionId of legalActions) {
    if (actionId >= ACTION.BUILD_CHOICE_START && actionId < ACTION.BUILD_END) {
      const encoded = actionId - ACTION.BUILD_CHOICE_START;
      availableTypes.add(Math.floor(encoded / ACTION.GALAXY_CELL_COUNT));
    }
  }
  const typeList = [...availableTypes].sort();

  const anyAvailable = typeList.length > 0;

  return (
    <div className="build-panel">
      <div className="build-header">
        <span className="text-xs text-[#94a3b8]">
          {selectedBuildType === null
            ? `Select a ship or structure to build.${activationsRemaining > 1 ? ` Activations left: ${activationsRemaining}.` : ''}`
            : `Click a highlighted sector on the map to place ${BUILD_TYPE_NAMES[selectedBuildType]}.${activationsRemaining > 1 ? ` Activations left: ${activationsRemaining}.` : ''}`
          }
        </span>
      </div>

      {!anyAvailable && !canStop && (
        <span className="text-xs text-[#fca5a5]">No valid builds available (check materials, tech, and supply).</span>
      )}

      {selectedBuildType === null ? (
        /* Step 1: pick ship type */
        <div className="build-types">
          {typeList.map((t) => (
            <div key={t} className="build-type-card" style={{ cursor: 'pointer' }} onClick={() => !busy && onSelectBuildType(t)}>
              <div className="build-type-header">
                <img
                  className="build-ship-img"
                  src={shipImageUrl(BUILD_TYPE_NAMES[t])}
                  alt={BUILD_TYPE_NAMES[t]}
                  onError={(e) => { e.currentTarget.style.display = 'none'; }}
                />
                <span className="build-type-icon">{BUILD_TYPE_ICONS[BUILD_TYPE_NAMES[t]] || '?'}</span>
                <span className="build-type-name">{BUILD_TYPE_NAMES[t]}</span>
                <span className="build-type-cost">⚙️ {buildCosts?.[BUILD_TYPE_NAMES[t]] ?? DEFAULT_BUILD_COST[BUILD_TYPE_NAMES[t]]}</span>
              </div>
            </div>
          ))}
          <div className="action-row" style={{ marginTop: '8px' }}>
            {canStop && (
              <button
                className="action-btn secondary"
                style={{ padding: '5px 10px', fontSize: '11px', flex: 'none' }}
                disabled={busy}
                onClick={() => onAction(ACTION.BUILD_STOP)}
              >
                Stop building
              </button>
            )}
          </div>
        </div>
      ) : (
        /* Step 2: waiting for map click */
        <div className="build-waiting-sectors">
          {builtShipsCount > 0 && (
            <div className="text-xs text-[#94a3b8] mb-1">
              Ships built this round: {builtShipsCount}
            </div>
          )}
          {!anyAvailable && (
            <span className="text-xs text-[#fca5a5]">No valid sectors for this ship type.</span>
          )}
          <div className="action-row" style={{ marginTop: '6px' }}>
            <button
              className="action-btn secondary"
              style={{ padding: '5px 10px', fontSize: '11px', flex: 'none' }}
              disabled={busy}
              onClick={() => onSelectBuildType(null)}
            >
              Change ship type
            </button>
            {canStop && (
              <button
                className="action-btn secondary"
                style={{ padding: '5px 10px', fontSize: '11px', flex: 'none' }}
                disabled={busy}
                onClick={() => onAction(ACTION.BUILD_STOP)}
              >
                Stop building
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
