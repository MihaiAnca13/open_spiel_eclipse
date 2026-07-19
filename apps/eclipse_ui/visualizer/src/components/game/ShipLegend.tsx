import { shipImageUrl } from '../../types/lobby';

const SHIP_TYPES: { name: string; type: string; color?: string }[] = [
  { name: 'Interceptor', type: 'interceptor' },
  { name: 'Cruiser', type: 'cruiser' },
  { name: 'Dreadnought', type: 'dreadnought' },
  { name: 'Starbase', type: 'starbase' },
  { name: 'Ancient', type: 'ancient', color: '#94a3b8' },
  { name: 'Guardian', type: 'guardian', color: '#7c2d12' },
  { name: 'GCDS', type: 'gcds', color: '#3b0764' },
];

interface ShipLegendProps {
  onClose?: () => void;
}

export default function ShipLegend({ onClose }: ShipLegendProps) {
  return (
    <div className="ship-legend-overlay">
      <div className="ship-legend-header">
        <span className="ship-legend-title">Ship icons</span>
        {onClose && (
          <button
            className="ship-legend-close"
            onClick={onClose}
            aria-label="Hide legend"
          >
            ×
          </button>
        )}
      </div>
      <div className="ship-legend-list">
        {SHIP_TYPES.map(({ name, type, color }) => (
          <div key={type} className="ship-legend-row">
            <img
              className="ship-legend-icon"
              src={shipImageUrl(type)}
              alt={name}
              style={{ borderColor: color ?? '#e2e8f0' }}
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
            <span className="ship-legend-name">{name}</span>
          </div>
        ))}
      </div>
      <p className="ship-legend-note">
        Player ships use the owner’s seat color; NPC ships use faction colors.
      </p>
    </div>
  );
}
