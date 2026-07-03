import { useEffect, useMemo, useState } from 'react';
import { ACTION } from '../../actionTypes';
import { shipImageUrl } from '../../types/lobby';
import type { Blueprint, Player, ShipPartCatalog, ShipPartDefinition, ShipStats, UpgradeState } from '../../types/game';

const SHIP_NAMES = ['Interceptor', 'Cruiser', 'Dreadnought', 'Starbase'] as const;

interface UpgradeChoice {
  actionId: number;
  ship: number;
  slot: number;
  partId: number;
  part?: ShipPartDefinition;
}

interface Props {
  legalActions: number[];
  busy: boolean;
  onAction: (actionId: number) => void;
  player: Player;
  upgrade?: UpgradeState;
  partCatalog: ShipPartCatalog;
  mode: 'normal' | 'discovery';
  discoveredPartId?: number;
}

function signed(value: number): string {
  return value > 0 ? `+${value}` : String(value);
}

function partClasses(part?: ShipPartDefinition, isRemoval = false): string {
  if (isRemoval) return 'ship-part-tile remove';
  if (!part) return 'ship-part-tile empty';
  return `ship-part-tile ${part.is_discovery ? 'discovery' : ''} ${part.die_color !== 'none' ? `die-${part.die_color}` : ''}`;
}

function getPartByName(partCatalog: ShipPartCatalog, name: string): ShipPartDefinition | undefined {
  if (!name || name === 'None') return undefined;
  return Object.values(partCatalog).find((part) => part.name === name);
}

function PartTile({
  part,
  isRemoval = false,
  label,
  disabled = false,
  onClick,
}: {
  part?: ShipPartDefinition;
  isRemoval?: boolean;
  label?: string;
  disabled?: boolean;
  onClick?: () => void;
}) {
  const title = part
    ? [
        part.name,
        part.die_color !== 'none' ? `${part.die_amount} ${part.die_color} ${part.is_missile ? 'missile' : 'cannon'}` : null,
        part.added_computer ? `Computer ${signed(part.added_computer)}` : null,
        part.added_shield ? `Shield ${signed(part.added_shield)}` : null,
        part.added_hull ? `Hull ${signed(part.added_hull)}` : null,
        part.net_energy ? `Energy ${signed(part.net_energy)}` : null,
        part.net_initiative ? `Initiative ${signed(part.net_initiative)}` : null,
        part.added_movement ? `Move ${part.added_movement}` : null,
      ].filter(Boolean).join(' | ')
    : isRemoval ? 'Remove part' : 'Empty slot';

  return (
    <button
      type="button"
      className={partClasses(part, isRemoval)}
      title={title}
      disabled={disabled}
      onClick={onClick}
    >
      <span className="part-name">{label ?? part?.name ?? (isRemoval ? 'Remove' : 'Empty')}</span>
      <span className="part-symbols">
        {part && part.die_color !== 'none' && (
          <span className={`part-dice die-${part.die_color}`}>{part.die_amount}{part.is_missile ? 'M' : ''}</span>
        )}
        {part?.added_computer ? <span className="part-stat">+{part.added_computer}</span> : null}
        {part?.added_shield ? <span className="part-stat shield">{signed(part.added_shield)}</span> : null}
        {part?.added_hull ? <span className="part-stat hull">{part.added_hull}</span> : null}
        {part?.added_movement ? <span className="part-stat move">{part.added_movement}</span> : null}
        {part?.net_energy ? <span className={`part-energy ${part.net_energy > 0 ? 'positive' : 'negative'}`}>⚡{signed(part.net_energy)}</span> : null}
        {part?.net_initiative ? <span className="part-init">↥{signed(part.net_initiative)}</span> : null}
        {part?.external ? <span className="part-external">EXT</span> : null}
      </span>
    </button>
  );
}

function StatStrip({ stats }: { stats: ShipStats }) {
  return (
    <div className="blueprint-stats">
      <span title="Initiative">↥ {stats.initiative}</span>
      <span title="Computer">+ {stats.computer}</span>
      <span title="Shield">▣ {stats.shield}</span>
      <span title="Hull">✹ {stats.hull}</span>
      <span title="Movement">⬡ {stats.movement}</span>
      <span className={stats.energy_net < 0 ? 'bad' : ''} title="Energy net">⚡ {stats.energy_net}</span>
    </div>
  );
}

export default function UpgradePanel({
  legalActions,
  busy,
  onAction,
  player,
  upgrade,
  partCatalog,
  mode,
  discoveredPartId,
}: Props) {
  const [selectedShip, setSelectedShip] = useState(0);
  const [selectedSlot, setSelectedSlot] = useState<number | null>(null);

  const choices = useMemo(() => {
    const decoded: UpgradeChoice[] = [];
    for (const actionId of legalActions) {
      if (actionId < ACTION.UPGRADE_CHOICE_START || actionId >= ACTION.UPGRADE_END) continue;
      const encoded = actionId - ACTION.UPGRADE_CHOICE_START;
      const ship = Math.floor(encoded / (ACTION.UPGRADE_SLOTS_PER_SHIP * ACTION.UPGRADE_PART_COUNT));
      const rem = encoded % (ACTION.UPGRADE_SLOTS_PER_SHIP * ACTION.UPGRADE_PART_COUNT);
      const slot = Math.floor(rem / ACTION.UPGRADE_PART_COUNT);
      const partId = rem % ACTION.UPGRADE_PART_COUNT;
      if (ship < 0 || ship >= ACTION.UPGRADE_SHIP_COUNT) continue;
      decoded.push({
        actionId,
        ship,
        slot,
        partId,
        part: partCatalog[String(partId)],
      });
    }
    return decoded;
  }, [legalActions, partCatalog]);

  const legalSlotsByShip = useMemo(() => {
    const slots = new Map<number, Set<number>>();
    for (const choice of choices) {
      const shipSlots = slots.get(choice.ship) ?? new Set<number>();
      shipSlots.add(choice.slot);
      slots.set(choice.ship, shipSlots);
    }
    return slots;
  }, [choices]);

  useEffect(() => {
    const shipHasChoices = legalSlotsByShip.get(selectedShip)?.size;
    if (shipHasChoices) return;
    const firstChoice = choices[0];
    if (firstChoice) {
      setSelectedShip(firstChoice.ship);
      setSelectedSlot(firstChoice.slot);
    }
  }, [choices, legalSlotsByShip, selectedShip]);

  useEffect(() => {
    const legalSlots = legalSlotsByShip.get(selectedShip);
    if (!legalSlots?.size) {
      setSelectedSlot(null);
      return;
    }
    if (selectedSlot !== null && legalSlots.has(selectedSlot)) return;
    setSelectedSlot([...legalSlots].sort((a, b) => a - b)[0]);
  }, [legalSlotsByShip, selectedShip, selectedSlot]);

  const selectedChoices = choices
    .filter((choice) => choice.ship === selectedShip && choice.slot === selectedSlot)
    .sort((left, right) => left.partId - right.partId);

  const canStop = legalActions.includes(ACTION.UPGRADE_STOP);
  const discoveredPart = discoveredPartId !== undefined ? partCatalog[String(discoveredPartId)] : undefined;
  const inventoryCounts = (player.parts_inventory ?? []).reduce<Record<string, number>>((counts, name) => {
    counts[name] = (counts[name] ?? 0) + 1;
    return counts;
  }, {});

  const renderBlueprint = (blueprint: Blueprint, ship: number) => {
    const legalSlots = legalSlotsByShip.get(ship) ?? new Set<number>();
    const isSelectedShip = selectedShip === ship;
    const slots = blueprint.slots.slice(0, blueprint.capacity);

    return (
      <div
        key={SHIP_NAMES[ship]}
        className={`blueprint-card ${isSelectedShip ? 'selected' : ''} ${legalSlots.size ? 'legal' : ''}`}
        onClick={() => {
          setSelectedShip(ship);
          const firstSlot = [...legalSlots].sort((a, b) => a - b)[0];
          setSelectedSlot(firstSlot ?? null);
        }}
      >
        <div className="blueprint-header">
          <img
            className="blueprint-ship-img"
            src={shipImageUrl(SHIP_NAMES[ship])}
            alt={SHIP_NAMES[ship]}
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
          />
          <div>
            <div className="blueprint-name">{SHIP_NAMES[ship]}</div>
            <div className="blueprint-subtitle">{blueprint.capacity} spaces</div>
          </div>
        </div>
        <StatStrip stats={blueprint.total_stats} />
        <div className="blueprint-grid">
          {slots.map((slotName, slot) => {
            const part = getPartByName(partCatalog, slotName);
            const isLegal = legalSlots.has(slot);
            const isSelected = isSelectedShip && selectedSlot === slot;
            return (
              <div
                key={`${SHIP_NAMES[ship]}-${slot}`}
                className={`blueprint-slot ${isLegal ? 'legal' : ''} ${isSelected ? 'selected' : ''}`}
                onClick={(event) => {
                  event.stopPropagation();
                  if (!isLegal) return;
                  setSelectedShip(ship);
                  setSelectedSlot(slot);
                }}
              >
                <PartTile part={part} disabled label={part?.name ?? 'Open'} />
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="upgrade-panel">
      <div className="upgrade-header">
        <div>
          <div className="upgrade-kicker">{mode === 'discovery' ? 'Discovery part' : 'Ship blueprints'}</div>
          <div className="upgrade-title">
            {mode === 'discovery'
              ? 'Place now or store'
              : `Modify a blueprint${upgrade && upgrade.activations_remaining > 1 ? ` · ${upgrade.activations_remaining} activations` : ''}`}
          </div>
        </div>
        {canStop && (
          <button className="action-btn secondary upgrade-stop" disabled={busy} onClick={() => onAction(ACTION.UPGRADE_STOP)}>
            {mode === 'discovery' ? 'Store for later' : 'Stop'}
          </button>
        )}
      </div>

      {mode === 'discovery' && (
        <div className="discovery-part-focus">
          <PartTile part={discoveredPart} disabled={busy} label={discoveredPart?.name ?? 'Discovery part'} />
        </div>
      )}

      <div className="blueprint-board">
        {player.blueprints.slice(0, ACTION.UPGRADE_SHIP_COUNT).map(renderBlueprint)}
      </div>

      <div className="upgrade-tray">
        <div className="tray-title">
          {selectedSlot === null
            ? 'Select a glowing slot'
            : `${SHIP_NAMES[selectedShip]} space ${selectedSlot + 1}`}
        </div>
        {selectedChoices.length > 0 ? (
          <div className="part-tray-grid">
            {selectedChoices.map((choice) => (
              <PartTile
                key={choice.actionId}
                part={choice.part}
                isRemoval={choice.partId === 0}
                disabled={busy}
                label={choice.partId === 0 ? 'Remove' : choice.part?.name}
                onClick={() => onAction(choice.actionId)}
              />
            ))}
          </div>
        ) : (
          <div className="upgrade-empty">No legal parts for this space.</div>
        )}
      </div>

      {Object.keys(inventoryCounts).length > 0 && (
        <div className="discovery-inventory">
          <div className="tray-title">Stored discovery parts</div>
          <div className="inventory-parts">
            {Object.entries(inventoryCounts).map(([name, count]) => (
              <span key={name} className="inventory-chip">
                {name}{count > 1 ? ` ×${count}` : ''}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
