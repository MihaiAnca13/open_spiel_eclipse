import React, { useState } from 'react';
import type { Resources } from '../../types/game';
import { ACTION, TRADE_LABELS } from '../../actionTypes';

interface TradePanelProps {
  tradeRate: number;
  legalTradeActions: number[];
  onTrade: (actionId: number) => void;
  resources: Resources;
}

export default function TradePanel({
  tradeRate,
  legalTradeActions,
  onTrade,
  resources,
}: TradePanelProps) {
  const [open, setOpen] = useState(false);
  const allConversions = Array.from({ length: 6 }, (_, i) => i);

  const canAfford = (conv: number): boolean => {
    switch (conv) {
      case 0: case 1: return resources.gold >= tradeRate;
      case 2: case 3: return resources.science >= tradeRate;
      case 4: case 5: return resources.materials >= tradeRate;
      default: return false;
    }
  };

  const handleTrade = (conv: number) => {
    if (!canAfford(conv)) return;
    onTrade(ACTION.TRADE_START + conv);
  };

  const handleOverlayKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setOpen(false);
    }
  };

  return (
    <>
      <div className="trade-panel">
        <span className="trade-label">Trade (×{tradeRate})</span>
        <button
          className="trade-btn"
          onClick={() => setOpen(!open)}
          title={open ? 'Close trade modal' : 'Open trade modal'}
          aria-label={open ? 'Close trade modal' : 'Open trade modal'}
        >
          ⇄
        </button>
      </div>

      {open && (
        <div
          className="trade-modal-overlay"
          onClick={() => setOpen(false)}
          onKeyDown={handleOverlayKeyDown}
          role="dialog"
          aria-label="Trade resources"
        >
          <div className="trade-modal" onClick={e => e.stopPropagation()}>
            <div className="trade-modal-header">
              <span>Trade (×{tradeRate})</span>
              <button
                className="trade-modal-close"
                onClick={() => setOpen(false)}
                aria-label="Close trade modal"
              >✕</button>
            </div>
            <div className="trade-modal-body">
              {allConversions.map(conv => {
                const info = TRADE_LABELS[conv];
                if (!info) return null;
                const affordable = canAfford(conv);
                const isLegal = legalTradeActions.includes(ACTION.TRADE_START + conv);
                const unavailable = !isLegal || !affordable;
                return (
                  <button
                    key={conv}
                    className={`trade-modal-option ${unavailable ? 'trade-modal-disabled' : ''}`}
                    onClick={() => handleTrade(conv)}
                    disabled={unavailable}
                    title={isLegal
                      ? affordable
                        ? `Pay ${tradeRate} ${info.from} → 1 ${info.to}`
                        : `Need ${tradeRate} ${info.from} (have ${
                            conv === 0 || conv === 1 ? resources.gold
                            : conv === 2 || conv === 3 ? resources.science
                            : resources.materials
                          })`
                      : 'Not available this turn'
                    }
                  >
                    <span className="trade-modal-emoji">{info.emoji}</span>
                    <div className="trade-modal-info">
                      <span className="trade-modal-from">{info.from}</span>
                      <span className="trade-modal-arrow">→</span>
                      <span className="trade-modal-to">{info.to}</span>
                    </div>
                    <span className="trade-modal-cost">
                      {isLegal
                        ? affordable ? `−${tradeRate}` : `Need ${tradeRate}`
                        : '—'}
                      {info.from}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
