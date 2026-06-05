interface DebugPanelProps {
  isStarted: boolean;
  isHost: boolean;
  debugBusy: boolean;
  debugStateText: string;
  setDebugStateText: (text: string) => void;
  dumpDebugState: () => void;
  loadDebugState: () => void;
}

export default function DebugPanel({
  isStarted,
  isHost,
  debugBusy,
  debugStateText,
  setDebugStateText,
  dumpDebugState,
  loadDebugState,
}: DebugPanelProps) {
  if (!isStarted || !isHost) return null;

  return (
    <div className="panel debug-state-panel">
      <div className="debug-state-header">
        <div>
          <h3 className="panel-title">Debug State</h3>
          <span className="text-xs text-[#94a3b8]">Canonical backend game blob</span>
        </div>
        <div className="debug-state-actions">
          <button className="btn-secondary" onClick={dumpDebugState} disabled={debugBusy}>
            Dump State
          </button>
          <button className="btn-primary" onClick={loadDebugState} disabled={debugBusy}>
            Load State
          </button>
        </div>
      </div>
      <textarea
        className="debug-state-textarea"
        value={debugStateText}
        onChange={(event) => setDebugStateText(event.target.value)}
        placeholder="Dump or paste a canonical game blob"
        spellCheck={false}
      />
    </div>
  );
}
