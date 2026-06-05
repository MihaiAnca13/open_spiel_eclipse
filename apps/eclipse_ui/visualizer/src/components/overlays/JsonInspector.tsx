import { useState } from 'react';
import type { SetupSnapshot } from '../../types/game';

interface JsonInspectorProps {
  snapshot: SetupSnapshot | null;
}

export default function JsonInspector({ snapshot }: JsonInspectorProps) {
  const [jsonExpanded, setJsonExpanded] = useState<boolean>(false);

  if (!snapshot) return null;

  return (
    <div className="json-inspector">
      <div className="json-title" onClick={() => setJsonExpanded(!jsonExpanded)}>
        <span>🔍 Inspect Raw Setup Snapshot JSON</span>
        <span>{jsonExpanded ? 'Collapse ▲' : 'Expand ▼'}</span>
      </div>
      {jsonExpanded && (
        <pre className="json-code">
          {JSON.stringify(snapshot, null, 2)}
        </pre>
      )}
    </div>
  );
}
