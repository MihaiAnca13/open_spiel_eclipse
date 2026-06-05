import { useEffect } from 'react';
import { WS_BASE } from '../types/lobby';
import type { SetupSnapshot } from '../types/game';

export function useGameSocket(
  setupFinalized: boolean,
  onSnapshotUpdate: (snapshot: SetupSnapshot) => void
) {
  useEffect(() => {
    if (!setupFinalized) return;
    const playerId = sessionStorage.getItem('eclipse_player_id') ?? '';
    let ws: WebSocket | null = null;
    let closed = false;
    const connect = () => {
      ws = new WebSocket(`${WS_BASE}/ws?player_id=${encodeURIComponent(playerId)}`);
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'lobby_state' && msg.lobby?.phase === 'started' && msg.lobby.snapshot) {
          onSnapshotUpdate(msg.lobby.snapshot as SetupSnapshot);
        }
      };
      ws.onclose = () => {
        if (!closed) setTimeout(connect, 1000);
      };
    };
    connect();
    return () => {
      closed = true;
      ws?.close();
    };
  }, [setupFinalized, onSnapshotUpdate]);
}
