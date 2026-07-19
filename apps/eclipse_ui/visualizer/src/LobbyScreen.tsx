import { useEffect, useRef, useState } from 'react';
import { ImageHoverPreview } from './ImageHoverPreview';
import { useImageHoverPreview } from './hooks/useImageHoverPreview';
import { API_BASE, WS_BASE, buildTechMarketRows, techImageUrl, TECH_CATEGORIES } from './types/lobby';
import type { LobbyData, LobbySeat } from './types/lobby';
import { errorMessage } from './utils/errors';
import { NPC_PLAYER_ID } from './constants';
import type { TechCatalog, TechMarketEntry } from './types/lobby';
import { SPECIES_THEME } from './theme';
import type { SetupSnapshot } from './types/game';

interface Props {
  speciesList: string[];
  techCatalog: TechCatalog;
  difficulties: string[];
  onStart: (snapshot: SetupSnapshot, mySeatIdx: number, playerNames: (string | null)[], isHost: boolean) => void;
}

function getOrCreatePlayerId(): string {
  // sessionStorage is per-tab: two tabs in one browser must be two distinct
  // players. It survives refresh (so we silently reconnect / auto-reclaim) and
  // only clears on tab close — that case is covered by the server freeing the
  // seat on disconnect, after which the returning player claims it.
  let id = sessionStorage.getItem('eclipse_player_id');
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem('eclipse_player_id', id);
  }
  return id;
}

async function post(path: string, body: object) {
  return fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// ─── Root component ──────────────────────────────────────────────────────────

export default function LobbyScreen({ speciesList, techCatalog, difficulties, onStart }: Props) {
  const playerId = useRef(getOrCreatePlayerId()).current;
  const [playerName, setPlayerName] = useState(() => localStorage.getItem('eclipse_player_name') ?? '');
  const [joined, setJoined] = useState(false);
  const [lobby, setLobby] = useState<LobbyData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [debugStartText, setDebugStartText] = useState('');
  const wsRef = useRef<WebSocket | null>(null);
  const startedRef = useRef(false);
  const nameSyncedRef = useRef(false);
  const mySeatIdxRef = useRef(-1);
  const playerNamesRef = useRef<(string | null)[]>([]);

  const effectiveSpecies = speciesList.length > 0 ? speciesList : Object.keys(SPECIES_THEME);

  function openWebSocket(attempt = 0) {
    const ws = new WebSocket(`${WS_BASE}/ws?player_id=${encodeURIComponent(playerId)}`);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'lobby_state') {
        const newLobby = msg.lobby as LobbyData;
        setLobby(newLobby);
        mySeatIdxRef.current = newLobby.seats.findIndex((s) => s.player_id === playerId);
        // Player id == seat index in the game core; AI seats get a generic label.
        playerNamesRef.current = newLobby.seats.map((s) =>
          s.state === 'ai' ? 'AI Bot' : s.player_name
        );
        // Lobby was reset (cleared by host) — return to name form
        if (!newLobby.host_player_id && !newLobby.seats.some((s) => s.player_id === playerId)) {
          setJoined(false);
          wsRef.current?.close();
        }
        if (
          newLobby.phase === 'started' &&
          newLobby.snapshot &&
          mySeatIdxRef.current >= 0 &&
          !startedRef.current
        ) {
          startedRef.current = true;
          onStart(
            newLobby.snapshot as SetupSnapshot,
            mySeatIdxRef.current,
            playerNamesRef.current,
            newLobby.host_player_id === playerId
          );
        }
      }
    };
    ws.onclose = () => {
      if (attempt < 3) setTimeout(() => openWebSocket(attempt + 1), Math.pow(2, attempt) * 1000);
    };
  }

  // On mount: check if we already have a seat (page refresh / reconnect).
  // Only auto-join if the server already knows us — avoids phantom joins after
  // server restart or lobby reset.
  useEffect(() => {
    const storedName = localStorage.getItem('eclipse_player_name');
    if (!storedName) return;
    setPlayerName(storedName);

    fetch(`${API_BASE}/lobby`)
      .then((r) => r.json())
      .then((currentLobby: LobbyData) => {
        // Reconnect if we hold a seat, OR a seat we held was freed on
        // disconnect and is still open (last_player_id) — /lobby/join reclaims it.
        const hasSeat = currentLobby.seats.some(
          (s) => s.player_id === playerId || (s.state === 'empty' && s.last_player_id === playerId)
        );
        if (!hasSeat) return; // No existing seat → stay on name form
        // Seat exists → silent reconnect
        return post('/lobby/join', { player_id: playerId, player_name: storedName }).then(() => {
          setJoined(true);
          openWebSocket();
        });
      })
      .catch(() => { /* network down, stay on form */ });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync name from seat on first lobby update (handles reconnect with original name)
  useEffect(() => {
    if (!lobby || nameSyncedRef.current) return;
    const idx = lobby.seats.findIndex((s) => s.player_id === playerId);
    if (idx >= 0 && lobby.seats[idx].player_name) {
      const original = lobby.seats[idx].player_name!;
      setPlayerName(original);
      localStorage.setItem('eclipse_player_name', original);
      nameSyncedRef.current = true;
    }
  }, [lobby, playerId]);

  async function handleJoin() {
    const name = playerName.trim();
    if (!name) { setError('Enter a name first'); return; }
    localStorage.setItem('eclipse_player_name', name);
    setError(null);
    try {
      const res = await post('/lobby/join', { player_id: playerId, player_name: name });
      if (res.ok || res.status === 409) {
        setJoined(true);
        openWebSocket();
      } else {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? 'Failed to join lobby');
      }
    } catch (e: unknown) { setError(errorMessage(e, 'Failed to join lobby')); }
  }

  async function handleStartFromDebugState() {
    const name = playerName.trim();
    if (!name) { setError('Enter a name first'); return; }
    if (!debugStartText.trim()) { setError('Paste a dumped game state first'); return; }

    setBusy(true); setError(null);
    try {
      const gameBlob = JSON.parse(debugStartText);
      localStorage.setItem('eclipse_player_name', name);
      const res = await post('/debug/state/start', {
        player_id: playerId,
        player_name: name,
        game_blob: gameBlob,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? 'Failed to start from state');
      }

      const data = await res.json();
      const startedLobby = data.lobby as LobbyData;
      const playerNames = startedLobby.seats.map((s) =>
        s.state === 'ai' ? 'AI Bot' : s.player_name
      );
      setJoined(true);
      startedRef.current = true;
      onStart(data.snapshot as SetupSnapshot, data.seat ?? 0, playerNames, true);
    } catch (e: unknown) {
      setError(errorMessage(e, 'Failed to start from state'));
    } finally {
      setBusy(false);
    }
  }

  async function apiCall(path: string, extra: object = {}) {
    setBusy(true); setError(null);
    try {
      const res = await post(path, { player_id: playerId, ...extra });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Failed: ${path}`);
      }
      if (path === '/lobby/finalize') {
        const data = await res.json();
        if (data.snapshot && !startedRef.current) {
          startedRef.current = true;
          onStart(
            data.snapshot as SetupSnapshot,
            mySeatIdxRef.current,
            playerNamesRef.current,
            lobby?.host_player_id === playerId
          );
        }
      }
    } catch (e: unknown) { setError(errorMessage(e, `Failed: ${path}`)); }
    finally { setBusy(false); }
  }

  async function handleWarpedUniverseChange(enabled: boolean) {
    setBusy(true); setError(null);
    try {
      if (enabled && lobby && lobby.num_players < 3) {
        const playerRes = await post('/lobby/num_players', { player_id: playerId, num_players: 3 });
        if (!playerRes.ok) {
          const body = await playerRes.json().catch(() => ({}));
          throw new Error(body.detail ?? 'Failed to set player count');
        }
      }

      const res = await post('/lobby/warped_universe', { player_id: playerId, warped_universe: enabled });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? 'Failed to change Warped Universe');
      }
    } catch (e: unknown) { setError(errorMessage(e, 'Failed to change Warped Universe')); }
    finally { setBusy(false); }
  }

  async function handleClaimSeat(idx: number) {
    setBusy(true); setError(null);
    try {
      const name = playerName || localStorage.getItem('eclipse_player_name') || 'Player';
      const res = await post(`/lobby/claim-seat/${idx}`, { player_id: playerId, player_name: name });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? 'Failed to claim seat');
      }
    } catch (e: unknown) { setError(errorMessage(e, 'Failed to claim seat')); }
    finally { setBusy(false); }
  }

  useEffect(() => () => { wsRef.current?.close(); }, []);

  // ── Name form ────────────────────────────────────────────────────────────
  if (!joined) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0f1015]">
        <div className="w-80 flex flex-col gap-4 bg-[#1a1c23] border border-[#2d313f] rounded-xl p-6">
          <div>
            <h1 className="text-lg font-bold text-[#f1f5f9]">Eclipse</h1>
            <p className="text-sm text-[#64748b] mt-0.5">Enter your name to join the lobby</p>
          </div>
          {error && <p className="text-sm text-[#f87171]">{error}</p>}
          <div className="form-group">
            <label>Your Name</label>
            <input
              type="text" value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleJoin(); }}
              placeholder="e.g. Alice" autoFocus
            />
          </div>
          <button className="btn-primary" onClick={handleJoin}>
            Enter Lobby
          </button>
          <details className="flex flex-col gap-3 text-left">
            <summary className="cursor-pointer text-xs font-semibold text-[#94a3b8]">
              Start from dumped state
            </summary>
            <div className="flex flex-col gap-3 pt-3">
              <textarea
                className="debug-state-textarea min-h-[140px]"
                value={debugStartText}
                onChange={(event) => setDebugStartText(event.target.value)}
                placeholder="Paste canonical game blob"
                spellCheck={false}
              />
              <button className="btn-secondary" onClick={handleStartFromDebugState} disabled={busy}>
                {busy ? 'Starting…' : 'Start Game From State'}
              </button>
            </div>
          </details>
        </div>
      </div>
    );
  }

  if (!lobby) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0f1015]">
        <p className="text-[#64748b]">Connecting…</p>
      </div>
    );
  }

  const mySeatIdx = lobby.seats.findIndex((s) => s.player_id === playerId);
  const isHost = lobby.host_player_id === playerId;
  const hasNoSeat = mySeatIdx === -1;

  if (lobby.phase === 'started' && hasNoSeat) {
    return (
      <StartedClaimScreen
        lobby={lobby}
        playerId={playerId}
        busy={busy}
        error={error}
        onClaimSeat={handleClaimSeat}
      />
    );
  }

  if (lobby.phase === 'setup') {
    return (
      <SetupPhase
        lobby={lobby} playerId={playerId} mySeatIdx={mySeatIdx} isHost={isHost}
        hasNoSeat={hasNoSeat} effectiveSpecies={effectiveSpecies} techCatalog={techCatalog} busy={busy} error={error}
        onSpeciesChange={(idx, sp) => apiCall(`/lobby/seat/${idx}/species`, { species: sp })}
        onAiSpeciesChange={(idx, sp) => apiCall(`/lobby/seat/${idx}/species`, { species: sp })}
        onFinalize={() => apiCall('/lobby/finalize')}
        onReset={() => apiCall('/lobby/reset')}
        onClaimSeat={handleClaimSeat}
      />
    );
  }

  // ── Waiting phase ─────────────────────────────────────────────────────────
  const canInitialize = isHost && lobby.seats.every((s) => s.state !== 'empty');
  const warpedUniverseSupported = lobby.num_players >= 3 && lobby.num_players <= 5;
  const canToggleWarpedUniverse = lobby.num_players <= 5;

  return (
    <div className="min-h-screen bg-[#0f1015] flex justify-center py-10 px-4">
      <div className="w-full max-w-[60%] min-w-[380px] flex flex-col gap-4">

        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-[#f1f5f9]">Eclipse — Lobby</h1>
            <p className="text-xs text-[#64748b] mt-0.5">
              {lobby.seats.filter((s) => s.state !== 'empty').length} / {lobby.num_players} seats filled
            </p>
          </div>
          {isHost && (
            <button onClick={() => apiCall('/lobby/reset')}
              className="text-xs text-[#64748b] hover:text-[#f87171] border border-[#2d313f] hover:border-[#f87171]/40 px-3 py-1.5 rounded-lg transition-colors">
              Reset Lobby
            </button>
          )}
        </div>

        {error && <p className="text-sm text-[#f87171] bg-[#7f1d1d]/30 border border-[#f87171]/30 rounded-lg px-3 py-2">{error}</p>}

        {hasNoSeat && (
          <p className="text-sm text-[#93c5fd] bg-[#1e3a5f]/40 border border-[#3b82f6]/30 rounded-lg px-3 py-2">
            Lobby is full — click a seat below to reclaim your spot.
          </p>
        )}

        {isHost && (
          <div className="bg-[#1a1c23] border border-[#2d313f] rounded-xl p-4 flex flex-col gap-3">
            <h3 className="text-xs font-semibold text-[#64748b] uppercase tracking-wider">Game Settings</h3>
            <div className="form-group mb-0">
              <label>Players: {lobby.num_players}</label>
              <input type="range" min="2" max="6" value={lobby.num_players}
                onChange={(e) => apiCall('/lobby/num_players', { num_players: Number(e.target.value) })} />
            </div>
            <div className="flex gap-3">
              <div className="form-group flex-1 mb-0">
                <label>NPC Difficulty</label>
                <select value={lobby.difficulty} onChange={(e) => apiCall('/lobby/difficulty', { difficulty: e.target.value })}>
                  {difficulties.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              <div className="form-group flex-1 mb-0">
                <label>RNG Seed</label>
                <input type="number" value={lobby.rng_seed}
                  onChange={(e) => apiCall('/lobby/seed', { rng_seed: Number(e.target.value) })} />
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                id="lobby-warped-universe"
                checked={warpedUniverseSupported && lobby.warped_universe}
                disabled={!canToggleWarpedUniverse || busy}
                onChange={(e) => handleWarpedUniverseChange(e.target.checked)}
              />
              <label htmlFor="lobby-warped-universe" className="text-[#cbd5e1] cursor-pointer">
                Warped Universe
              </label>
              {lobby.num_players < 3 && (
                <span className="text-[#64748b]">enabling adds a 3rd seat</span>
              )}
              {lobby.num_players > 5 && (
                <span className="text-[#64748b]">3-5 players only</span>
              )}
            </div>
          </div>
        )}

        {!isHost && (
          <div className="bg-[#1a1c23] border border-[#2d313f] rounded-xl px-4 py-3 flex gap-5 text-xs text-[#64748b]">
            <span>Players: <span className="text-[#94a3b8] font-medium">{lobby.num_players}</span></span>
            <span>Difficulty: <span className="text-[#94a3b8] font-medium">{lobby.difficulty}</span></span>
            <span>Seed: <span className="text-[#94a3b8] font-medium">{lobby.rng_seed}</span></span>
            <span>Warped: <span className="text-[#94a3b8] font-medium">{lobby.warped_universe ? 'On' : 'Off'}</span></span>
          </div>
        )}

        <div className="bg-[#1a1c23] border border-[#2d313f] rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-[#2d313f]">
            <h3 className="text-xs font-semibold text-[#64748b] uppercase tracking-wider">Seats</h3>
          </div>
          <div className="divide-y divide-[#2d313f]">
            {lobby.seats.map((seat, idx) => (
              <WaitingSeat key={idx} seat={seat} idx={idx}
                isMe={mySeatIdx === idx} isHost={isHost} lobbyHostId={lobby.host_player_id}
                canClaim={hasNoSeat}
                onSetAI={() => apiCall(`/lobby/seat/${idx}/set_ai`)}
                onSetEmpty={() => apiCall(`/lobby/seat/${idx}/set_empty`)}
                onClaim={() => handleClaimSeat(idx)}
              />
            ))}
          </div>
        </div>

        {isHost ? (
          <button className="btn-primary" onClick={() => apiCall('/lobby/initialize')} disabled={!canInitialize || busy}>
            {busy ? 'Initializing…' : canInitialize ? 'Initialize Game' : 'Fill all seats to continue'}
          </button>
        ) : (
          <p className="text-center text-xs text-[#64748b] py-1">Waiting for host to initialize the game…</p>
        )}
      </div>
    </div>
  );
}

// ─── Started game seat claim ────────────────────────────────────────────────

interface StartedClaimScreenProps {
  lobby: LobbyData;
  playerId: string;
  busy: boolean;
  error: string | null;
  onClaimSeat: (idx: number) => void;
}

function StartedClaimScreen({ lobby, playerId, busy, error, onClaimSeat }: StartedClaimScreenProps) {
  const claimedSeatIdx = lobby.seats.findIndex((seat) => seat.player_id === playerId);

  return (
    <div className="min-h-screen bg-[#0f1015] flex justify-center py-10 px-4">
      <div className="w-full max-w-[560px] min-w-[340px] flex flex-col gap-4">
        <div>
          <h1 className="text-lg font-bold text-[#f1f5f9]">Eclipse — Claim Seat</h1>
          <p className="text-xs text-[#64748b] mt-0.5">
            This game was started from a saved state. Pick your player seat to join.
          </p>
        </div>

        {error && <p className="text-sm text-[#f87171] bg-[#7f1d1d]/30 border border-[#f87171]/30 rounded-lg px-3 py-2">{error}</p>}

        <div className="bg-[#1a1c23] border border-[#2d313f] rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-[#2d313f] flex items-center justify-between">
            <h3 className="text-xs font-semibold text-[#64748b] uppercase tracking-wider">Seats</h3>
            {claimedSeatIdx >= 0 && <span className="text-xs text-[#60a5fa]">Seat {claimedSeatIdx + 1} claimed</span>}
          </div>
          <div className="divide-y divide-[#2d313f]">
            {lobby.seats.map((seat, idx) => {
              const isUnclaimedHuman = seat.state === 'human' && !seat.player_id;
              const isClaimed = seat.player_id === playerId;
              const name = seat.state === 'ai'
                ? 'AI Bot'
                : seat.player_name || (isUnclaimedHuman ? 'Unclaimed player' : 'Human player');
              const species = seat.species || 'Unknown species';
              return (
                <button
                  key={idx}
                  type="button"
                  disabled={busy || !isUnclaimedHuman}
                  onClick={() => onClaimSeat(idx)}
                  className={[
                    'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors',
                    isUnclaimedHuman ? 'hover:bg-[#1e3a5f]/20 cursor-pointer' : 'cursor-default opacity-70',
                    isClaimed ? 'bg-[#1e3a5f]/30' : '',
                  ].filter(Boolean).join(' ')}
                >
                  <span className="text-xs text-[#475569] font-mono w-4 flex-shrink-0">{idx + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-[#f1f5f9]">{name}</span>
                      {isUnclaimedHuman && <span className="text-[10px] text-[#3b82f6] font-semibold">claim</span>}
                      {seat.state === 'ai' && <span className="text-[10px] text-[#64748b] font-medium">AI</span>}
                      {isClaimed && <span className="text-[10px] text-[#60a5fa] font-semibold">YOU</span>}
                    </div>
                    <p className="text-xs text-[#475569] mt-0.5">{species}</p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Waiting phase seat row ──────────────────────────────────────────────────

interface WaitingSeatProps {
  seat: LobbySeat; idx: number; isMe: boolean; isHost: boolean;
  lobbyHostId: string | null; canClaim: boolean;
  onSetAI: () => void; onSetEmpty: () => void; onClaim: () => void;
}

function WaitingSeat({ seat, idx, isMe, isHost, lobbyHostId, canClaim, onSetAI, onSetEmpty, onClaim }: WaitingSeatProps) {
  const isHostSeat = seat.player_id === lobbyHostId;
  const name = seat.state === 'empty' ? 'Open' : seat.state === 'ai' ? 'AI Bot' : (seat.player_name ?? 'Player');
  const sub = seat.state === 'empty' ? 'Waiting for player…' : seat.state === 'ai' ? 'AI controlled' : isHostSeat ? 'Host' : 'Human player';

  return (
    <div
      onClick={canClaim && seat.state !== 'empty' ? onClaim : undefined}
      className={[
        'flex items-center gap-3 px-4 py-3',
        canClaim && seat.state !== 'empty' ? 'cursor-pointer hover:bg-[#1e3a5f]/20' : '',
      ].filter(Boolean).join(' ')}
    >
      <span className="text-xs text-[#475569] font-mono w-4 flex-shrink-0">{idx + 1}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-medium ${seat.state === 'empty' ? 'text-[#475569] italic' : 'text-[#f1f5f9]'}`}>
            {name}
          </span>
          {isHostSeat && <span className="text-[10px] text-[#a855f7] font-semibold">HOST</span>}
          {isMe && <span className="text-[10px] text-[#60a5fa] font-semibold">YOU</span>}
          {canClaim && seat.state !== 'empty' && <span className="text-[10px] text-[#3b82f6] font-semibold">click to reclaim</span>}
        </div>
        <p className="text-xs text-[#475569] mt-0.5">{sub}</p>
      </div>
      {isHost && seat.state === 'empty' && (
        <button onClick={(e) => { e.stopPropagation(); onSetAI(); }}
          className="text-xs text-[#64748b] hover:text-[#cbd5e1] border border-[#2d313f] hover:border-[#475569] px-2.5 py-1 rounded-lg transition-colors">
          Mark AI
        </button>
      )}
      {isHost && seat.state === 'ai' && (
        <button onClick={(e) => { e.stopPropagation(); onSetEmpty(); }}
          className="text-xs text-[#64748b] hover:text-[#cbd5e1] border border-[#2d313f] hover:border-[#475569] px-2.5 py-1 rounded-lg transition-colors">
          Open Seat
        </button>
      )}
    </div>
  );
}

// ─── Setup phase ─────────────────────────────────────────────────────────────

interface SetupPhaseProps {
  lobby: LobbyData; playerId: string; mySeatIdx: number; isHost: boolean;
  hasNoSeat: boolean; effectiveSpecies: string[]; techCatalog: TechCatalog; busy: boolean; error: string | null;
  onSpeciesChange: (idx: number, species: string) => void;
  onAiSpeciesChange: (idx: number, species: string) => void;
  onFinalize: () => void; onReset: () => void; onClaimSeat: (idx: number) => void;
}

function SetupPhase({
  lobby, mySeatIdx, isHost, hasNoSeat, effectiveSpecies, techCatalog, busy, error,
  onSpeciesChange, onAiSpeciesChange, onFinalize, onReset, onClaimSeat,
}: SetupPhaseProps) {
  const stage1 = lobby.stage1_snapshot as SetupSnapshot | undefined;
  const rawTurnOrder: number[] = stage1?.state?.turn_order ?? [];
  const turnOrder = rawTurnOrder.filter((id: number) => id !== NPC_PLAYER_ID);
  const techTray: Record<string, TechMarketEntry> = stage1?.state?.tech_tray ?? {};
  const techRows = buildTechMarketRows(techCatalog, techTray);
  const { preview, beginPreview, clearPreview } = useImageHoverPreview();

  const pickerOrder = lobby.picker_order;
  const currentPickerSeat = pickerOrder[lobby.current_picker_idx] ?? -1;
  const allPicked = lobby.current_picker_idx >= pickerOrder.length;
  const isMyTurn = !allPicked && currentPickerSeat === mySeatIdx;

  // Local pending species for the current picker — lets you browse before confirming
  const myCurrentSpecies = mySeatIdx >= 0 ? lobby.seats[mySeatIdx].species : 'Terran Factions';
  const [pendingSpecies, setPendingSpecies] = useState(myCurrentSpecies);

  // Re-sync pending species when our seat changes (after claiming)
  useEffect(() => {
    if (mySeatIdx >= 0) setPendingSpecies(lobby.seats[mySeatIdx].species);
  // Only re-init when mySeatIdx changes, not on every lobby update
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mySeatIdx]);

  function getSpeciesOptions(seatIdx: number) {
    const takenAliens = lobby.seats
      .filter((_s, i) => i !== seatIdx)
      .map((s) => s.species)
      .filter((sp) => !SPECIES_THEME[sp]?.isTerran);
    return effectiveSpecies.map((name) => ({
      value: name,
      label: SPECIES_THEME[name]?.displayLabel ?? name,
      disabled: !SPECIES_THEME[name]?.isTerran && takenAliens.includes(name),
    }));
  }

  function handleConfirmPick() {
    if (mySeatIdx < 0) return;
    onSpeciesChange(mySeatIdx, pendingSpecies);
  }

  return (
    <div className="w-[95%] mx-auto app-container">
      <header className="header">
        <h1>Eclipse — Species Selection</h1>
        <p>Turn order resolved. Species are chosen in reverse turn order — last in turn picks first.</p>
      </header>

      {error && (
        <p className="text-sm text-[#f87171] bg-[#7f1d1d]/30 border border-[#f87171]/30 rounded-lg px-3 py-2 mb-4">
          {error}
        </p>
      )}

      {hasNoSeat && (
        <div className="bg-[#1e3a5f]/40 border border-[#3b82f6]/30 rounded-lg px-4 py-3 mb-4 text-sm text-[#93c5fd]">
          Your session expired. Click a seat below to reclaim your spot.
        </div>
      )}

      <div className="main-layout">
        {/* Left: picker order + species selection */}
        <div className="panel flex flex-col gap-4">

          {/* Picker order */}
          {pickerOrder.length > 0 && (
            <div>
              <h3 className="panel-title">Picker Order</h3>
              <div className="flex flex-col gap-1 mt-2">
                {pickerOrder.map((seatIdx, rank) => {
                  const seat = lobby.seats[seatIdx];
                  const turnPos = turnOrder.indexOf(seatIdx) + 1;
                  const isPicked = rank < lobby.current_picker_idx;
                  const isCurrent = rank === lobby.current_picker_idx;
                  return (
                    <div key={seatIdx}
                      className={`flex items-center gap-2 text-sm rounded px-2 py-1 ${isCurrent ? 'bg-[#a855f7]/10 border border-[#a855f7]/20' : ''}`}>
                      <span className={`font-bold w-4 text-right text-xs ${isPicked ? 'text-[#334155]' : isCurrent ? 'text-[#a855f7]' : 'text-[#475569]'}`}>
                        {rank + 1}.
                      </span>
                      <span className={`flex-1 text-sm ${isPicked ? 'text-[#334155] line-through' : 'text-[#cbd5e1]'}`}>
                        {seat?.player_name ?? `Seat ${seatIdx + 1}`}
                      </span>
                      <span className="text-[10px] text-[#475569]">turn #{turnPos}</span>
                      {isCurrent && <span className="text-[10px] text-[#a855f7] font-semibold">NOW</span>}
                      {isPicked && <span className="text-[#334155] text-xs">✓</span>}
                    </div>
                  );
                })}
                {allPicked && (
                  <p className="text-xs text-[#64748b] mt-1 px-2">All picks done. Host can finalize.</p>
                )}
              </div>
            </div>
          )}

          {/* Species cards */}
          <div>
            <h3 className="panel-title">Species Selection</h3>
            <div className="flex flex-col gap-2 mt-2">
              {lobby.seats.map((seat, idx) => {
                const pickerRank = pickerOrder.indexOf(idx);
                const isPicked = pickerRank >= 0 && pickerRank < lobby.current_picker_idx;
                const isCurrent = idx === currentPickerSeat && !allPicked;
                const isMyCard = mySeatIdx === idx;
                const isAiCard = seat.state === 'ai';
                const options = getSpeciesOptions(idx);

                // What value does the dropdown show?
                const displaySpecies = isMyCard && isMyTurn ? pendingSpecies : seat.species;
                // Dropdown editable? Only on your own pick turn, or host editing AI.
                // Once all picks are done, nobody changes species.
                const dropdownEditable = (isMyCard && isMyTurn) || (isAiCard && isHost);

                return (
                  <div
                    key={idx}
                    onClick={hasNoSeat ? () => onClaimSeat(idx) : undefined}
                    className={[
                      'bg-[#1a1c23] border rounded-xl px-4 py-3 flex flex-col gap-2',
                      isCurrent ? 'border-[#a855f7]/50' : 'border-[#2d313f]',
                      isMyCard ? 'border-[#60a5fa]/30' : '',
                      hasNoSeat ? 'cursor-pointer hover:border-[#3b82f6]/40 hover:bg-[#1e3a5f]/10' : '',
                    ].filter(Boolean).join(' ')}
                  >
                    {/* Seat header */}
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-[#475569] font-mono w-4">{idx + 1}</span>
                      <span className="text-sm font-medium text-[#f1f5f9] flex-1">
                        {isAiCard ? 'AI Bot' : (seat.player_name ?? `Seat ${idx + 1}`)}
                      </span>
                      {seat.player_id === lobby.host_player_id && <span className="text-[10px] text-[#a855f7] font-semibold">HOST</span>}
                      {isMyCard && !hasNoSeat && <span className="text-[10px] text-[#60a5fa] font-semibold">YOU</span>}
                      {isAiCard && <span className="text-[10px] text-[#64748b] font-medium">AI</span>}
                      {isPicked && <span className="text-[10px] text-[#475569]">✓ picked</span>}
                      {isCurrent && <span className="text-[10px] text-[#a855f7] font-semibold">← picking now</span>}
                      {hasNoSeat && <span className="text-[10px] text-[#3b82f6] font-semibold">click to claim</span>}
                    </div>

                    {/* Species selector */}
                    {seat.state !== 'empty' && (
                      <select
                        value={displaySpecies}
                        disabled={!dropdownEditable}
                        onClick={(e) => hasNoSeat && e.stopPropagation()}
                        onChange={(e) => {
                          if (isMyCard && isMyTurn) {
                            // Local state only — confirmed via Pick button
                            setPendingSpecies(e.target.value);
                          } else if (isAiCard && isHost) {
                            // AI seat: immediate API call, no confirm step
                            onAiSpeciesChange(idx, e.target.value);
                          }
                        }}
                      >
                        {options.map((opt) => (
                          <option key={opt.value} value={opt.value} disabled={opt.disabled}>{opt.label}</option>
                        ))}
                      </select>
                    )}
                    {seat.state === 'empty' && (
                      <p className="text-xs text-[#475569] italic">Empty seat</p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Action area */}
          <div className="flex flex-col gap-2 border-t border-[#2d313f] pt-4">
            {isMyTurn && !hasNoSeat ? (
              /* Pick button — shown when it's your turn, replaces Finalize */
              <button className="btn-primary" onClick={handleConfirmPick} disabled={busy}>
                {busy ? 'Picking…' : `Pick ${SPECIES_THEME[pendingSpecies]?.displayLabel ?? pendingSpecies}`}
              </button>
            ) : allPicked && isHost ? (
              /* Finalize — only shown after all picks done, host only */
              <button className="btn-primary" onClick={onFinalize} disabled={busy}>
                {busy ? 'Spawning Map…' : 'Finalize Species & Spawn Map'}
              </button>
            ) : allPicked && !isHost ? (
              <p className="text-center text-xs text-[#64748b]">Waiting for host to finalize…</p>
            ) : !hasNoSeat ? (
              /* Waiting for someone else's turn */
              <p className="text-center text-xs text-[#64748b]">
                Waiting for {lobby.seats[currentPickerSeat]?.player_name ?? `Seat ${currentPickerSeat + 1}`} to pick…
              </p>
            ) : null}

            {isHost && (
              <button onClick={onReset}
                className="text-xs text-center text-[#64748b] hover:text-[#f87171] py-1 transition-colors">
                Reset Lobby
              </button>
            )}
          </div>
        </div>

        {/* Right: tech market */}
        <div className="board-container">
          <div className="panel">
            <h3 className="panel-title">Round 1 Technology Market</h3>
            <span className="text-xs text-[#64748b]">Resolved from seed — available at game start</span>
            <div className="tech-market mt-3">
              {TECH_CATEGORIES.map((category) => {
                const techs = techRows[category];
                if (!techs.length) return null;
                return (
                  <div key={category} className="tech-row">
                    {techs.map(([name, tech]) => (
                      <div
                        key={name}
                        className={`tech-card ${tech.category} ${tech.count === 0 ? 'unavailable' : ''}`}
                        onMouseEnter={() =>
                          beginPreview({ src: techImageUrl(name, tech.category), label: name })
                        }
                        onMouseLeave={clearPreview}
                      >
                        <img
                          className="tech-image"
                          src={techImageUrl(name, tech.category)}
                          alt=""
                          onError={(event) => {
                            event.currentTarget.style.display = 'none';
                          }}
                        />
                        <div className="tech-name">{name}</div>
                        <div className="tech-meta">
                          <span className={`tech-category-badge ${tech.category}`}>{tech.category}</span>
                          <span className="tech-count">{tech.count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
      <ImageHoverPreview preview={preview} />
    </div>
  );
}
