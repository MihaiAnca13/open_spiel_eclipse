import { useEffect, useMemo, useState } from 'react';
import './App.css';
import { ImageHoverPreview, useImageHoverPreview } from './ImageHoverPreview';
import { SPECIES_THEME, getPlayerColor } from './theme';
import ActionPanel from './ActionPanel';
import { ACTION } from './actionTypes';
import { API_BASE, buildTechMarketRows } from './types/lobby';
import type { SetupSnapshot } from './types/game';

// Custom Hooks
import { useGameSocket } from './hooks/useGameSocket';
import { useSectorAssets } from './hooks/useSectorAssets';
import { useSectorLayouts } from './hooks/useSectorLayouts';

// Utils
import { bagCount, EMPTY_LEGAL_ACTIONS } from './utils/game';

// Components
import SetupPanel from './components/setup/SetupPanel';
import GalaxyMap from './components/game/GalaxyMap';
import EconomyPanel from './components/game/EconomyPanel';
import TechMarket from './components/game/TechMarket';
import DebugPanel from './components/overlays/DebugPanel';
import JsonInspector from './components/overlays/JsonInspector';

export interface AppProps {
  initialMetadata: any;
  initialSnapshot?: SetupSnapshot;
  mySeatIdx?: number;
  playerNames?: (string | null)[];
  isHost?: boolean;
}

function App({
  initialMetadata,
  initialSnapshot,
  mySeatIdx = -1,
  playerNames = [],
  isHost = false,
}: AppProps) {
  const playerLabel = (pid: number) => playerNames[pid] || `Player ${pid + 1}`;
  const [rngSeed, setRngSeed] = useState<number>(initialSnapshot?.config.rng_seed ?? 42);
  const [numPlayers, setNumPlayers] = useState<number>(initialSnapshot?.config.players ?? 4);
  const [gameMetadata] = useState<any>(initialMetadata);
  const [difficulty, setDifficulty] = useState<string>(initialMetadata.npc_difficulties?.[0] ?? 'Easy');
  const [snapshot, setSnapshot] = useState<SetupSnapshot | null>(initialSnapshot ?? null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [actionInProgress, setActionInProgress] = useState<boolean>(false);
  const [previewRotation, setPreviewRotation] = useState<number | null>(null);
  const [previewingDrawnTile, setPreviewingDrawnTile] = useState<number | null>(null);
  const [previewingDrawnTileIndex, setPreviewingDrawnTileIndex] = useState<number | null>(null);
  const [debugStateText, setDebugStateText] = useState<string>('');
  const [debugBusy, setDebugBusy] = useState<boolean>(false);
  const { preview, beginPreview, clearPreview } = useImageHoverPreview();
  const [setupFinalized, setSetupFinalized] = useState<boolean>(initialSnapshot?.finalized ?? false);
  const [playerStartingSectors, setPlayerStartingSectors] = useState<Record<number, number>>({});
  const [selectedRareTech, setSelectedRareTech] = useState<{ name: string; order: number } | null>(null);

  const speciesList: string[] = gameMetadata.species ?? Object.keys(SPECIES_THEME);
  const [speciesChoices, setSpeciesChoices] = useState<Record<number, string>>(() => {
    if (initialSnapshot) {
      return Object.fromEntries(
        initialSnapshot.config.staged_players.map((p, i) => [i, p.species ?? speciesList[i % speciesList.length]])
      );
    }
    const choices: Record<number, string> = {};
    for (let i = 0; i < 6; i++) {
      choices[i] = speciesList[i % speciesList.length];
    }
    return choices;
  });
  const [aiChoices, setAiChoices] = useState<Record<number, boolean>>(() => {
    if (initialSnapshot) {
      return Object.fromEntries(
        initialSnapshot.config.staged_players.map((p, i) => [i, p.is_ai])
      );
    }
    return Object.fromEntries(Array.from({ length: 6 }, (_, i) => [i, i !== 0]));
  });

  const getAiDefault = (playerId: number) => aiChoices[playerId] ?? (playerId !== 0);

  const gameState = snapshot?.state ?? null;
  const techRows = gameState
    ? buildTechMarketRows(gameMetadata.tech_catalog ?? {}, gameState.tech_tray)
    : {};

  const getSpeciesOptions = (playerId: number) => {
    const selectedSpecies = Array.from({ length: numPlayers }, (_, i) => i)
      .filter(id => id !== playerId)
      .map(id => speciesChoices[id]);
    
    return speciesList.map((name: string) => {
      const isTerran = SPECIES_THEME[name]?.isTerran;
      const isAlienAndTaken = !isTerran && selectedSpecies.includes(name);
      
      return {
        value: name,
        label: SPECIES_THEME[name]?.displayLabel ?? name,
        disabled: isAlienAndTaken,
      };
    });
  };

  const stagedPlayers = useMemo(
    () =>
      Array.from({ length: numPlayers }, (_, playerId) => ({
        species: speciesChoices[playerId] || speciesList[playerId % speciesList.length],
        is_ai: getAiDefault(playerId)
      })),
    [aiChoices, numPlayers, speciesChoices, speciesList]
  );

  const handleInitializeStage1 = async () => {
    setLoading(true);
    setError(null);
    try {
      const config = {
        players: numPlayers,
        rng_seed: rngSeed,
        npc_difficulty: difficulty,
        staged_players: stagedPlayers
      };
      const response = await fetch(`${API_BASE}/setup/pre-choice`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (!response.ok) {
        throw new Error('FastAPI server is not responding. Make sure apps/eclipse_ui/run.sh or the backend is running.');
      }
      const data = await response.json();
      setSnapshot(data);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleFinalizeStage2 = async () => {
    if (!snapshot) return;
    setLoading(true);
    setError(null);

    const playerChoices = Array.from({ length: numPlayers }, (_, playerId) => ({
      species: speciesChoices[playerId] || speciesList[playerId % speciesList.length],
      is_ai: getAiDefault(playerId)
    }));

    try {
      const response = await fetch(`${API_BASE}/setup/finalize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          snapshot,
          player_choices: playerChoices
        })
      });
      if (!response.ok) {
        throw new Error('Stage 2 setup failed. Check FastAPI/native bindings.');
      }
      const data = await response.json();
      setSnapshot(data);
      setSetupFinalized(true);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSpeciesChange = (playerId: number, species: string) => {
    setSpeciesChoices((prev) => ({ ...prev, [playerId]: species }));
    
    // Assign random starting sector for terran players
    if (SPECIES_THEME[species]?.isTerran) {
      const terranSectors = [221, 223, 225, 227, 229, 231];
      const takenSectors = Object.values(playerStartingSectors);
      const availableSectors = terranSectors.filter(s => !takenSectors.includes(s));
      if (availableSectors.length > 0) {
        const randomSector = availableSectors[Math.floor(Math.random() * availableSectors.length)];
        setPlayerStartingSectors(prev => ({ ...prev, [playerId]: randomSector }));
      }
    } else {
      // Remove starting sector assignment for non-terran
      setPlayerStartingSectors(prev => {
        const newSectors = { ...prev };
        delete newSectors[playerId];
        return newSectors;
      });
    }
  };

  const handleAiChange = (playerId: number, isAi: boolean) => {
    setAiChoices((prev) => ({ ...prev, [playerId]: isAi }));
  };

  // Custom hooks to manage async/websocket/layout subscriptions
  useGameSocket(setupFinalized, setSnapshot);
  const sectorImages = useSectorAssets();
  const sectorLayouts = useSectorLayouts();

  const submitAction = async (actionId: number) => {
    if (actionInProgress) return;
    setActionInProgress(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/game/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seat: mySeatIdx, action_id: actionId }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `Action failed (${response.status})`);
      }
      const updated = await response.json();
      if (updated) setSnapshot(updated as SetupSnapshot);
    } catch (err: any) {
      setError(err.message || 'Action failed');
    } finally {
      setActionInProgress(false);
    }
  };

  const getPlayerId = () => sessionStorage.getItem('eclipse_player_id') ?? '';

  const dumpDebugState = async () => {
    if (debugBusy) return;
    setDebugBusy(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/debug/state/dump`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_id: getPlayerId() }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `Dump failed (${response.status})`);
      }
      const data = await response.json();
      setDebugStateText(JSON.stringify(data.game_blob, null, 2));
    } catch (err: any) {
      setError(err.message || 'Dump failed');
    } finally {
      setDebugBusy(false);
    }
  };

  const loadDebugState = async () => {
    if (debugBusy) return;
    setDebugBusy(true);
    setError(null);
    try {
      if (!debugStateText.trim()) {
        throw new Error('Paste a dumped game state first');
      }
      const gameBlob = JSON.parse(debugStateText);
      const response = await fetch(`${API_BASE}/debug/state/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_id: getPlayerId(), game_blob: gameBlob }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `Load failed (${response.status})`);
      }
      const updated = await response.json();
      if (updated) setSnapshot(updated as SetupSnapshot);
      setDebugStateText(JSON.stringify(gameBlob, null, 2));
    } catch (err: any) {
      setError(err.message || 'Load failed');
    } finally {
      setDebugBusy(false);
    }
  };

  const playerIds = Array.from({ length: numPlayers }, (_, playerId) => playerId);

  // ── Gameplay (Action phase) derived state ──────────────────────────────────
  const legalActions = snapshot?.legal_actions ?? EMPTY_LEGAL_ACTIONS;
  const exploreState = gameState?.explore_state;
  const explorePhase = exploreState?.phase ?? 'inactive';
  const researchState = gameState?.research_state;
  const isTerminal = snapshot?.is_terminal ?? false;
  const isStarted = setupFinalized && snapshot?.current_player !== undefined;
  const isMyTurn = isStarted && snapshot?.current_player === mySeatIdx && !isTerminal;
  const isResearchPhase = isMyTurn && researchState?.phase === 'choose_tech';
  const currentPlayerLabel =
    snapshot?.current_player !== undefined ? playerLabel(snapshot.current_player) : '';
  const selectedSectorId = exploreState?.selected_sector_id ?? 0;
  const legalRotations = useMemo(
    () =>
      legalActions
        .filter((action) => action >= ACTION.EXPLORE_ROT_START && action < ACTION.EXPLORE_ROT_START + 6)
        .map((action) => action - ACTION.EXPLORE_ROT_START)
        .sort((a, b) => a - b),
    [legalActions]
  );
  const inRotationPreview =
    isMyTurn &&
    explorePhase === 'choose_rotation' &&
    selectedSectorId > 0 &&
    legalRotations.length > 0;
  const currentPreviewRotation =
    inRotationPreview && previewRotation !== null && legalRotations.includes(previewRotation)
      ? previewRotation
      : inRotationPreview
        ? legalRotations[0]
        : null;
  const ancientsOnSelected =
    !!gameState &&
    selectedSectorId > 0 &&
    gameState.unit_registry.some((u) => u.sector_id === selectedSectorId && u.player_id === 255);

  const confirmPreviewRotation = () => {
    if (currentPreviewRotation === null || !legalRotations.includes(currentPreviewRotation)) return;
    submitAction(ACTION.EXPLORE_ROT_START + currentPreviewRotation);
  };

  // Bonus actions available this turn.
  const legalTradeActions = isMyTurn
    ? legalActions.filter(a => a >= ACTION.TRADE_START && a < ACTION.COLONY_SHIP_START)
    : [];
  const legalColonyShipActions = isMyTurn
    ? legalActions.filter(a => a >= ACTION.COLONY_SHIP_START)
    : [];

  // Decode colony ship actions into (sectorId, slotIdx, track) for display.
  const colonyShipPlacements = legalColonyShipActions.map(actionId => {
    const encoded = actionId - ACTION.COLONY_SHIP_START;
    const cellIdx = Math.floor(encoded / ACTION.COLONY_SHIP_CODES_PER_CELL);
    const rem = encoded % ACTION.COLONY_SHIP_CODES_PER_CELL;
    const slotIdx = Math.floor(rem / ACTION.COLONY_SHIP_TRACKS);
    const track = rem % ACTION.COLONY_SHIP_TRACKS;
    const qIdx = Math.floor(cellIdx / 15);
    const rIdx = cellIdx % 15;
    const sector = gameState?.galaxy[qIdx]?.[rIdx];
    return { actionId, sectorId: sector?.sector_id ?? 0, slotIdx, track };
  });

  const inSelectDrawnTile = isMyTurn && explorePhase === 'select_drawn_tile';

  // Clear drawn tile preview state when it's no longer the select_drawn_tile phase.
  useEffect(() => {
    if (!inSelectDrawnTile) {
      setPreviewingDrawnTile(null);
      setPreviewingDrawnTileIndex(null);
    }
  }, [inSelectDrawnTile]);

  // Clear selected rare tech when no longer in the choose_tech research phase.
  useEffect(() => {
    if (!isResearchPhase) {
      setSelectedRareTech(null);
    }
  }, [isResearchPhase]);

  return (
    <div className="w-[95%] mx-auto app-container">
      <header className="header">
        <h1>Eclipse setup & map visualizer</h1>
        <p>Interactive staged setup backed by the shared OpenSpiel Eclipse core</p>
        {mySeatIdx >= 0 && (
          <span className="text-xs text-[#60a5fa] font-semibold">
            You are Player {mySeatIdx + 1} — {playerLabel(mySeatIdx)}
          </span>
        )}
      </header>

      {error && (
        <div className="bg-[#7f1d1d] border border-[#f87171] p-3 rounded-lg mb-4 text-sm">
          <strong>Error connecting to backend:</strong> {error}
        </div>
      )}

      <div className={`main-layout ${setupFinalized ? '!grid-cols-1' : ''}`}>
        {!setupFinalized && (
          <SetupPanel
            rngSeed={rngSeed}
            setRngSeed={setRngSeed}
            numPlayers={numPlayers}
            setNumPlayers={setNumPlayers}
            difficulty={difficulty}
            setDifficulty={setDifficulty}
            gameMetadata={gameMetadata}
            speciesChoices={speciesChoices}
            handleSpeciesChange={handleSpeciesChange}
            getAiDefault={getAiDefault}
            handleAiChange={handleAiChange}
            playerIds={playerIds}
            playerLabel={playerLabel}
            gameState={gameState}
            loading={loading}
            handleInitializeStage1={handleInitializeStage1}
            handleFinalizeStage2={handleFinalizeStage2}
            snapshot={snapshot}
            getSpeciesOptions={getSpeciesOptions}
            playerStartingSectors={playerStartingSectors}
            speciesList={speciesList}
          />
        )}

        <div className="board-container">
          {gameState && (
            <div className="turn-order-bar">
              <span className="text-xs uppercase text-[#94a3b8] font-bold mr-2">
                {isStarted ? `Round ${gameState.current_round} · Turn Order:` : 'Active Turn Order:'}
              </span>
              {gameState.turn_order.map((playerId, idx) => {
                if (playerId === 255) return null;
                const player = gameState.players[playerId];
                const isActive = gameState.current_player === playerId;
                const passed = player?.has_passed;
                return (
                  <div key={playerId} className={`turn-badge ${isActive ? 'active' : ''} ${passed ? 'passed' : ''}`}>
                    <span className="turn-num">{idx + 1}</span>
                    <span style={{ color: getPlayerColor(playerId) }}>{playerLabel(playerId)} ({player?.species_id || 'Choosing...'})</span>
                    {player?.is_ai && <span className="text-[9px] bg-[#334155] text-[#93c5fd] px-1 py-0.5 rounded">AI</span>}
                    {passed && <span className="text-[9px] text-[#94a3b8]">✓ passed</span>}
                  </div>
                );
              })}
            </div>
          )}

          {isStarted && gameState && (
            <div className="stacks-bar">
              <span className="text-xs uppercase text-[#94a3b8] font-bold mr-1">Sector stacks left:</span>
              <span className="stack-pill inner" title="Ring I (Inner) tiles remaining">
                <span className="stack-ring">I</span> {bagCount(gameState.sector_bag_inner)}
              </span>
              <span className="stack-pill middle" title="Ring II (Middle) tiles remaining">
                <span className="stack-ring">II</span> {bagCount(gameState.sector_bag_middle)}
              </span>
              <span className="stack-pill outer" title="Ring III (Outer) tiles remaining">
                <span className="stack-ring">III</span> {bagCount(gameState.sector_bag_outer)}
              </span>
            </div>
          )}

          {isStarted && gameState && (
            <div className="game-hud">
              <ActionPanel
                explore={exploreState}
                research={researchState}
                selectedRareTech={selectedRareTech}
                onClearSelectedRareTech={() => setSelectedRareTech(null)}
                legalActions={legalActions}
                isMyTurn={isMyTurn}
                isTerminal={isTerminal}
                busy={actionInProgress}
                currentPlayerLabel={currentPlayerLabel}
                ancientsOnSelected={ancientsOnSelected}
                previewRotation={currentPreviewRotation}
                sectorImages={sectorImages}
                onPreviewDrawnTile={(id, idx) => {
                  setPreviewingDrawnTile(id);
                  setPreviewingDrawnTileIndex(idx);
                }}
                onClearPreviewDrawnTile={() => {
                  setPreviewingDrawnTile(null);
                  setPreviewingDrawnTileIndex(null);
                }}
                onConfirmPreviewRotation={confirmPreviewRotation}
                onAction={submitAction}
              />
              <EconomyPanel
                gameState={gameState}
                mySeatIdx={mySeatIdx}
                playerLabel={playerLabel}
                colonyShipPlacements={colonyShipPlacements}
                legalTradeActions={legalTradeActions}
                submitAction={submitAction}
                gameMetadata={gameMetadata}
              />
            </div>
          )}

          <GalaxyMap
            gameState={gameState}
            sectorImages={sectorImages}
            sectorLayouts={sectorLayouts}
            legalActions={legalActions}
            isMyTurn={isMyTurn}
            explorePhase={explorePhase}
            selectedSectorId={selectedSectorId}
            previewRotation={previewRotation}
            setPreviewRotation={setPreviewRotation}
            previewingDrawnTile={previewingDrawnTile}
            previewingDrawnTileIndex={previewingDrawnTileIndex}
            inSelectDrawnTile={inSelectDrawnTile}
            submitAction={submitAction}
            playerLabel={playerLabel}
            beginPreview={beginPreview}
            clearPreview={clearPreview}
          />

          {gameState && (
            <TechMarket
              techRows={techRows}
              isResearchPhase={isResearchPhase}
              legalActions={legalActions}
              selectedRareTech={selectedRareTech}
              setSelectedRareTech={setSelectedRareTech}
              submitAction={submitAction}
              beginPreview={beginPreview}
              clearPreview={clearPreview}
            />
          )}

          <DebugPanel
            isStarted={isStarted}
            isHost={isHost}
            debugBusy={debugBusy}
            debugStateText={debugStateText}
            setDebugStateText={setDebugStateText}
            dumpDebugState={dumpDebugState}
            loadDebugState={loadDebugState}
          />

          <JsonInspector snapshot={snapshot} />
        </div>
      </div>
      <ImageHoverPreview preview={preview} />
    </div>
  );
}

export default App;
