import { SPECIES_THEME } from '../../theme';

interface SetupPanelProps {
  rngSeed: number;
  setRngSeed: (seed: number) => void;
  numPlayers: number;
  setNumPlayers: (num: number) => void;
  difficulty: string;
  setDifficulty: (diff: string) => void;
  gameMetadata: any;
  speciesChoices: Record<number, string>;
  handleSpeciesChange: (playerId: number, species: string) => void;
  getAiDefault: (playerId: number) => boolean;
  handleAiChange: (playerId: number, isAi: boolean) => void;
  playerIds: number[];
  playerLabel: (pid: number) => string;
  gameState: any;
  loading: boolean;
  handleInitializeStage1: () => void;
  handleFinalizeStage2: () => void;
  snapshot: any;
  getSpeciesOptions: (playerId: number) => { value: string; label: string; disabled?: boolean }[];
  playerStartingSectors: Record<number, number>;
  speciesList: string[];
}

export default function SetupPanel({
  rngSeed,
  setRngSeed,
  numPlayers,
  setNumPlayers,
  difficulty,
  setDifficulty,
  gameMetadata,
  speciesChoices,
  handleSpeciesChange,
  getAiDefault,
  handleAiChange,
  playerIds,
  playerLabel,
  gameState,
  loading,
  handleInitializeStage1,
  handleFinalizeStage2,
  snapshot,
  getSpeciesOptions,
  playerStartingSectors,
  speciesList,
}: SetupPanelProps) {
  return (
    <div className="panel">
      <div>
        <h3 className="panel-title">Setup Config</h3>
        <span className="text-xs text-[#94a3b8]">
          Stage 1 resolves initial randomness from this UI-driven configuration
        </span>
      </div>

      <div className="form-group">
        <label>RNG Seed</label>
        <input type="number" value={rngSeed} onChange={(e) => setRngSeed(Number(e.target.value))} />
      </div>

      <div className="form-group">
        <label>Number of Players ({numPlayers})</label>
        <input type="range" min="2" max="6" value={numPlayers} onChange={(e) => setNumPlayers(Number(e.target.value))} />
      </div>

      <div className="form-group">
        <label>NPC Difficulty (GCDS & Guardians)</label>
        <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
          {(gameMetadata?.npc_difficulties ?? ['Easy', 'Medium', 'Hard']).map((d: string) => (
            <option key={d} value={d}>{d}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-4 border-t border-[#2d313f] pt-4 mt-2">
        <div>
          <h3 className="panel-title">Player Choices</h3>
          <span className="text-xs text-[#94a3b8]">
            Species and control flags are UI-owned and sent into the setup pipeline
          </span>
        </div>

        <div className="flex flex-col gap-2.5">
          {playerIds.map((playerId) => {
            const currentTurnIdx = gameState?.turn_order.indexOf(playerId) ?? -1;
            const isFirstPicker = gameState
              ? currentTurnIdx === gameState.players.length - 1
              : false;
            return (
              <div key={playerId} className={`player-choice-card ${isFirstPicker ? 'active' : ''}`}>
                <div className="player-card-header">
                  <span className="text-[13px] font-bold">{playerLabel(playerId)}</span>
                  <span className="player-badge">
                    {gameState
                      ? isFirstPicker
                        ? '🔥 Picker 1'
                        : `Turn Position: ${currentTurnIdx + 1}`
                      : 'Pending Stage 1 turn order'}
                  </span>
                </div>

                <div className="form-group gap-1">
                  <select
                    value={speciesChoices[playerId] || speciesList[playerId % speciesList.length]}
                    onChange={(e) => handleSpeciesChange(playerId, e.target.value)}
                  >
                    {getSpeciesOptions(playerId).map((opt: { value: string; label: string; disabled?: boolean }) => (
                      <option key={opt.value} value={opt.value} disabled={opt.disabled}>{opt.label}</option>
                    ))}
                  </select>
                  {SPECIES_THEME[speciesChoices[playerId]]?.isTerran && playerStartingSectors[playerId] && (
                    <span className="text-xs text-[#94a3b8] mt-1">
                      Starting Sector: {playerStartingSectors[playerId]}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    id={`ai-${playerId}`}
                    checked={getAiDefault(playerId)}
                    onChange={(e) => handleAiChange(playerId, e.target.checked)}
                  />
                  <label htmlFor={`ai-${playerId}`} className="text-[#cbd5e1] cursor-pointer">
                    Control as AI Agent
                  </label>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <button className="btn-primary" onClick={handleInitializeStage1} disabled={loading}>
        {loading ? 'Processing C++...' : 'Generate Pre-Choice Snapshot'}
      </button>

      {snapshot && (
        <div className="flex flex-col gap-4 border-t border-[#2d313f] pt-4 mt-4">
          <div>
            <h3 className="panel-title">Stage 2: Finalize Setup</h3>
            <span className="text-xs text-[#94a3b8]">
              Applies current player choices to the deterministic Stage 1 snapshot
            </span>
          </div>
          <button className="btn-secondary" onClick={handleFinalizeStage2} disabled={loading}>
            {loading ? 'Processing C++...' : 'Finalize Species & Spawn Map'}
          </button>
        </div>
      )}
    </div>
  );
}
