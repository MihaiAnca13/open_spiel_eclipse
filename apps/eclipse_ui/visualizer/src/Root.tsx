import { useState } from 'react';
import App from './App';
import type { GameMetadata, SetupSnapshot } from './types/game';
import LobbyScreen from './LobbyScreen';

type View =
  | { screen: 'lobby' }
  | { screen: 'game'; snapshot: SetupSnapshot; mySeatIdx: number; playerNames: (string | null)[]; isHost: boolean };

export default function Root({ initialMetadata }: { initialMetadata: GameMetadata }) {
  const [view, setView] = useState<View>({ screen: 'lobby' });

  if (view.screen === 'game') {
    return (
      <App
        initialMetadata={initialMetadata}
        initialSnapshot={view.snapshot}
        mySeatIdx={view.mySeatIdx}
        playerNames={view.playerNames}
        isHost={view.isHost}
      />
    );
  }

  return (
    <LobbyScreen
      speciesList={initialMetadata.species ?? []}
      techCatalog={initialMetadata.tech_catalog ?? {}}
      difficulties={initialMetadata.npc_difficulties ?? ['Easy', 'Medium', 'Hard']}
      onStart={(snapshot, mySeatIdx, playerNames, isHost) =>
        setView({ screen: 'game', snapshot, mySeatIdx, playerNames, isHost })}
    />
  );
}
