import { useState } from 'react';
import App from './App';
import type { SetupSnapshot } from './App';
import LobbyScreen from './LobbyScreen';

type View =
  | { screen: 'lobby' }
  | { screen: 'game'; snapshot: SetupSnapshot; mySeatIdx: number; playerNames: (string | null)[] };

export default function Root({ initialMetadata }: { initialMetadata: any }) {
  const [view, setView] = useState<View>({ screen: 'lobby' });

  if (view.screen === 'game') {
    return (
      <App
        initialMetadata={initialMetadata}
        initialSnapshot={view.snapshot}
        mySeatIdx={view.mySeatIdx}
        playerNames={view.playerNames}
      />
    );
  }

  return (
    <LobbyScreen
      speciesList={initialMetadata.species ?? []}
      difficulties={initialMetadata.npc_difficulties ?? ['Easy', 'Medium', 'Hard']}
      onStart={(snapshot, mySeatIdx, playerNames) =>
        setView({ screen: 'game', snapshot, mySeatIdx, playerNames })}
    />
  );
}
