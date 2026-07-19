import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Root from './Root'
import { API_BASE } from './types/lobby'
import { SPECIES_THEME } from './theme'
import type { GameMetadata } from './types/game'

const DEFAULT_METADATA: GameMetadata = {
  species: Object.keys(SPECIES_THEME),
  tech_catalog: {},
  ship_part_catalog: {},
  discovery_catalog: {},
  npc_difficulties: ['Easy', 'Medium', 'Hard'],
}

async function bootstrap() {
  const metadata: GameMetadata = await fetch(`${API_BASE}/metadata`)
    .then((r) => r.json() as Promise<GameMetadata>)
    .catch(() => DEFAULT_METADATA)

  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <Root initialMetadata={metadata} />
    </StrictMode>,
  )
}

bootstrap()
