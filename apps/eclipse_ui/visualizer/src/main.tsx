import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Root from './Root'
import { API_BASE } from './types/lobby'
import { SPECIES_THEME } from './theme'

const DEFAULT_METADATA = {
  species: Object.keys(SPECIES_THEME),
  tech_catalog: {},
  npc_difficulties: ['Easy', 'Medium', 'Hard'],
}

async function bootstrap() {
  const metadata = await fetch(`${API_BASE}/metadata`)
    .then((r) => r.json())
    .catch(() => DEFAULT_METADATA)

  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <Root initialMetadata={metadata} />
    </StrictMode>,
  )
}

bootstrap()
