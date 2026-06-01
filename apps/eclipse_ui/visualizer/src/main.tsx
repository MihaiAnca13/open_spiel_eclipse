import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { SPECIES_THEME } from './theme'

const API_BASE = 'http://127.0.0.1:8000'

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
      <App initialMetadata={metadata} />
    </StrictMode>,
  )
}

bootstrap()
