import { techImageUrl } from './types/lobby';

interface ResearchTracksProps {
  militaryMask: number;
  gridMask: number;
  nanoMask: number;
  techCatalog: Record<string, { category: string; order: number }>;
}

export default function ResearchTracks({
  militaryMask,
  gridMask,
  nanoMask,
  techCatalog,
}: ResearchTracksProps) {
  const getResearchedTechs = (mask: number, category: string) => {
    const techs: { name: string; order: number }[] = [];
    for (const [name, tech] of Object.entries(techCatalog)) {
      if (tech.category === category) {
        const bit = 1n << BigInt(tech.order);
        if ((BigInt(mask) & bit) !== 0n) {
          techs.push({ name, order: tech.order });
        }
      }
    }
    return techs.sort((a, b) => a.order - b.order);
  };

  const militaryTechs = getResearchedTechs(militaryMask, 'Military');
  const gridTechs = getResearchedTechs(gridMask, 'Grid');
  const nanoTechs = getResearchedTechs(nanoMask, 'Nano');

  const renderTechRow = (techs: { name: string; order: number }[], category: string, color: string) => {
    if (techs.length === 0) return null;
    return (
      <div key={category} className="research-track-row">
        <span className={`research-track-label ${color}`}>{category}</span>
        <div className="research-track-techs">
          {techs.map((tech) => (
            <div
              key={tech.name}
              className="research-tech-item"
              title={tech.name}
            >
              <img
                src={techImageUrl(tech.name, category)}
                alt={tech.name}
                className="research-tech-image"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (militaryTechs.length === 0 && gridTechs.length === 0 && nanoTechs.length === 0) {
    return null;
  }

  return (
    <div className="panel research-tracks-panel">
      <h3 className="panel-title">Research Tracks</h3>
      <div className="research-tracks">
        {renderTechRow(militaryTechs, 'Military', 'military')}
        {renderTechRow(gridTechs, 'Grid', 'grid')}
        {renderTechRow(nanoTechs, 'Nano', 'nano')}
      </div>
    </div>
  );
}
