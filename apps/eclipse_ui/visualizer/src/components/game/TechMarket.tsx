import { ACTION } from '../../actionTypes';
import { TECH_CATEGORIES, techImageUrl, type TechMarketEntry } from '../../types/lobby';

interface TechMarketProps {
  techRows: Record<string, [string, TechMarketEntry][]>;
  isResearchPhase: boolean;
  legalActions: number[];
  selectedRareTech: { name: string; order: number } | null;
  setSelectedRareTech: (tech: { name: string; order: number } | null) => void;
  submitAction: (actionId: number) => void;
  beginPreview: (preview: { src: string; label: string }) => void;
  clearPreview: () => void;
}

export default function TechMarket({
  techRows,
  isResearchPhase,
  legalActions,
  selectedRareTech,
  setSelectedRareTech,
  submitAction,
  beginPreview,
  clearPreview,
}: TechMarketProps) {
  return (
    <div className="panel">
      <h3 className="panel-title">Round 1 Technology Market</h3>
      <div className="tech-market">
        {TECH_CATEGORIES.map((category) => {
          const techs = techRows[category];
          if (!techs || !techs.length) return null;
          return (
            <div key={category} className="tech-row">
              {techs.map(([techName, tech]) => {
                let isResearchable = false;
                let onTechClick: (() => void) | undefined = undefined;

                if (isResearchPhase) {
                  if (tech.category !== 'Rare') {
                    const actionId = ACTION.RESEARCH_STANDARD_START + (tech.order ?? 0);
                    if (legalActions.includes(actionId)) {
                      isResearchable = true;
                      onTechClick = () => submitAction(actionId);
                    }
                  } else {
                    const rareIdx = (tech.order ?? 0) - 24;
                    const isAnyTrackLegal = [0, 1, 2].some((track) =>
                      legalActions.includes(ACTION.RESEARCH_RARE_START + rareIdx * 3 + track)
                    );
                    if (isAnyTrackLegal) {
                      isResearchable = true;
                      onTechClick = () => setSelectedRareTech({ name: techName, order: tech.order ?? 0 });
                    }
                  }
                }

                return (
                  <div
                    key={techName}
                    className={`tech-card ${tech.category} ${tech.count === 0 ? 'unavailable' : ''} ${
                      isResearchable ? 'researchable' : ''
                    } ${selectedRareTech?.order === tech.order ? 'selected-research' : ''}`}
                    onClick={onTechClick}
                    onMouseEnter={() =>
                      beginPreview({ src: techImageUrl(techName, tech.category), label: techName })
                    }
                    onMouseLeave={clearPreview}
                  >
                    <img
                      className="tech-image"
                      src={techImageUrl(techName, tech.category)}
                      alt=""
                      onError={(event) => {
                        event.currentTarget.style.display = 'none';
                      }}
                    />
                    <div className="tech-name">{techName}</div>
                    <div className="tech-meta">
                      <span className={`tech-category-badge ${tech.category}`}>{tech.category}</span>
                      <span className="tech-count">{tech.count}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
