- [x] integrate scoring
- [x] initiative? (same player ship order selection?)
- [x] traitor card  
- [x] diplomacy
- [x] discovery tiles (parts)
- [x] warp portal
- [x] NPC difficulty
- [x] minor species
- [x] warped universe module
- [x] Optimize sector coordinate lookups in hot paths (e.g., diplomacy, combat) from O(C) to O(1) by adding a flat `std::array<HexCoord, 396> sector_coord_map` to [State](cci:2://file:///home/mihai/personal/open_spiel_eclipse/open_spiel/games/eclipse/state.h:180:0-268:1). To prevent stale values and keep serialization clean, rebuild the map in a single pass of the galaxy only after setup, explore tile placement, and deserialization ([from_json](cci:1://file:///home/mihai/personal/open_spiel_eclipse/open_spiel/games/eclipse/sectors.h:54:0-78:1)).



# Backend Gaps
- [ ] **REACTION actions**: `ActionType::REACTION` declared but never implemented. After passing, should allow 1 Activation of Upgrade/Build/Move.
- [x] **Neutron Absorber**: Rare tech — negates enemy NEUTRON BOMBS during pop attack (Planta's species weakness overrides).
- [x] **Artifact Key**: Nano tech — gain 5 Resources of a single type per Artifact on controlled sectors. Pending choice sub-action in research flow.
- [x] **Ancient sector discovery tile award in Combat Phase**: Deferred from explore, awarded by combat `discovery_award` step.
- [ ] **Geminga/Simeis 147 sectors**: Two special sectors granting extra actions commented out in `sectors.h`.

maybe later:
- [ ] galactic events module (extra sectors)
- [ ] supernova sector


# UI Integration Backlog (Not Yet Integrated)
- [ ] **Diplomacy & Ambassador Pacts**: Lacks visual diplomacy grid, pact indicator, or ambassador board space in the UI to manage alliances/pacts.
- [x] **Upgrade Action & Discovery Parts**: UI includes blueprint customization, discovery reward choice, and stored discovery part inventory.
- [x] **Warped Universe Module Selection**: Lobby/setup config can enable the module for supported player counts.
- [x] **Warped Universe Wormhole Visuals**: Layout loaders and selection are operational, but galaxy rendering still needs a focused visual review for warp-region/wormhole clarity.
- [ ] **Minor Species Module**: Ambassador relations and abilities have no UI presence or interactive components.
