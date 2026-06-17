// Action ids - mirror open_spiel/games/eclipse/eclipse.cc:36-52.
export const ACTION = {
  PASS: 0,
  RESEARCH: 1,
  RESEARCH_STOP: 2,
  RESEARCH_STANDARD_START: 3,
  RESEARCH_RARE_START: 27,
  EXPLORE: 83,
  EXPLORE_PLACE: 84,
  EXPLORE_DISCARD: 85,
  EXPLORE_ROT_START: 86, // +0..5
  EXPLORE_CLAIM_YES: 92,
  EXPLORE_CLAIM_NO: 93,
  EXPLORE_DISCOVERY_REWARD: 94,
  EXPLORE_DISCOVERY_VP: 95,
  EXPLORE_SELECT_TILE_START: 96, // +0..1
  EXPLORE_DRAW_AGAIN: 98,
  EXPLORE_SKIP_SECOND: 99,
  EXPLORE_STOP: 100,
  EXPLORE_ZONE_START: 101,        // + hex_to_index(q, r)
  TRADE_START: 326,              // + TradeConversion (0..5)
  COLONY_SHIP_START: 332,        // + cell*24 + slot*3 + track
  COLONY_SHIP_SLOTS_PER_CELL: 8,
  COLONY_SHIP_TRACKS: 3,         // 0=Money, 1=Science, 2=Materials
  COLONY_SHIP_CODES_PER_CELL: 24, // SLOTS_PER_CELL * TRACKS

  // Influence action IDs — mirror open_spiel/games/eclipse/eclipse.cc:75-79.
  INFLUENCE: 5732,
  INFLUENCE_STOP: 5733,
  INFLUENCE_TO_CELL_START: 5734,   // + galaxy cell index (0..224)
  RECLAIM_FROM_CELL_START: 5959,   // + galaxy cell index (0..224)
  CHOOSE_RETURN_TRACK_START: 6184, // + 0=Money, 1=Science, 2=Materials

  // Build action IDs — mirror open_spiel/games/eclipse/eclipse.cc:54,81-83.
  BUILD: 75,
  BUILD_STOP: 6187,
  BUILD_CHOICE_START: 6188,           // + type * GALAXY_CELL_COUNT + cell_idx
  BUILD_END: 7538,                    // BUILD_CHOICE_START + 6 * 225
  GALAXY_CELL_COUNT: 225,

  // Upgrade action IDs — mirror open_spiel/games/eclipse/eclipse.cc:86-91.
  UPGRADE: 7539,
  UPGRADE_STOP: 7540,
  UPGRADE_CHOICE_START: 7541,

  // Move action IDs — mirror open_spiel/games/eclipse/eclipse.cc:96-102.
  MOVE: 8502,
  MOVE_STOP: 8503,
  MOVE_CHOICE_START: 8504,               // + unit_idx * 7 + direction (0-5=hex, 6=warp)
  MOVE_WARP_START: 9400,                 // MOVE_CHOICE_START + 128 * 7
  MOVE_WARP_DESTINATION_START: 9528,     // MOVE_WARP_START + 128
  MOVE_CODES_PER_UNIT: 7,

  // Upkeep action IDs — mirror open_spiel/games/eclipse/eclipse.cc:105-106.
  // Colony placement / bankruptcy resolution during upkeep reuse the existing
  // COLONY_SHIP_START, RECLAIM_FROM_CELL_START, TRADE_START and
  // CHOOSE_RETURN_TRACK_START ranges (see AppendLegalUpkeepColonyShipActions /
  // AppendReclaimActions in eclipse.cc).
  UPKEEP_COLONY_DONE: 9753,
  UPKEEP_PAY_DONE: 9754,

  // Combat action IDs — mirror open_spiel/games/eclipse/eclipse.cc:109-126.
  COMBAT_CONTINUE: 9755,
  COMBAT_ATTACK: 9756,
  COMBAT_RETREAT_TO_CELL_START: 9757,    // + galaxy cell index (0..224)
  COMBAT_DICE_TARGET_START: 9982,        // + unit_registry index (0..127)
  COMBAT_REP_SELECT_START: 10110,        // + drawn-tile index (0..4)
  COMBAT_REP_SKIP: 10115,
  COMBAT_POP_TARGET_START: 10116,        // + population slot (0..15)
  COMBAT_INFLUENCE_YES: 10132,
  COMBAT_INFLUENCE_NO: 10133,
  COMBAT_DISCOVERY_REWARD: 10134,
  COMBAT_DISCOVERY_VP: 10135,
  COMBAT_INFLUENCE_TO_CELL_START: 10136, // + galaxy cell index (0..224)
} as const;

// Population track names, indexed by track id (mirrors PopTrack in bonus.h).
export const POP_TRACK_LABELS = ['💰', '🔬', '⚙️'] as const;

// TradeConversion indices (mirroring bonus.h enum)
export const TRADE = {
  GOLD_TO_SCIENCE: 0,
  GOLD_TO_MATERIALS: 1,
  SCIENCE_TO_GOLD: 2,
  SCIENCE_TO_MATERIALS: 3,
  MATERIALS_TO_GOLD: 4,
  MATERIALS_TO_SCIENCE: 5,
} as const;

export const TRADE_LABELS: Record<number, { from: string; to: string; emoji: string }> = {
  0: { from: 'gold',      to: 'science',   emoji: '💰→🔬' },
  1: { from: 'gold',      to: 'materials', emoji: '💰→⚙️' },
  2: { from: 'science',   to: 'gold',      emoji: '🔬→💰' },
  3: { from: 'science',   to: 'materials', emoji: '🔬→⚙️' },
  4: { from: 'materials', to: 'gold',      emoji: '⚙️→💰' },
  5: { from: 'materials', to: 'science',   emoji: '⚙️→🔬' },
};
