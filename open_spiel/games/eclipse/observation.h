//
// Observation tensor layout for Eclipse — the single source of truth.
//
// Every field of ::State is exposed here, with exactly one deliberate
// exception: Sector::discovery_tile. That field holds the identity of a tile
// that is still FACE DOWN (it is written at placement time in
// explore.cpp:472-477 / setup.cpp:257-258 and only cleared to NONE when a
// player claims it). Exposing it would let the agent read unclaimed discovery
// tiles, which is both cheating and would make a trained agent unfair to play
// against. Only `discovery_tile_present` is exposed.
//
// Things that look hidden but are legitimately deducible from public play ARE
// exposed: the sector-ring bag bitmasks (the ring tile lists are public and
// every placement is public) and the per-value remaining counts of the
// reputation-tile bag. The `tech_bag` and `discovery_bag` *contents* remain
// counts only.
//
// Everything is written SEAT-RELATIVE from the viewing player's perspective:
// player block 0 is always the viewer, blocks 1..5 are the other seats in
// increasing seat order wrapping around. Every player id stored in the state
// (combat attacker, ambassador_from, upkeep player, ...) is likewise remapped
// through RelativeSeat() so the tensor is canonical.
//
// Python mirror: open_spiel/python/eclipse/obs_layout.py. That file asserts its
// TOTAL == game.observation_tensor_shape()[0], so a change here fails loudly
// there instead of silently mis-reshaping.
//
#ifndef OPEN_SPIEL_GAMES_ECLIPSE_OBSERVATION_H_
#define OPEN_SPIEL_GAMES_ECLIPSE_OBSERVATION_H_

#include <cstdint>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/eclipse/state.h"

namespace open_spiel {
namespace eclipse {
namespace obs {

// ── Cardinalities, all pinned to the engine's own enums/tables ─────────────
constexpr int kMaxSeats = MAX_PLAYERS;            // 6
constexpr int kSeatSlots = kMaxSeats;             // one block per seat, viewer first
constexpr int kSpeciesCount = 7;                  // Species (resources.h:19-27)
constexpr int kShipTypeCount = 7;                 // ShipType (types.h:13)
constexpr int kPlayerShipTypes = 4;               // INTERCEPTOR..STARBASE -> blueprints
constexpr int kNpcShipTypes = 3;                  // ANCIENT, GUARDIAN, GCDS
constexpr int kPlanetTypeCount = 8;               // PlanetType (sectors.h:82-86)
constexpr int kDieColorCount = 5;                 // DieColor YELLOW..PURPLE (dice.h)
constexpr int kShipPartCount = 43;                // ShipPartId minus NONE (tech.h:151-196)
constexpr int kBlueprintSlots = 8;                // Blueprint::slots[8] (tech.h:343)
constexpr int kTechBitCount = 40;                 // TechBit bits 1..40 (tech.h:23-69)
constexpr int kTechTrackCount = 3;                // MILITARY, GRID, NANO
constexpr int kTechTrayCount = 40;                // State::tech_tray (state.h:214)
constexpr int kTechTrackCapacity = 8;             // research.cpp:76-79
constexpr int kBuildTypeCount = 6;                // BuildType (build.h:36-44)
constexpr int kRepSlots = 5;                      // Player::reputation_track cap
constexpr int kRepSlotKindCount = 3;              // ReputationSlotKind (types.h:39-43)
constexpr int kRepTileValueCount = 5;             // ONE..FOUR, NONE (types.h:25)
constexpr int kMinorSpeciesCount = MINOR_SPECIES_COUNT;  // 9
constexpr int kSectorTypeCount = 7;               // SectorType (sectors.h:92)
constexpr int kNpcDifficultyCount = 3;            // EASY, MEDIUM, HARD
constexpr int kNpcTypeCount = 3;                  // GCDS, GUARDIAN, ANCIENT
constexpr int kRoundPhaseCount = 4;               // RoundPhase (state.h:89-94)
constexpr int kMaxRounds = 9;                     // terminal at current_round > 8
constexpr int kHexDirections = 6;
constexpr int kGalaxyCells = GALAXY_CELL_COUNT;   // 225 (dense 15x15)
constexpr int kGalaxyDim = MAP_SIZE;             // 15

// Relative-seat one-hot width: viewer, 5 others, NPC, unowned/none.
constexpr int kRelSeatWidth = kMaxSeats + 2;      // 8

// Sub-state phase widths (each matches its enum exactly).
constexpr int kExplorePhaseCount = 10;            // explore.h:19-30
constexpr int kResearchPhaseCount = 2;
constexpr int kBuildPhaseCount = 2;
constexpr int kInfluencePhaseCount = 3;
constexpr int kUpgradePhaseCount = 2;
constexpr int kMovePhaseCount = 3;
constexpr int kDiplomacyPhaseCount = 6;           // diplomacy.h:37-44
constexpr int kCombatPhaseCount = 11;             // combat.h:141-153
constexpr int kUpkeepStepCount = 5;               // state.h:184-190

// Combat container capacities (combat.h).
constexpr int kBattleQueueCap = 8;
constexpr int kInitiativeCap = 16;
constexpr int kRetreatingCap = 16;
constexpr int kRetreatDestCap = 6;
constexpr int kRepDrawCap = 5;
constexpr int kParticipantsCap = 6;
constexpr int kPendingTargetCap = 128;            // unit_registry capacity
constexpr int kDieFaces = 6;

// Bounded queue exposure for the two unbounded std::vector queues.
constexpr int kPendingReturnCap = 8;

// ── V2 keyed entity block ────────────────────────────────────────────────
// These tables are deliberately appended as a checkpoint-incompatible V2
// extension. Rows retain the engine/action keys instead of pooling them away.
constexpr int kDiscoveryBitCount = 30;
constexpr int kUnitRows = 128;
constexpr int kPlanetSlotsPerCell = 8;
constexpr int kPlanetSlotRows = kGalaxyCells * kPlanetSlotsPerCell;
constexpr int kUnitRowSize =
    1 +                     // valid / registry key is row index
    kRelSeatWidth +          // owner, viewer-relative
    kShipTypeCount +         // type
    1 +                      // source cell id
    2 +                      // source q, r
    1 +                      // damage
    1 +                      // arrival order
    1 +                      // active move
    1 +                      // pending warp
    1;                       // legal combat-die target
constexpr int kUnitRouteSize = kHexDirections;  // destination cell per direction; -1 is zero
constexpr int kPlanetSlotSize = 4;              // valid, type, occupied, orbital
constexpr int kV2GlobalSize =
    kMaxSeats +              // viewer absolute-seat routing key
    kTechBitCount +          // exact tech-bag histogram
    kDiscoveryBitCount +     // public revealed-discovery histogram
    (kDiscoveryBitCount + 1);  // currently revealed discovery, NONE at 0
constexpr int kV2SeatSize = 2 + kTechTrackCount * kTechBitCount;
// valid, absolute-seat routing key, then the three distinct 40-bit tracks.
constexpr int kV2CellSize = 2;  // sector definition id, rotation
constexpr int kV2BattleRecordSize = 3 + kParticipantsCap * 2;
constexpr int kV2DestroyedRecordSize = 4;
constexpr int kV2DieRecordSize = 2;
constexpr int kV2RetreatRecordSize = 4;
constexpr int kV2CombatSize =
    kBattleQueueCap * kV2BattleRecordSize +
    32 * kV2DestroyedRecordSize +
    kInitiativeCap * kShipTypeCount +
    64 * kV2DieRecordSize +
    kRetreatingCap * kV2RetreatRecordSize +
    1;  // population-attack target cell
constexpr int kV2KeyedSize =
    kV2GlobalSize + kSeatSlots * kV2SeatSize + kGalaxyCells * kV2CellSize +
    kUnitRows * kUnitRowSize + kUnitRows * kUnitRouteSize +
    kPlanetSlotRows * kPlanetSlotSize + kV2CombatSize;

// ── Feature-group sizes ───────────────────────────────────────────────────
// A tile's intrinsic, publicly-visible shape. Used both per galaxy cell and
// for the tiles held in ExploreState (so the agent can actually evaluate the
// tile it drew before choosing rotate/place/discard).
constexpr int kTileFeatureSize =
    1                       // valid
    + 1                     // points (VP)
    + kHexDirections        // unrotated wormhole edges
    + kPlanetTypeCount      // printed planet slots by type
    + kSectorTypeCount      // ring/type one-hot
    + 1                     // has_artifact
    + 1                     // has_guardian
    + 1                     // is_gcds
    + 1                     // has_warp_portal
    + 1                     // start_with_discovery
    + 1;                    // starting_ancients

// One ship blueprint: derived stats + which parts are mounted + slot shape.
constexpr int kShipStatsSize =
    1                       // initiative
    + 1                     // computer
    + 1                     // shield
    + 1                     // energy_net
    + 1                     // hull
    + 1                     // movement
    + kDieColorCount        // cannons by die colour
    + kDieColorCount;       // missiles by die colour
constexpr int kBlueprintSize =
    kShipStatsSize
    + kShipPartCount        // count of each part mounted
    + kBlueprintSlots       // per-slot occupied
    + 1;                    // capacity

constexpr int kRepSlotSize =
    kRepSlotKindCount       // kind one-hot
    + 1                     // holds_ambassador
    + kRelSeatWidth         // ambassador_from, seat-relative
    + kRepTileValueCount    // rep_value one-hot
    + 1;                    // pending_track_choice

// ── Block A: global ───────────────────────────────────────────────────────
constexpr int kGlobalSize =
    1                       // round / kMaxRounds
    + kMaxRounds            // round one-hot
    + kRoundPhaseCount      // phase one-hot
    + kRelSeatWidth         // current_player, seat-relative
    + 1                     // is the viewer to move
    + (kMaxSeats - 1)       // num_players one-hot (2..6)
    + 1                     // warped_universe
    + kNpcDifficultyCount * kNpcTypeCount   // difficulties as one-hots
    + kNpcTypeCount * 8     // NPC combat stats per type
    + 3                     // sector bag popcounts
    + 10 + 16 + 22          // sector bag bitmasks (inner/middle/outer)
    + 1                     // tech_bag size
    + 1                     // discovery_bag size
    + 1                     // reputation_tiles bag size
    + 4                     // reputation bag remaining per value
    + kMinorSpeciesCount    // minor_species_pool bitmap
    + kRelSeatWidth         // minor_species_pending_track
    + 1                     // next_arrival_order (normalised)
    + 8;                    // reserve

// ── Block B: one identical block per seat ─────────────────────────────────
constexpr int kPlayerSize =
    1                       // slot occupied
    + 1                     // is the viewer
    + 1                     // is_ai
    + 1                     // eliminated
    + 1                     // has_passed
    + kSpeciesCount         // species one-hot
    + 6                     // species action activations
    + kMaxSeats             // turn-order position one-hot
    + 1                     // turn-order position known
    + (kMaxSeats + 1)       // pass-order position one-hot (0 = not passed)
    + 1                     // live total VP
    + 9                     // live VP by scoring category
    + 1                     // vp_at_elimination
    + 1                     // vp_at_elimination valid
    + 3                     // gold / science / materials
    + 3                     // population-track indices
    + 3                     // ACTUAL production per track
    + 1                     // PlayerIncome (money)
    + 1                     // PlayerUpkeepCost
    + 1                     // net cash flow
    + 1                     // IsPlayerSolvent
    + 5                     // disks: sectors/actions/reactions/available/extra
    + 2                     // colony ships total/available
    + 2                     // orbitals / monoliths
    + 1                     // trade_rate
    + 3                     // graveyard_counts
    + kTechBitCount         // researched techs, all 40 bits
    + kTechTrackCount       // tiles per track
    + kTechTrackCount       // remaining track capacity
    + kPlayerShipTypes * kBlueprintSize
    + kShipPartCount        // parts_inventory counts
    + 1                     // parts_inventory total
    + kRepSlots * kRepSlotSize
    + 1                     // ambassador_tiles_held
    + 1                     // ambassador_tiles_pending_return
    + 1                     // traitor_held
    + 1                     // discovery_vp_tiles_kept
    + kMinorSpeciesCount    // owned_minor_species bitmap
    + 1                     // warp_portal_eligible
    + 1                     // pending_artifact_key_chunks
    + kBuildTypeCount       // current build cost per BuildType
    + 4;                    // reserve

// Named offsets INTO a player block, for the field groups that tests and the
// Python mirror need to address. WriteObservationTensor asserts its running
// cursor against each of these, so a wrong constant fails loudly rather than
// silently addressing the wrong float.
constexpr int kPlayerOccupiedOffset = 0;
constexpr int kPlayerIsViewerOffset = 1;
constexpr int kPlayerIsAiOffset = 2;
constexpr int kPlayerEliminatedOffset = 3;
constexpr int kPlayerHasPassedOffset = 4;
constexpr int kPlayerSpeciesOffset = 5;              // + kSpeciesCount
constexpr int kPlayerActivationsOffset = 12;         // + 6
constexpr int kPlayerTurnPosOffset = 18;             // + kMaxSeats
constexpr int kPlayerPassPosOffset = 25;             // + kMaxSeats + 1
constexpr int kPlayerVpTotalOffset = 32;
constexpr int kPlayerVpBreakdownOffset = 33;         // + 9 categories
constexpr int kPlayerVpAtElimOffset = 42;
constexpr int kPlayerGoldOffset = 44;
constexpr int kPlayerScienceOffset = 45;
constexpr int kPlayerMaterialsOffset = 46;
constexpr int kPlayerProdIndexOffset = 47;           // + 3 (cubes remaining)
constexpr int kPlayerProductionOffset = 50;          // + 3 (ACTUAL production)
constexpr int kPlayerIncomeOffset = 53;
constexpr int kPlayerUpkeepCostOffset = 54;
constexpr int kPlayerNetCashOffset = 55;
constexpr int kPlayerSolventOffset = 56;
constexpr int kPlayerDisksOnSectorsOffset = 57;
constexpr int kPlayerDisksAvailableOffset = 60;
constexpr int kPlayerColonyTotalOffset = 62;
constexpr int kPlayerColonyAvailOffset = 63;
constexpr int kPlayerOrbitalsOffset = 64;
constexpr int kPlayerMonolithsOffset = 65;
constexpr int kPlayerTradeRateOffset = 66;
constexpr int kPlayerGraveyardOffset = 67;           // + 3
constexpr int kPlayerTechBitsOffset = 70;
constexpr int kPlayerBlueprintsOffset = kPlayerTechBitsOffset + kTechBitCount
                                        + 2 * kTechTrackCount;          // 116
constexpr int kPlayerPartsInvOffset =
    kPlayerBlueprintsOffset + kPlayerShipTypes * kBlueprintSize;        // 388
constexpr int kPlayerRepTrackOffset = kPlayerPartsInvOffset + kShipPartCount + 1;  // 432
// Within one reputation slot: the partner's seat-relative one-hot.
constexpr int kRepSlotAmbassadorFromOffset = kRepSlotKindCount + 1;     // 4
constexpr int kPlayerAmbassadorHeldOffset =
    kPlayerRepTrackOffset + kRepSlots * kRepSlotSize;                   // 522
constexpr int kPlayerTraitorOffset = kPlayerAmbassadorHeldOffset + 2;   // 524
constexpr int kPlayerDiscoveryVpOffset = kPlayerAmbassadorHeldOffset + 3;
constexpr int kPlayerMinorSpeciesOffset = kPlayerAmbassadorHeldOffset + 4;
constexpr int kPlayerWarpEligibleOffset =
    kPlayerMinorSpeciesOffset + kMinorSpeciesCount;                     // 535
constexpr int kPlayerArtifactChunksOffset = kPlayerWarpEligibleOffset + 1;
constexpr int kPlayerBuildCostOffset = kPlayerWarpEligibleOffset + 2;   // + kBuildTypeCount

// ── Block C: galaxy, dense 15x15 x channels ───────────────────────────────
// Channel order is CELL-MAJOR / CHANNEL-MINOR: index = C + cell*kCellChannels
// + channel. The Python spatial encoder reshapes (15,15,C) then permutes to
// (C,15,15) and depends on exactly this convention.
// The galaxy is 76% of the whole tensor (225x multiplier), so this channel set
// is the ONLY sensible trim lever if the throughput/GPU-memory measurement
// says the tensor is too wide. Trim from the bottom of this list up; never trim
// the player, combat or sub-state blocks — together they are under a quarter of
// the size and carry the information the agent was most blind to.
constexpr int kRingKinds = 4;   // inner / middle / outer / special(centre,start,guardian,warp)
enum CellChannel : int {
  kCellPresent = 0,
  kCellOwner,                                   // + kRelSeatWidth
  kCellPoints = kCellOwner + kRelSeatWidth,
  kCellWormhole,                                // + kHexDirections (ROTATED)
  kCellPlanetPrinted = kCellWormhole + kHexDirections,   // + kPlanetTypeCount
  kCellPlanetPopulated = kCellPlanetPrinted + kPlanetTypeCount,  // + kPlanetTypeCount
  kCellOrbital = kCellPlanetPopulated + kPlanetTypeCount,
  kCellMonolith,
  kCellOrbitalPopSlot,          // the Orbital's own virtual Money slot
  kCellDiscoveryPresent,        // presence ONLY — never the tile identity
  kCellWarpPortalVp,
  kCellHasWarpPortal,
  kCellHasArtifact,
  kCellHasGuardian,
  kCellIsGcds,
  kCellRing,                                    // + kRingKinds
  kCellMyShips = kCellRing + kRingKinds,        // + kPlayerShipTypes
  kCellEnemyShips = kCellMyShips + kPlayerShipTypes,     // + kPlayerShipTypes
  kCellNpcShips = kCellEnemyShips + kPlayerShipTypes,    // + kNpcShipTypes
  kCellDamage = kCellNpcShips + kNpcShipTypes,
  kCellMyAnchor,                // owner==me or a ship of mine is here
  kCellCombatActive,            // == combat_state.active_sector_id
  kCellInBattleQueue,
  kCellInfluenceUncontrolled,   // in combat_state.influence_uncontrolled_sectors
  kCellMoveActiveUnit,          // move_state.active_unit_idx sits here
  kCellExploreZone,             // explore_state's chosen zone hex
  kCellWarpLink,                                // + kHexDirections (warped only)
  kCellLayoutKind = kCellWarpLink + kHexDirections,   // + kSectorTypeCount (warped
                                                      // only: layout_kinds[cell], the
                                                      // per-cell inner/mid/outer/warp
                                                      // tag that gates explorability)
  kCellWarpDestCell = kCellLayoutKind + kSectorTypeCount,   // + kHexDirections (warped
                                                            // only: normalised warp
                                                            // destination cell, 0 if none)
  kCellWarpDestDir = kCellWarpDestCell + kHexDirections,    // + kHexDirections (warped
                                                            // only: arrival edge, 0 if none)
  kCellChannels = kCellWarpDestDir + kHexDirections,
};
constexpr int kGalaxySize = kGalaxyCells * kCellChannels;

// ── Block D: tech market ──────────────────────────────────────────────────
constexpr int kTechMarketSize =
    kTechTrayCount          // copies available per tech
    + kTechTrayCount        // cost to the VIEWER right now, per tech
    + kTechTrackCount       // cheapest available cost per track
    + 5;                    // reserve

// ── Block E: combat ───────────────────────────────────────────────────────
constexpr int kBattleEntrySize =
    1                       // valid
    + 1                     // sector_id (normalised)
    + 1                     // participant_count
    + kMaxSeats             // participant seat bitmap (relative)
    + kRelSeatWidth;        // defender, seat-relative
constexpr int kInitiativeEntrySize =
    1                       // valid
    + kRelSeatWidth         // owner, seat-relative
    + kShipTypeCount        // ship type one-hot
    + 1                     // initiative value
    + 1                     // is_npc
    + 1                     // destroyed
    + 1                     // retreating
    + 1                     // alive_count
    + 1;                    // destroyed_count
constexpr int kCombatSize =
    1                       // active
    + kCombatPhaseCount     // phase one-hot
    + 1                     // active_sector_id
    + 1                     // engagement_round
    + kRelSeatWidth         // current_attacker_id
    + kRelSeatWidth         // current_defender_id
    + kRelSeatWidth         // pending_player
    + 1 + 1                 // battle_queue_size, current_battle_idx
    + kBattleQueueCap * kBattleEntrySize
    + 1 + 1                 // initiative_size, initiative_idx
    + kInitiativeCap * kInitiativeEntrySize
    + kRelSeatWidth * kShipTypeCount   // destroyed ships by (seat, type)
    + kShipTypeCount + 1    // ship_order_queue counts + idx
    + kShipTypeCount        // active_ship_type one-hot
    + kRetreatDestCap + 1   // retreat_destinations + size
    + kRepDrawCap * kRepTileValueCount + 1   // drawn_tiles + size
    + kRelSeatWidth         // tile_select_player
    + kRelSeatWidth         // rep_draw_target
    + kParticipantsCap      // reputation_drawn_mask
    + kParticipantsCap      // reputation_retreat_penalty_mask
    + 1                     // reputation_earned
    + kRelSeatWidth         // pending_target_group_player
    + kShipTypeCount        // pending_target_group_type
    + kPendingTargetCap     // pending_target_indices bitmap
    + 1                     // pending_target_count
    + kDieFaces             // pending die value histogram
    + kDieColorCount + 1    // pending die colour histogram (+NONE)
    + kDieFaces             // current die value one-hot
    + kDieColorCount + 1    // current die colour one-hot
    + 1 + 1                 // pending_die_count, pending_die_index
    + 1 + 1                 // pending_dice_are_missiles, ..._pop_attack
    + kRelSeatWidth * kShipTypeCount   // retreating groups by (seat, type)
    + 1                     // retreating_group_count
    + 1                     // pop_attack_sector_id
    + kRelSeatWidth         // pop_attack_player
    + kRelSeatWidth         // pop_attack_owner
    + 1                     // pop_attack_damage_remaining
    + 1                     // pop_attack_unit_index
    + 1                     // influence_uncontrolled_size
    + 1                     // influence_scan_index
    + 1                     // influence_turn_order_index
    + kRelSeatWidth         // influence_decision_player
    + 1                     // influence_decision_sector
    + 1                     // discovery_decision_sector
    + kRelSeatWidth         // discovery_decision_player
    + 8;                    // reserve

// ── Block F: upkeep ───────────────────────────────────────────────────────
constexpr int kPendingReturnSize = kPlanetTypeCount + 1;  // type one-hot + is_orbital
constexpr int kUpkeepSize =
    1                       // active
    + kUpkeepStepCount      // step one-hot
    + kRelSeatWidth         // player_id
    + 1                     // pending_returns size
    + kPendingReturnCap * kPendingReturnSize
    + 4;                    // reserve

// ── Block G: in-flight action sub-states ──────────────────────────────────
constexpr int kExploreSize =
    1                       // active
    + kExplorePhaseCount    // phase one-hot
    + kRelSeatWidth         // player_id
    + 1                     // activations_remaining
    + 1 + 1                 // zone_q, zone_r (normalised)
    + 1                     // zone target known
    + kSectorTypeCount      // ring one-hot
    + 1                     // drawn_count
    + 2 * kTileFeatureSize  // the drawn tiles, fully described
    + kTileFeatureSize      // the selected tile
    + kHexDirections        // chosen_rotation one-hot
    + 1                     // rotation chosen
    + kHexDirections        // selected tile's ROTATED wormhole edges
    + kShipPartCount + 1;   // discovered_part one-hot (+none)
constexpr int kSimpleActionSize(int phase_count) {
  return 1 + phase_count + kRelSeatWidth + 1;
}
constexpr int kInfluenceSubSize =
    kSimpleActionSize(kInfluencePhaseCount)
    + 1                     // pending_returns size
    + kPendingReturnCap * kPendingReturnSize;
constexpr int kMoveSubSize =
    kSimpleActionSize(kMovePhaseCount)
    + 1                     // has active unit
    + kShipTypeCount        // active unit ship type
    + 1                     // active unit damage
    + 1                     // steps_remaining
    + 1                     // has warp unit
    + kShipTypeCount;       // warp unit ship type
constexpr int kDiplomacySubSize =
    1                       // active
    + kDiplomacyPhaseCount  // phase one-hot
    + kRelSeatWidth         // player_id (proposer / chooser)
    + kRelSeatWidth         // partner_id
    + 3                     // rearrange_side (0/1/none)
    + 3                     // pop_track_side
    + 3                     // selected_track
    + 3;                    // return_side
constexpr int kActionStatesSize =
    kExploreSize
    + kSimpleActionSize(kResearchPhaseCount)
    + kSimpleActionSize(kBuildPhaseCount)
    + kSimpleActionSize(kUpgradePhaseCount)
    + kInfluenceSubSize
    + kMoveSubSize
    + kDiplomacySubSize
    + 4;                    // reserve

// ── Block offsets ─────────────────────────────────────────────────────────
constexpr int kGlobalStart = 0;
constexpr int kPlayersStart = kGlobalStart + kGlobalSize;
constexpr int kGalaxyStart = kPlayersStart + kSeatSlots * kPlayerSize;
constexpr int kTechMarketStart = kGalaxyStart + kGalaxySize;
constexpr int kCombatStart = kTechMarketStart + kTechMarketSize;
constexpr int kUpkeepStart = kCombatStart + kCombatSize;
constexpr int kActionStatesStart = kUpkeepStart + kUpkeepSize;
constexpr int kV2KeyedStart = kActionStatesStart + kActionStatesSize;
constexpr int kTotalSize = kV2KeyedStart + kV2KeyedSize;

constexpr int PlayerBlockStart(int slot) {
  return kPlayersStart + slot * kPlayerSize;
}
constexpr int CellStart(int cell) {
  return kGalaxyStart + cell * kCellChannels;
}
constexpr int V2UnitStart(int unit) {
  return kV2KeyedStart + kV2GlobalSize + kSeatSlots * kV2SeatSize +
         kGalaxyCells * kV2CellSize +
         unit * kUnitRowSize;
}
constexpr int V2UnitRoutesStart(int unit) {
  return kV2KeyedStart + kV2GlobalSize + kSeatSlots * kV2SeatSize +
         kGalaxyCells * kV2CellSize +
         kUnitRows * kUnitRowSize + unit * kUnitRouteSize;
}
constexpr int V2PlanetSlotStart(int cell, int slot) {
  return kV2KeyedStart + kV2GlobalSize + kSeatSlots * kV2SeatSize +
         kGalaxyCells * kV2CellSize +
         kUnitRows * (kUnitRowSize + kUnitRouteSize) +
         (cell * kPlanetSlotsPerCell + slot) * kPlanetSlotSize;
}

// Writes the full observation for `player` into `values` (size kTotalSize).
void WriteObservationTensor(const ::State& state, int player, int num_players,
                            absl::Span<float> values);

}  // namespace obs
}  // namespace eclipse
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_ECLIPSE_OBSERVATION_H_
