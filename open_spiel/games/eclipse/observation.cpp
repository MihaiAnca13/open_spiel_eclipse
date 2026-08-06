#include "open_spiel/games/eclipse/observation.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "open_spiel/games/eclipse/systems/actions/build.h"
#include "open_spiel/games/eclipse/systems/actions/research.h"
#include "open_spiel/games/eclipse/systems/scoring.h"
#include "open_spiel/games/eclipse/systems/upkeep.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace eclipse {
namespace obs {
namespace {

// ── primitive writers ─────────────────────────────────────────────────────
inline void Frac(absl::Span<float> v, int i, float x, float max_v) {
  v[i] = max_v > 0.0f ? std::min(std::max(x / max_v, -1.0f), 1.0f) : 0.0f;
}
inline void OneHot(absl::Span<float> v, int base, int idx, int count) {
  if (idx >= 0 && idx < count) v[base + idx] = 1.0f;
}
inline void Bits(absl::Span<float> v, int base, uint64_t mask, int count) {
  for (int i = 0; i < count; ++i) {
    v[base + i] = ((mask >> i) & 1ull) ? 1.0f : 0.0f;
  }
}
inline void Flag(absl::Span<float> v, int i, bool b) { v[i] = b ? 1.0f : 0.0f; }

// ── seat remapping — everything in the tensor is viewer-relative ──────────
// Slot 0 is always the viewer; slots 1..num_players-1 are the other seats in
// increasing seat order, wrapping. Trailing slots stay zero (occupied = 0).
inline int SeatForSlot(int slot, int viewer, int num_players) {
  return (viewer + slot) % num_players;
}

// Index into a kRelSeatWidth one-hot. 0..5 = a real seat (viewer-relative),
// kMaxSeats = an NPC, kMaxSeats+1 = none/invalid.
constexpr int kRelNpc = kMaxSeats;
constexpr int kRelNone = kMaxSeats + 1;

inline int RelSeat(int id, int viewer, int num_players) {
  if (id < 0 || id >= num_players) return kRelNone;
  return (id - viewer + num_players) % num_players;
}
// For unit ownership, where 255 means "NPC" rather than "nobody".
inline int RelSeatOrNpc(int id, int viewer, int num_players) {
  if (id == NPC_PLAYER_ID) return kRelNpc;
  return RelSeat(id, viewer, num_players);
}
// For sector ownership: <num_players is a player, 255 is unowned, anything
// else in between is NPC-held (mirrors the pre-existing encoding).
inline int RelSectorOwner(uint8_t owner, int viewer, int num_players) {
  if (owner < static_cast<uint8_t>(num_players)) {
    return RelSeat(owner, viewer, num_players);
  }
  return owner == 255 ? kRelNone : kRelNpc;
}

// Which of the four coarse ring kinds a sector type belongs to.
inline int RingKind(SectorType t) {
  switch (t) {
    case SectorType::INNER: return 0;
    case SectorType::MIDDLE: return 1;
    case SectorType::OUTER: return 2;
    default: return 3;  // CENTER / STARTING / GUARDIAN / WARP
  }
}

// ── shared feature writers ────────────────────────────────────────────────
// A tile's intrinsic public shape. Used per galaxy cell and for the tiles held
// in ExploreState, so the agent can evaluate a drawn tile before deciding
// rotate / place / discard. Returns the number of floats written.
int WriteTileFeatures(absl::Span<float> v, int base, uint16_t sector_id) {
  int o = base;
  const SectorDefinition* def =
      sector_id == 0 ? nullptr : get_sector_definition(sector_id);
  Flag(v, o++, def != nullptr);
  if (def == nullptr) {
    o = base + kTileFeatureSize;
    return kTileFeatureSize;
  }
  Frac(v, o++, static_cast<float>(def->points), 4.0f);
  Bits(v, o, def->wormholes_mask, kHexDirections);
  o += kHexDirections;
  for (const PlanetSlot& s : def->slots) {
    int t = static_cast<int>(s.type);
    if (t >= 0 && t < kPlanetTypeCount) v[o + t] += 1.0f / 6.0f;
  }
  o += kPlanetTypeCount;
  OneHot(v, o, static_cast<int>(def->type), kSectorTypeCount);
  o += kSectorTypeCount;
  Flag(v, o++, def->has_artifact);
  Flag(v, o++, def->has_guardian);
  Flag(v, o++, def->is_gcds);
  Flag(v, o++, def->has_warp_portal);
  Flag(v, o++, def->start_with_discovery);
  Frac(v, o++, static_cast<float>(def->starting_ancients), 4.0f);
  SPIEL_CHECK_EQ(o, base + kTileFeatureSize);
  return kTileFeatureSize;
}

int WriteShipStats(absl::Span<float> v, int base, const ShipStats& s) {
  int o = base;
  Frac(v, o++, static_cast<float>(s.initiative), 8.0f);
  Frac(v, o++, static_cast<float>(s.computer), 6.0f);
  Frac(v, o++, static_cast<float>(s.shield), 6.0f);
  Frac(v, o++, static_cast<float>(s.energy_net), 12.0f);
  Frac(v, o++, static_cast<float>(s.hull), 8.0f);
  Frac(v, o++, static_cast<float>(s.movement), 6.0f);
  for (int c = 0; c < kDieColorCount; ++c) {
    Frac(v, o++, static_cast<float>(s.cannons[c]), 4.0f);
  }
  for (int c = 0; c < kDieColorCount; ++c) {
    Frac(v, o++, static_cast<float>(s.missiles[c]), 4.0f);
  }
  SPIEL_CHECK_EQ(o, base + kShipStatsSize);
  return kShipStatsSize;
}

int WriteBlueprint(absl::Span<float> v, int base, const Blueprint& bp) {
  int o = base;
  o += WriteShipStats(v, o, bp.total_stats);
  const int parts_base = o;
  for (int i = 0; i < kBlueprintSlots; ++i) {
    const int part = static_cast<int>(bp.slots[i]);
    if (part > 0 && part <= kShipPartCount) {
      v[parts_base + part - 1] += 1.0f / 4.0f;
    }
  }
  o += kShipPartCount;
  for (int i = 0; i < kBlueprintSlots; ++i) {
    Flag(v, o + i, bp.slots[i] != ShipPartId::NONE);
  }
  o += kBlueprintSlots;
  Frac(v, o++, static_cast<float>(bp.capacity),
       static_cast<float>(kBlueprintSlots));
  SPIEL_CHECK_EQ(o, base + kBlueprintSize);
  return kBlueprintSize;
}

int WritePendingReturns(absl::Span<float> v, int base,
                        const std::vector<PendingReturn>& q) {
  int o = base;
  Frac(v, o++, static_cast<float>(q.size()), 20.0f);
  const int n = std::min<int>(q.size(), kPendingReturnCap);
  for (int i = 0; i < n; ++i) {
    const int e = o + i * kPendingReturnSize;
    OneHot(v, e, static_cast<int>(q[i].type), kPlanetTypeCount);
    Flag(v, e + kPlanetTypeCount, q[i].is_orbital);
  }
  o += kPendingReturnCap * kPendingReturnSize;
  return o - base;
}

}  // namespace

// ══════════════════════════════════════════════════════════════════════════
void WriteObservationTensor(const ::State& state, int player, int num_players,
                            absl::Span<float> values) {
  SPIEL_CHECK_EQ(static_cast<int>(values.size()), kTotalSize);
  std::fill(values.begin(), values.end(), 0.0f);
  const auto scores = compute_all_player_scores(state);

  // ── Block A: global ─────────────────────────────────────────────────────
  {
    int o = kGlobalStart;
    Frac(values, o++, static_cast<float>(state.current_round),
         static_cast<float>(kMaxRounds));
    OneHot(values, o, static_cast<int>(state.current_round), kMaxRounds);
    o += kMaxRounds;
    OneHot(values, o, static_cast<int>(state.current_phase), kRoundPhaseCount);
    o += kRoundPhaseCount;
    OneHot(values, o, RelSeat(state.current_player, player, num_players),
           kRelSeatWidth);
    o += kRelSeatWidth;
    Flag(values, o++, state.current_player == player);
    OneHot(values, o, num_players - 2, kMaxSeats - 1);
    o += kMaxSeats - 1;
    Flag(values, o++, state.warped_universe);

    const NPCDifficulty diffs[kNpcTypeCount] = {
        state.gcds_difficulty, state.guardian_difficulty,
        state.ancient_difficulty};
    for (int t = 0; t < kNpcTypeCount; ++t) {
      OneHot(values, o, static_cast<int>(diffs[t]), kNpcDifficultyCount);
      o += kNpcDifficultyCount;
    }
    // NPC combat profiles, so the agent can price a fight without a lookup.
    const NPCType npc_types[kNpcTypeCount] = {
        NPCType::GCDS, NPCType::GUARDIAN, NPCType::ANCIENT};
    for (int t = 0; t < kNpcTypeCount; ++t) {
      const NPC* npc = nullptr;
      for (const NPC& cand : NPC_TABLE) {
        if (cand.type == npc_types[t] && cand.difficulty == diffs[t]) {
          npc = &cand;
          break;
        }
      }
      if (npc != nullptr) {
        Frac(values, o + 0, static_cast<float>(npc->cannon_amount), 4.0f);
        Frac(values, o + 1, static_cast<float>(npc->missile_amount), 4.0f);
        Frac(values, o + 2, static_cast<float>(npc->computer), 6.0f);
        Frac(values, o + 3, static_cast<float>(npc->shield), 6.0f);
        Frac(values, o + 4, static_cast<float>(npc->hull), 8.0f);
        Frac(values, o + 5, static_cast<float>(npc->initiative), 8.0f);
        Frac(values, o + 6, static_cast<float>(npc->cannon), 5.0f);
        Frac(values, o + 7, static_cast<float>(npc->missile), 5.0f);
      }
      o += 8;
    }

    Frac(values, o++, static_cast<float>(__builtin_popcount(state.sector_bag_inner)), 10.0f);
    Frac(values, o++, static_cast<float>(__builtin_popcount(state.sector_bag_middle)), 16.0f);
    Frac(values, o++, static_cast<float>(__builtin_popcount(state.sector_bag_outer)), 22.0f);
    // Bag composition is deducible from public play (ring tile lists and every
    // placement are public), so the masks themselves are fair to expose.
    Bits(values, o, state.sector_bag_inner, 10);   o += 10;
    Bits(values, o, state.sector_bag_middle, 16);  o += 16;
    Bits(values, o, state.sector_bag_outer, 22);   o += 22;

    // Bag SIZES only — never contents.
    Frac(values, o++, static_cast<float>(state.tech_bag.size()), 130.0f);
    Frac(values, o++, static_cast<float>(state.discovery_bag.size()), 40.0f);
    Frac(values, o++, static_cast<float>(state.reputation_tiles.size()), 40.0f);
    int rep_remaining[4] = {0, 0, 0, 0};
    for (const ReputationTiles t : state.reputation_tiles) {
      const int idx = static_cast<int>(t);
      if (idx >= 0 && idx < 4) ++rep_remaining[idx];
    }
    for (int i = 0; i < 4; ++i) {
      Frac(values, o++, static_cast<float>(rep_remaining[i]),
           static_cast<float>(REPUTATION_TILE_COUNTS[i]));
    }

    for (const uint8_t ms : state.minor_species_pool) {
      if (ms < kMinorSpeciesCount) values[o + ms] = 1.0f;
    }
    o += kMinorSpeciesCount;
    OneHot(values, o,
           RelSeat(state.minor_species_pending_track, player, num_players),
           kRelSeatWidth);
    o += kRelSeatWidth;
    Frac(values, o++, static_cast<float>(state.next_arrival_order), 400.0f);
    o += 8;  // reserve
    SPIEL_CHECK_EQ(o, kPlayersStart);
  }

  // ── Block B: one identical block per seat, viewer first ─────────────────
  for (int slot = 0; slot < kSeatSlots; ++slot) {
    int o = PlayerBlockStart(slot);
    if (slot >= num_players) continue;  // leave zero-padded, occupied = 0
    const int seat = SeatForSlot(slot, player, num_players);
    const ::Player& p = state.players[seat];
    const SpeciesData& sp = SPECIES_TABLE[static_cast<size_t>(p.species_id)];

    Flag(values, o++, true);                    // occupied
    Flag(values, o++, seat == player);          // is the viewer
    Flag(values, o++, p.is_ai);
    Flag(values, o++, p.eliminated);
    Flag(values, o++, p.has_passed);
    OneHot(values, o, static_cast<int>(p.species_id), kSpeciesCount);
    o += kSpeciesCount;
    Frac(values, o++, static_cast<float>(sp.activations.explore), 4.0f);
    Frac(values, o++, static_cast<float>(sp.activations.research), 4.0f);
    Frac(values, o++, static_cast<float>(sp.activations.upgrade), 4.0f);
    Frac(values, o++, static_cast<float>(sp.activations.build), 4.0f);
    Frac(values, o++, static_cast<float>(sp.activations.move), 4.0f);
    Frac(values, o++, static_cast<float>(sp.activations.influence), 4.0f);

    int turn_pos = -1;
    for (int i = 0; i < num_players; ++i) {
      if (state.turn_order[i] == seat) { turn_pos = i; break; }
    }
    OneHot(values, o, turn_pos, kMaxSeats);
    o += kMaxSeats;
    Flag(values, o++, turn_pos >= 0);
    int pass_pos = 0;  // 0 = has not passed
    for (size_t i = 0; i < state.pass_order.size(); ++i) {
      if (state.pass_order[i] == seat) { pass_pos = static_cast<int>(i) + 1; break; }
    }
    OneHot(values, o, pass_pos, kMaxSeats + 1);
    o += kMaxSeats + 1;

    const PlayerScoreBreakdown& sb = scores[seat];
    Frac(values, o++, static_cast<float>(sb.total_vp), 60.0f);
    Frac(values, o++, static_cast<float>(sb.reputation_vp), 30.0f);
    Frac(values, o++, static_cast<float>(sb.ambassador_vp), 10.0f);
    Frac(values, o++, static_cast<float>(sb.sector_vp), 30.0f);
    Frac(values, o++, static_cast<float>(sb.monolith_vp), 20.0f);
    Frac(values, o++, static_cast<float>(sb.discovery_vp), 20.0f);
    Frac(values, o++, static_cast<float>(sb.tech_track_vp), 20.0f);
    Frac(values, o++, static_cast<float>(sb.traitor_vp), 4.0f);
    Frac(values, o++, static_cast<float>(sb.species_vp), 20.0f);
    Frac(values, o++, static_cast<float>(sb.minor_species_vp), 20.0f);
    Frac(values, o++, static_cast<float>(std::max<int16_t>(p.vp_at_elimination, 0)), 60.0f);
    Flag(values, o++, p.vp_at_elimination >= 0);

    Frac(values, o++, static_cast<float>(p.resources.gold), 40.0f);
    Frac(values, o++, static_cast<float>(p.resources.science), 40.0f);
    Frac(values, o++, static_cast<float>(p.resources.materials), 40.0f);
    // The raw track index (cubes remaining, 12 = empty of population) ...
    Frac(values, o++, static_cast<float>(p.resources.gold_prod), 12.0f);
    Frac(values, o++, static_cast<float>(p.resources.science_prod), 12.0f);
    Frac(values, o++, static_cast<float>(p.resources.materials_prod), 12.0f);
    // ... and the ACTUAL production it maps to, which runs the other way and
    // is a 13-entry nonlinear lookup the network should not have to learn.
    Frac(values, o++, static_cast<float>(PlayerIncome(p)), 28.0f);
    Frac(values, o++, static_cast<float>(PlayerScienceProduction(p)), 28.0f);
    Frac(values, o++, static_cast<float>(PlayerMaterialsProduction(p)), 28.0f);
    // Solvency, every step — not just while the upkeep sub-state is running.
    const int income = PlayerIncome(p);
    const int upkeep = PlayerUpkeepCost(p);
    Frac(values, o++, static_cast<float>(income), 28.0f);
    Frac(values, o++, static_cast<float>(upkeep), 30.0f);
    Frac(values, o++, static_cast<float>(static_cast<int>(p.resources.gold) + income - upkeep), 30.0f);
    Flag(values, o++, IsPlayerSolvent(p));

    Frac(values, o++, static_cast<float>(p.disks_on_sectors), 16.0f);
    Frac(values, o++, static_cast<float>(p.disks_on_actions), 12.0f);
    Frac(values, o++, static_cast<float>(p.disks_on_reactions), 12.0f);
    Frac(values, o++, static_cast<float>(p.available_influence_discs()), 12.0f);
    Frac(values, o++, static_cast<float>(p.extra_influence_discs), 2.0f);
    Frac(values, o++, static_cast<float>(p.colony_ships_total), 12.0f);
    Frac(values, o++, static_cast<float>(p.colony_ships_available), 12.0f);
    Frac(values, o++, static_cast<float>(p.orbitals), 10.0f);
    Frac(values, o++, static_cast<float>(p.monoliths), 6.0f);
    Frac(values, o++, static_cast<float>(p.trade_rate), 4.0f);
    for (int g = 0; g < 3; ++g) {
      Frac(values, o++, static_cast<float>(p.graveyard_counts[g]), 12.0f);
    }

    SPIEL_CHECK_EQ(o, PlayerBlockStart(slot) + kPlayerTechBitsOffset);
    // All 40 tech bits (bit i of TechBit lives at index i-1).
    const uint64_t all_techs = p.researched_techs_military |
                               p.researched_techs_grid | p.researched_techs_nano;
    Bits(values, o, all_techs >> 1, kTechBitCount);
    o += kTechBitCount;
    const TechCategory tracks[kTechTrackCount] = {
        TechCategory::MILITARY, TechCategory::GRID, TechCategory::NANO};
    for (int t = 0; t < kTechTrackCount; ++t) {
      const int count = get_track_tile_count(p, tracks[t]);
      Frac(values, o + t, static_cast<float>(count),
           static_cast<float>(kTechTrackCapacity));
      Frac(values, o + kTechTrackCount + t,
           static_cast<float>(std::max(0, kTechTrackCapacity - count)),
           static_cast<float>(kTechTrackCapacity));
    }
    o += 2 * kTechTrackCount;

    SPIEL_CHECK_EQ(o, PlayerBlockStart(slot) + kPlayerBlueprintsOffset);
    for (int s = 0; s < kPlayerShipTypes; ++s) {
      o += WriteBlueprint(values, o, p.blueprints[s]);
    }
    SPIEL_CHECK_EQ(o, PlayerBlockStart(slot) + kPlayerPartsInvOffset);
    const int inv_base = o;
    for (const ShipPartId part : p.parts_inventory) {
      const int idx = static_cast<int>(part);
      if (idx > 0 && idx <= kShipPartCount) values[inv_base + idx - 1] += 1.0f / 4.0f;
    }
    o += kShipPartCount;
    Frac(values, o++, static_cast<float>(p.parts_inventory.size()), 24.0f);

    SPIEL_CHECK_EQ(o, PlayerBlockStart(slot) + kPlayerRepTrackOffset);
    for (int s = 0; s < kRepSlots; ++s) {
      const int e = o + s * kRepSlotSize;
      if (s >= static_cast<int>(p.reputation_track.size())) continue;
      const ReputationSlot& slot_v = p.reputation_track[s];
      int so = e;
      OneHot(values, so, static_cast<int>(slot_v.kind), kRepSlotKindCount);
      so += kRepSlotKindCount;
      Flag(values, so++, slot_v.holds_ambassador);
      // WHO the relation is with — the field that made diplomacy unreadable.
      OneHot(values, so,
             slot_v.holds_ambassador
                 ? RelSeat(slot_v.ambassador_from, player, num_players)
                 : kRelNone,
             kRelSeatWidth);
      so += kRelSeatWidth;
      OneHot(values, so, static_cast<int>(slot_v.rep_value), kRepTileValueCount);
      so += kRepTileValueCount;
      Flag(values, so++, slot_v.pending_track_choice);
      SPIEL_CHECK_EQ(so, e + kRepSlotSize);
    }
    o += kRepSlots * kRepSlotSize;

    Frac(values, o++, static_cast<float>(p.ambassador_tiles_held), 5.0f);
    Frac(values, o++, static_cast<float>(p.ambassador_tiles_pending_return), 5.0f);
    Flag(values, o++, p.traitor_held);
    Frac(values, o++, static_cast<float>(p.discovery_vp_tiles_kept), 20.0f);
    for (const uint8_t ms : p.owned_minor_species) {
      if (ms < kMinorSpeciesCount) values[o + ms] = 1.0f;
    }
    o += kMinorSpeciesCount;
    Flag(values, o++, p.warp_portal_eligible);
    Frac(values, o++, static_cast<float>(p.pending_artifact_key_chunks), 6.0f);
    for (int b = 0; b < kBuildTypeCount; ++b) {
      Frac(values, o++,
           static_cast<float>(calculate_build_cost(p, static_cast<BuildType>(b))),
           12.0f);
    }
    o += 4;  // reserve
    SPIEL_CHECK_EQ(o, PlayerBlockStart(slot) + kPlayerSize);
  }

  // ── Block C: galaxy ─────────────────────────────────────────────────────
  {
    // One bucketing pass over the registry instead of the old
    // O(225 x |registry|) rescan-per-cell.
    struct CellUnits {
      float mine[kPlayerShipTypes] = {0};
      float enemy[kPlayerShipTypes] = {0};
      float npc[kNpcShipTypes] = {0};
      float damage = 0.0f;
      bool my_ship = false;
    };
    std::array<CellUnits, kGalaxyCells> cells{};
    for (const Unit& u : state.unit_registry) {
      const HexCoord hc = state.galaxy.FindSectorCoord(u.sector_id);
      if (hc.q == -128) continue;
      if (!in_galaxy_bounds(hc.q, hc.r)) continue;
      CellUnits& cu = cells[hex_to_index(hc.q, hc.r)];
      cu.damage += static_cast<float>(u.damage);
      const int t = static_cast<int>(u.type);
      if (u.player_id == NPC_PLAYER_ID) {
        const int npc_idx = t - kPlayerShipTypes;
        if (npc_idx >= 0 && npc_idx < kNpcShipTypes) cu.npc[npc_idx] += 1.0f;
      } else if (t >= 0 && t < kPlayerShipTypes) {
        if (u.player_id == static_cast<uint8_t>(player)) {
          cu.mine[t] += 1.0f;
          cu.my_ship = true;
        } else if (u.player_id < static_cast<uint8_t>(num_players)) {
          cu.enemy[t] += 1.0f;
        }
      }
    }

    const CombatState& cs = state.combat_state;
    for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
      for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
        if (!in_galaxy_bounds(q, r)) continue;
        const int cell = hex_to_index(q, r);
        const int b = CellStart(cell);
        const Sector& sec = state.galaxy.at(q, r);
        const CellUnits& cu = cells[cell];

        if (state.warped_universe) {
          const int lk = static_cast<int>(state.layout_kinds[cell]);
          for (int d = 0; d < kHexDirections; ++d) {
            Flag(values, b + kCellWarpLink + d,
                 state.warp_link_dest_cell[cell * 6 + d] != 255);
          }
          (void)lk;
        }
        // Units are reported even on an unplaced cell (they cannot be there,
        // but the write is harmless and keeps the loop branch-free).
        for (int t = 0; t < kPlayerShipTypes; ++t) {
          Frac(values, b + kCellMyShips + t, cu.mine[t], 8.0f);
          Frac(values, b + kCellEnemyShips + t, cu.enemy[t], 8.0f);
        }
        for (int t = 0; t < kNpcShipTypes; ++t) {
          Frac(values, b + kCellNpcShips + t, cu.npc[t], 8.0f);
        }
        Frac(values, b + kCellDamage, cu.damage, 16.0f);

        if (sec.sector_id == 0) continue;

        Flag(values, b + kCellPresent, true);
        OneHot(values, b + kCellOwner,
               RelSectorOwner(sec.owner_id, player, num_players), kRelSeatWidth);
        Frac(values, b + kCellPoints, static_cast<float>(sec.points), 4.0f);

        const SectorDefinition* def = get_sector_definition(sec.sector_id);
        if (def != nullptr) {
          Bits(values, b + kCellWormhole,
               rotate_edge_mask(def->wormholes_mask, sec.rotation),
               kHexDirections);
          const int nslots = static_cast<int>(def->slots.size());
          for (int i = 0; i < nslots; ++i) {
            const int t = static_cast<int>(def->slots[i].type);
            if (t < 0 || t >= kPlanetTypeCount) continue;
            values[b + kCellPlanetPrinted + t] += 1.0f / 6.0f;
            if ((sec.occupied_slots_mask >> i) & 1u) {
              values[b + kCellPlanetPopulated + t] += 1.0f / 6.0f;
            }
          }
          Flag(values, b + kCellHasWarpPortal, def->has_warp_portal);
          Flag(values, b + kCellHasArtifact, def->has_artifact);
          Flag(values, b + kCellHasGuardian, def->has_guardian);
          Flag(values, b + kCellIsGcds, def->is_gcds);
          OneHot(values, b + kCellRing, RingKind(def->type), kRingKinds);
          // The Orbital adds one virtual Money slot at index slots.size().
          if (sec.orbital_built) {
            Flag(values, b + kCellOrbitalPopSlot,
                 (sec.occupied_slots_mask >> nslots) & 1u);
          }
        }
        Flag(values, b + kCellOrbital, sec.orbital_built);
        Flag(values, b + kCellMonolith, sec.monolith_built);
        // Presence only. sec.discovery_tile is the identity of a FACE-DOWN
        // tile and is deliberately never written.
        Flag(values, b + kCellDiscoveryPresent, sec.discovery_tile_present);
        Frac(values, b + kCellWarpPortalVp,
             static_cast<float>(sec.player_warp_portal_vp), 2.0f);
        Flag(values, b + kCellMyAnchor,
             sec.owner_id == static_cast<uint8_t>(player) || cu.my_ship);

        if (cs.phase != CombatState::Phase::inactive) {
          Flag(values, b + kCellCombatActive, cs.active_sector_id == sec.sector_id);
          for (int i = 0; i < std::min<int>(cs.battle_queue_size, kBattleQueueCap); ++i) {
            if (cs.battle_queue[i].sector_id == sec.sector_id) {
              Flag(values, b + kCellInBattleQueue, true);
              break;
            }
          }
          for (int i = 0; i < std::min<int>(cs.influence_uncontrolled_size,
                                           static_cast<int>(cs.influence_uncontrolled_sectors.size()));
               ++i) {
            if (cs.influence_uncontrolled_sectors[i] == sec.sector_id) {
              Flag(values, b + kCellInfluenceUncontrolled, true);
              break;
            }
          }
        }
        const MoveState& ms = state.move_state;
        if (ms.active_unit_idx != 255 &&
            ms.active_unit_idx < state.unit_registry.size() &&
            state.unit_registry[ms.active_unit_idx].sector_id == sec.sector_id) {
          Flag(values, b + kCellMoveActiveUnit, true);
        }
        const ExploreState& es = state.explore_state;
        if (es.phase != ExplorePhase::inactive && es.zone_q == q && es.zone_r == r) {
          Flag(values, b + kCellExploreZone, true);
        }
      }
    }
  }

  // ── Block D: tech market ────────────────────────────────────────────────
  {
    int o = kTechMarketStart;
    const ::Player& me = state.players[player];
    float cheapest[kTechTrackCount] = {1.0f, 1.0f, 1.0f};
    for (int i = 0; i < kTechTrayCount; ++i) {
      Frac(values, o + i, static_cast<float>(state.tech_tray[i]),
           static_cast<float>(std::max<uint8_t>(TECH_TABLE[i].copies, 1)));
    }
    for (int i = 0; i < kTechTrayCount; ++i) {
      if (state.tech_tray[i] == 0) continue;
      const TechDefinition& td = TECH_TABLE[i];
      TechCategory cat = td.category;
      if (cat == TechCategory::RARE) cat = TechCategory::MILITARY;
      const float cost = static_cast<float>(calculate_research_cost(me, td, cat));
      Frac(values, o + kTechTrayCount + i, cost, 20.0f);
      const int t = static_cast<int>(cat);
      if (t >= 0 && t < kTechTrackCount) {
        cheapest[t] = std::min(cheapest[t], std::min(cost / 20.0f, 1.0f));
      }
    }
    o += 2 * kTechTrayCount;
    for (int t = 0; t < kTechTrackCount; ++t) values[o + t] = cheapest[t];
    o += kTechTrackCount;
    o += 5;  // reserve
    SPIEL_CHECK_EQ(o, kCombatStart);
  }

  // ── Block E: combat, in full ────────────────────────────────────────────
  {
    int o = kCombatStart;
    const CombatState& cs = state.combat_state;
    const bool active = cs.phase != CombatState::Phase::inactive;
    Flag(values, o++, active);
    if (active) OneHot(values, o, static_cast<int>(cs.phase), kCombatPhaseCount);
    o += kCombatPhaseCount;
    Frac(values, o++, static_cast<float>(cs.active_sector_id), 395.0f);
    Frac(values, o++, static_cast<float>(cs.engagement_round), 20.0f);
    OneHot(values, o, RelSeatOrNpc(cs.current_attacker_id, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    OneHot(values, o, RelSeatOrNpc(cs.current_defender_id, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    OneHot(values, o, RelSeat(cs.pending_player, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    Frac(values, o++, static_cast<float>(cs.battle_queue_size), kBattleQueueCap);
    Frac(values, o++, static_cast<float>(cs.current_battle_idx), kBattleQueueCap);
    for (int i = 0; i < kBattleQueueCap; ++i) {
      const int e = o + i * kBattleEntrySize;
      if (i >= cs.battle_queue_size) continue;
      const CombatSectorInfo& bi = cs.battle_queue[i];
      Flag(values, e, true);
      Frac(values, e + 1, static_cast<float>(bi.sector_id), 395.0f);
      Frac(values, e + 2, static_cast<float>(bi.participant_count), kParticipantsCap);
      for (int pi = 0; pi < std::min<int>(bi.participant_count, kParticipantsCap); ++pi) {
        const int rel = RelSeat(bi.participant_ids[pi], player, num_players);
        if (rel < kMaxSeats) values[e + 3 + rel] = 1.0f;
      }
      const int def_idx = bi.defender_idx;
      const int def_seat = def_idx < kParticipantsCap ? bi.participant_ids[def_idx] : 255;
      OneHot(values, e + 3 + kMaxSeats,
             RelSeatOrNpc(def_seat, player, num_players), kRelSeatWidth);
    }
    o += kBattleQueueCap * kBattleEntrySize;

    Frac(values, o++, static_cast<float>(cs.initiative_size), kInitiativeCap);
    Frac(values, o++, static_cast<float>(cs.initiative_idx), kInitiativeCap);
    for (int i = 0; i < kInitiativeCap; ++i) {
      const int e = o + i * kInitiativeEntrySize;
      if (i >= cs.initiative_size) continue;
      const InitiativeGroup& g = cs.initiative_timeline[i];
      int io = e;
      Flag(values, io++, true);
      OneHot(values, io, RelSeatOrNpc(g.player_id, player, num_players), kRelSeatWidth);
      io += kRelSeatWidth;
      OneHot(values, io, static_cast<int>(g.type), kShipTypeCount);
      io += kShipTypeCount;
      Frac(values, io++, static_cast<float>(g.initiative), 8.0f);
      Flag(values, io++, g.is_npc);
      Flag(values, io++, g.destroyed);
      Flag(values, io++, g.retreating);
      Frac(values, io++, static_cast<float>(g.alive_count), 8.0f);
      Frac(values, io++, static_cast<float>(g.destroyed_count), 8.0f);
      SPIEL_CHECK_EQ(io, e + kInitiativeEntrySize);
    }
    o += kInitiativeCap * kInitiativeEntrySize;

    for (int i = 0; i < std::min<int>(cs.destroyed_ships_size,
                                      static_cast<int>(cs.destroyed_ships.size())); ++i) {
      const DestroyedShipRecord& d = cs.destroyed_ships[i];
      const int rel = RelSeatOrNpc(d.player_id, player, num_players);
      const int t = static_cast<int>(d.type);
      if (rel < kRelSeatWidth && t >= 0 && t < kShipTypeCount) {
        values[o + rel * kShipTypeCount + t] += static_cast<float>(d.count) / 8.0f;
      }
    }
    o += kRelSeatWidth * kShipTypeCount;

    for (int i = 0; i < std::min<int>(cs.ship_order_size, kInitiativeCap); ++i) {
      const int t = static_cast<int>(cs.ship_order_queue[i]);
      if (t >= 0 && t < kShipTypeCount) values[o + t] += 1.0f / 4.0f;
    }
    o += kShipTypeCount;
    Frac(values, o++, static_cast<float>(cs.ship_order_idx), kInitiativeCap);
    OneHot(values, o, static_cast<int>(cs.active_ship_type), kShipTypeCount);
    o += kShipTypeCount;

    for (int i = 0; i < std::min<int>(cs.retreat_destinations_size, kRetreatDestCap); ++i) {
      Frac(values, o + i, static_cast<float>(cs.retreat_destinations[i]), 395.0f);
    }
    o += kRetreatDestCap;
    Frac(values, o++, static_cast<float>(cs.retreat_destinations_size), kRetreatDestCap);

    for (int i = 0; i < std::min<int>(cs.drawn_tiles_size, kRepDrawCap); ++i) {
      OneHot(values, o + i * kRepTileValueCount,
             static_cast<int>(cs.drawn_tiles[i]), kRepTileValueCount);
    }
    o += kRepDrawCap * kRepTileValueCount;
    Frac(values, o++, static_cast<float>(cs.drawn_tiles_size), kRepDrawCap);
    OneHot(values, o, RelSeat(cs.tile_select_player, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    OneHot(values, o, RelSeat(cs.rep_draw_target, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    for (int i = 0; i < kParticipantsCap; ++i) {
      Frac(values, o + i, static_cast<float>(cs.reputation_drawn_mask[i]), 4.0f);
    }
    o += kParticipantsCap;
    for (int i = 0; i < kParticipantsCap; ++i) {
      Frac(values, o + i, static_cast<float>(cs.reputation_retreat_penalty_mask[i]), 4.0f);
    }
    o += kParticipantsCap;
    Frac(values, o++, static_cast<float>(cs.reputation_earned), 15.0f);

    // The dice actually rolled and the units they may be assigned to — the
    // agent was previously assigning damage blind.
    OneHot(values, o, RelSeatOrNpc(cs.pending_target_group_player, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    OneHot(values, o, static_cast<int>(cs.pending_target_group_type), kShipTypeCount);
    o += kShipTypeCount;
    for (int i = 0; i < std::min<int>(cs.pending_target_count, kPendingTargetCap); ++i) {
      const int idx = cs.pending_target_indices[i];
      if (idx < kPendingTargetCap) values[o + idx] = 1.0f;
    }
    o += kPendingTargetCap;
    Frac(values, o++, static_cast<float>(cs.pending_target_count), 32.0f);
    const int die_n = std::min<int>(cs.pending_die_count,
                                   static_cast<int>(cs.pending_die_values.size()));
    for (int i = 0; i < die_n; ++i) {
      const int face = static_cast<int>(cs.pending_die_values[i]) - 1;
      if (face >= 0 && face < kDieFaces) values[o + face] += 1.0f / 8.0f;
    }
    o += kDieFaces;
    for (int i = 0; i < die_n; ++i) {
      const int col = static_cast<int>(cs.pending_die_colors[i]);
      if (col >= 0 && col <= kDieColorCount) values[o + col] += 1.0f / 8.0f;
    }
    o += kDieColorCount + 1;
    if (cs.pending_die_index < die_n) {
      OneHot(values, o, static_cast<int>(cs.pending_die_values[cs.pending_die_index]) - 1, kDieFaces);
      OneHot(values, o + kDieFaces, static_cast<int>(cs.pending_die_colors[cs.pending_die_index]),
             kDieColorCount + 1);
    }
    o += kDieFaces + kDieColorCount + 1;
    Frac(values, o++, static_cast<float>(cs.pending_die_count), 64.0f);
    Frac(values, o++, static_cast<float>(cs.pending_die_index), 64.0f);
    Flag(values, o++, cs.pending_dice_are_missiles);
    Flag(values, o++, cs.pending_dice_pop_attack);

    for (int i = 0; i < std::min<int>(cs.retreating_group_count, kRetreatingCap); ++i) {
      const int rel = RelSeatOrNpc(cs.retreating_players[i], player, num_players);
      const int t = static_cast<int>(cs.retreating_types[i]);
      if (rel < kRelSeatWidth && t >= 0 && t < kShipTypeCount) {
        values[o + rel * kShipTypeCount + t] += 1.0f / 4.0f;
      }
    }
    o += kRelSeatWidth * kShipTypeCount;
    Frac(values, o++, static_cast<float>(cs.retreating_group_count), kRetreatingCap);

    Frac(values, o++, static_cast<float>(cs.pop_attack_sector_id), 395.0f);
    OneHot(values, o, RelSeatOrNpc(cs.pop_attack_player, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    OneHot(values, o, RelSeat(cs.pop_attack_owner, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    Frac(values, o++, static_cast<float>(cs.pop_attack_damage_remaining), 20.0f);
    Frac(values, o++, static_cast<float>(cs.pop_attack_unit_index), 128.0f);
    Frac(values, o++, static_cast<float>(cs.influence_uncontrolled_size), 225.0f);
    Frac(values, o++, static_cast<float>(cs.influence_scan_index), 225.0f);
    Frac(values, o++, static_cast<float>(cs.influence_turn_order_index), 6.0f);
    OneHot(values, o, RelSeat(cs.influence_decision_player, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    Frac(values, o++, static_cast<float>(cs.influence_decision_sector), 395.0f);
    Frac(values, o++, static_cast<float>(cs.discovery_decision_sector), 395.0f);
    OneHot(values, o, RelSeat(cs.discovery_decision_player, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    o += 8;  // reserve
    SPIEL_CHECK_EQ(o, kUpkeepStart);
  }

  // ── Block F: upkeep ─────────────────────────────────────────────────────
  {
    int o = kUpkeepStart;
    const UpkeepState& us = state.upkeep_state;
    const bool active = us.step != UpkeepState::Step::inactive;
    Flag(values, o++, active);
    if (active) OneHot(values, o, static_cast<int>(us.step), kUpkeepStepCount);
    o += kUpkeepStepCount;
    OneHot(values, o, RelSeat(us.player_id, player, num_players), kRelSeatWidth);
    o += kRelSeatWidth;
    o += WritePendingReturns(values, o, us.pending_returns);
    o += 4;  // reserve
    SPIEL_CHECK_EQ(o, kActionStatesStart);
  }

  // ── Block G: in-flight action sub-states ────────────────────────────────
  {
    int o = kActionStatesStart;

    // Explore, in full: the drawn and selected tiles are described feature by
    // feature so the rotate / place / discard decision is not made blind.
    {
      const ExploreState& es = state.explore_state;
      const bool active = es.phase != ExplorePhase::inactive;
      const int base = o;
      Flag(values, o++, active);
      if (active) OneHot(values, o, static_cast<int>(es.phase), kExplorePhaseCount);
      o += kExplorePhaseCount;
      OneHot(values, o, RelSeat(es.player_id, player, num_players), kRelSeatWidth);
      o += kRelSeatWidth;
      Frac(values, o++, static_cast<float>(es.activations_remaining), 4.0f);
      Frac(values, o++, static_cast<float>(es.zone_q), static_cast<float>(GALAXY_RADIUS));
      Frac(values, o++, static_cast<float>(es.zone_r), static_cast<float>(GALAXY_RADIUS));
      Flag(values, o++, active && in_galaxy_bounds(es.zone_q, es.zone_r));
      if (active) OneHot(values, o, static_cast<int>(es.ring), kSectorTypeCount);
      o += kSectorTypeCount;
      Frac(values, o++, static_cast<float>(es.drawn_count), 2.0f);
      for (int i = 0; i < 2; ++i) {
        o += WriteTileFeatures(values, o,
                               i < es.drawn_count ? es.drawn_sector_ids[i] : 0);
      }
      o += WriteTileFeatures(values, o, es.selected_sector_id);
      const bool has_rot = active && es.chosen_rotation < kHexDirections;
      if (has_rot) OneHot(values, o, es.chosen_rotation, kHexDirections);
      o += kHexDirections;
      Flag(values, o++, has_rot);
      const SectorDefinition* sel_def =
          es.selected_sector_id == 0 ? nullptr
                                     : get_sector_definition(es.selected_sector_id);
      if (sel_def != nullptr) {
        Bits(values, o, rotate_edge_mask(sel_def->wormholes_mask, es.chosen_rotation),
             kHexDirections);
      }
      o += kHexDirections;
      OneHot(values, o, static_cast<int>(es.discovered_part), kShipPartCount + 1);
      o += kShipPartCount + 1;
      SPIEL_CHECK_EQ(o, base + kExploreSize);
    }

    // Research / Build / Upgrade genuinely have only these three fields --
    // each activation is atomic, nothing else is staged.
    auto simple = [&](int active_flag, int phase_idx, int phase_count,
                      uint8_t pid, uint8_t activations) {
      const int base = o;
      Flag(values, o++, active_flag != 0);
      if (active_flag) OneHot(values, o, phase_idx, phase_count);
      o += phase_count;
      OneHot(values, o, RelSeat(pid, player, num_players), kRelSeatWidth);
      o += kRelSeatWidth;
      Frac(values, o++, static_cast<float>(activations), 4.0f);
      SPIEL_CHECK_EQ(o, base + kSimpleActionSize(phase_count));
    };
    const ResearchState& rs = state.research_state;
    simple(rs.phase != ResearchState::Phase::inactive, static_cast<int>(rs.phase),
           kResearchPhaseCount, rs.player_id, rs.activations_remaining);
    const BuildState& bs = state.build_state;
    simple(bs.phase != BuildState::Phase::inactive, static_cast<int>(bs.phase),
           kBuildPhaseCount, bs.player_id, bs.activations_remaining);
    const UpgradeState& ugs = state.upgrade_state;
    simple(ugs.phase != UpgradeState::Phase::inactive, static_cast<int>(ugs.phase),
           kUpgradePhaseCount, ugs.player_id, ugs.activations_remaining);

    {
      const InfluenceState& is = state.influence_state;
      const int base = o;
      simple(is.phase != InfluenceState::Phase::inactive, static_cast<int>(is.phase),
             kInfluencePhaseCount, is.player_id, is.activations_remaining);
      o += WritePendingReturns(values, o, is.pending_returns);
      SPIEL_CHECK_EQ(o, base + kInfluenceSubSize);
    }

    {
      const MoveState& ms = state.move_state;
      const int base = o;
      simple(ms.phase != MoveState::Phase::inactive, static_cast<int>(ms.phase),
             kMovePhaseCount, ms.player_id, ms.activations_remaining);
      const bool has_active = ms.active_unit_idx != 255 &&
                              ms.active_unit_idx < state.unit_registry.size();
      Flag(values, o++, has_active);
      if (has_active) {
        OneHot(values, o, static_cast<int>(state.unit_registry[ms.active_unit_idx].type),
               kShipTypeCount);
      }
      o += kShipTypeCount;
      Frac(values, o++,
           has_active ? static_cast<float>(state.unit_registry[ms.active_unit_idx].damage) : 0.0f,
           8.0f);
      Frac(values, o++, static_cast<float>(ms.steps_remaining), 6.0f);
      const bool has_warp = ms.warp_unit_idx != 255 &&
                            ms.warp_unit_idx < state.unit_registry.size();
      Flag(values, o++, has_warp);
      if (has_warp) {
        OneHot(values, o, static_cast<int>(state.unit_registry[ms.warp_unit_idx].type),
               kShipTypeCount);
      }
      o += kShipTypeCount;
      SPIEL_CHECK_EQ(o, base + kMoveSubSize);
    }

    // Diplomacy — previously absent from the tensor entirely.
    {
      const DiplomacyState& ds = state.diplomacy_state;
      const int base = o;
      const bool active = ds.phase != DiplomacyState::Phase::inactive;
      Flag(values, o++, active);
      if (active) OneHot(values, o, static_cast<int>(ds.phase), kDiplomacyPhaseCount);
      o += kDiplomacyPhaseCount;
      OneHot(values, o, RelSeat(ds.player_id, player, num_players), kRelSeatWidth);
      o += kRelSeatWidth;
      OneHot(values, o, RelSeat(ds.partner_id, player, num_players), kRelSeatWidth);
      o += kRelSeatWidth;
      OneHot(values, o, ds.rearrange_side < 2 ? ds.rearrange_side : 2, 3);
      o += 3;
      OneHot(values, o, ds.pop_track_side < 2 ? ds.pop_track_side : 2, 3);
      o += 3;
      OneHot(values, o, ds.selected_track < 3 ? ds.selected_track : 2, 3);
      o += 3;
      OneHot(values, o, ds.return_side < 2 ? ds.return_side : 2, 3);
      o += 3;
      SPIEL_CHECK_EQ(o, base + kDiplomacySubSize);
    }

    o += 4;  // reserve
    SPIEL_CHECK_EQ(o, kTotalSize);
  }
}

}  // namespace obs
}  // namespace eclipse
}  // namespace open_spiel
