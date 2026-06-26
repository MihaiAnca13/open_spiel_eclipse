//
// Combat Phase state machine implementation.
//
// Randomness model: combat is fully deterministic given the chance outcomes
// fed to it. Every weapon die is rolled by a PendingRandomEvent::combat_roll
// chance node (one node per die) and every reputation tile is drawn by a
// PendingRandomEvent::reputation_draw chance node. No std::mt19937 lives in
// this file, so cloned MCTS states share no hidden RNG and the search sees the
// true combat variance.
//

#include "combat.h"
#include "../state.h"
#include "../resources.h"
#include "../galaxy.h"
#include "../tech.h"
#include "../species.h"
#include "../dice.h"
#include <algorithm>

namespace open_spiel::eclipse {

// ---- helpers -----------------------------------------------------------

bool IsNPC(uint8_t p) { return p == NPC_PLAYER_ID; }

// Counts alive ships of a (player, type) pair currently in the given sector.
uint8_t CountAliveInSector(const ::State& state, uint8_t player_id,
                           ShipType type, uint16_t sector_id) {
    uint8_t count = 0;
    for (const Unit& u : state.unit_registry) {
        if (u.player_id == player_id && u.type == type &&
            u.sector_id == sector_id) {
            ++count;
        }
    }
    return count;
}

uint8_t CountAliveInSectorAnyType(const ::State& state, uint8_t player_id,
                                  uint16_t sector_id) {
    uint8_t count = 0;
    for (const Unit& u : state.unit_registry) {
        if (u.player_id == player_id && u.sector_id == sector_id) {
            ++count;
        }
    }
    return count;
}

// Build a unique list of player_ids that have at least one ship in the given
// sector. NPC ships are always included.
std::vector<uint8_t> CollectSectorParticipants(const ::State& state,
                                               uint16_t sector_id) {
    std::vector<uint8_t> result;
    for (const Unit& u : state.unit_registry) {
        if (u.sector_id == sector_id) {
            if (std::find(result.begin(), result.end(), u.player_id) ==
                result.end()) {
                result.push_back(u.player_id);
            }
        }
    }
    return result;
}

namespace {
// Resolve an NPC stat row for a (type, difficulty) lookup. Returns nullptr if
// no row matches.
const NPC* NpcRow(const ::State& state, ShipType type) {
    NPCType nt = (type == ShipType::GCDS)       ? NPCType::GCDS
               : (type == ShipType::GUARDIAN)   ? NPCType::GUARDIAN
                                                : NPCType::ANCIENT;
    NPCDifficulty diff = (nt == NPCType::GCDS)       ? state.gcds_difficulty
                       : (nt == NPCType::GUARDIAN)   ? state.guardian_difficulty
                                                     : state.ancient_difficulty;
    for (const NPC& row : NPC_TABLE) {
        if (row.type == nt && row.difficulty == diff) return &row;
    }
    return nullptr;
}
}  // namespace

int ComputerForPlayer(const ::State& state, uint8_t player_id, ShipType type) {
    if (player_id == NPC_PLAYER_ID) {
        const NPC* row = NpcRow(state, type);
        return row ? row->computer : 0;
    }
    return state.players[player_id]
        .blueprints[static_cast<size_t>(type)]
        .total_stats.computer;
}

int ShieldForPlayer(const ::State& state, uint8_t player_id, ShipType type) {
    if (player_id == NPC_PLAYER_ID) {
        const NPC* row = NpcRow(state, type);
        return row ? row->shield : 0;
    }
    return state.players[player_id]
        .blueprints[static_cast<size_t>(type)]
        .total_stats.shield;
}

int HullForPlayer(const ::State& state, uint8_t player_id, ShipType type) {
    if (player_id == NPC_PLAYER_ID) {
        const NPC* row = NpcRow(state, type);
        return row ? row->hull : 1;
    }
    return state.players[player_id]
        .blueprints[static_cast<size_t>(type)]
        .total_stats.hull;
}

int InitiativeForPlayer(const ::State& state, uint8_t player_id, ShipType type) {
    if (player_id == NPC_PLAYER_ID) {
        const NPC* row = NpcRow(state, type);
        return row ? row->initiative : 0;
    }
    return state.players[player_id]
        .blueprints[static_cast<size_t>(type)]
        .total_stats.initiative;
}

bool IsCurrentPairParticipant(const CombatState& cs, uint8_t player_id) {
    if (cs.current_attacker_id == kNoPlayer &&
        cs.current_defender_id == kNoPlayer) {
        return true;
    }
    return player_id == cs.current_attacker_id ||
           player_id == cs.current_defender_id;
}

bool IsEnemyInCurrentPair(const CombatState& cs, uint8_t attacker_id,
                          uint8_t target_id) {
    if (target_id == attacker_id) return false;
    if (cs.current_attacker_id == kNoPlayer &&
        cs.current_defender_id == kNoPlayer) {
        return true;
    }
    return IsCurrentPairParticipant(cs, attacker_id) &&
           IsCurrentPairParticipant(cs, target_id);
}

int DamageForDie(uint8_t roll, DieColor color) {
    if (color == DieColor::PURPLE) {
        return RIFT_CANNON_FACES[roll - 1].damage;
    }
    const int color_idx = static_cast<int>(color);
    if (color_idx >= 0 && color_idx < 4) return DIE_DAMAGE[color_idx];
    return 0;
}

int SelfDamageForDie(uint8_t roll, DieColor color) {
    if (color == DieColor::PURPLE) {
        return RIFT_CANNON_FACES[roll - 1].self_damage;
    }
    return 0;
}

// Whether a die hits a target with the given shield value.
//
// The Rift Cannon (purple) uses its own die faces: a face hits iff it shows
// target damage, completely ignoring Computers and Shields and the usual
// 6-always-hits / 1-always-misses rules. Every other die follows the standard
// rule: blank (1) always misses, burst (6) always hits, otherwise
// roll + computers - shields >= 6.
bool DieHitsShield(const ::State& state, uint8_t attacker_id,
                   ShipType attacker_type, uint8_t roll, DieColor color,
                   int shield) {
    if (color == DieColor::NONE) return false;
    if (color == DieColor::PURPLE) {
        return RIFT_CANNON_FACES[roll - 1].damage > 0;
    }
    if (roll == 1) return false;
    if (roll == 6) return true;
    const int computer = ComputerForPlayer(state, attacker_id, attacker_type);
    return roll + computer - shield >= 6;
}

bool DieHitsTarget(const ::State& state, uint8_t attacker_id,
                   ShipType attacker_type, uint8_t roll, DieColor color,
                   const Unit& target) {
    const int shield = (color == DieColor::PURPLE)
                           ? 0
                           : ShieldForPlayer(state, target.player_id,
                                             target.type);
    return DieHitsShield(state, attacker_id, attacker_type, roll, color, shield);
}

void RecordDestroyedShip(::State& state, uint8_t owner, ShipType type,
                         uint8_t destroyed_by) {
    DestroyedShipRecord rec{owner, type, 1, destroyed_by};
    CombatState& cs = state.combat_state;
    if (cs.destroyed_ships_size < kMaxDestroyedPerBattle) {
        cs.destroyed_ships[cs.destroyed_ships_size++] = rec;
    }
}

void ApplyDamageToUnit(::State& state, Unit& unit, int damage,
                       uint8_t destroyed_by) {
    if (damage <= 0) return;
    unit.damage = static_cast<uint8_t>(unit.damage + damage);
    if (unit.damage > HullForPlayer(state, unit.player_id, unit.type)) {
        RecordDestroyedShip(state, unit.player_id, unit.type, destroyed_by);
        unit.sector_id = kGraveyardSectorId;
    }
}

// Apply the Rift Cannon's self-damage to one of the firing group's own ships in
// the active sector. Called once per rolled die, independent of targeting.
void ApplyRiftSelfDamage(::State& state, uint8_t attacker_id,
                         ShipType attacker_type, int self_damage) {
    if (self_damage <= 0) return;
    const uint16_t sector = state.combat_state.active_sector_id;
    for (size_t i = 0; i < state.unit_registry.size(); ++i) {
        Unit& unit = state.unit_registry[i];
        if (unit.player_id == attacker_id && unit.type == attacker_type &&
            unit.sector_id == sector) {
            ApplyDamageToUnit(state, unit, self_damage, attacker_id);
            return;
        }
    }
}

bool IsLegalDieTarget(const ::State& state, uint8_t attacker_id,
                      size_t target_idx) {
    if (target_idx >= state.unit_registry.size()) return false;
    const Unit& target = state.unit_registry[target_idx];
    const CombatState& cs = state.combat_state;
    return target.sector_id == cs.active_sector_id &&
           IsEnemyInCurrentPair(cs, attacker_id, target.player_id);
}

uint8_t CountLegalTargets(const ::State& state, uint8_t attacker_id) {
    uint8_t n = 0;
    for (size_t i = 0; i < state.unit_registry.size(); ++i) {
        if (IsLegalDieTarget(state, attacker_id, i)) ++n;
    }
    return n;
}

uint8_t CountHittableTargets(const ::State& state, uint8_t attacker_id,
                             ShipType type, uint8_t roll, DieColor color) {
    uint8_t n = 0;
    for (size_t i = 0; i < state.unit_registry.size(); ++i) {
        if (!IsLegalDieTarget(state, attacker_id, i)) continue;
        if (DieHitsTarget(state, attacker_id, type, roll, color,
                          state.unit_registry[i])) {
            ++n;
        }
    }
    return n;
}

// Pick the best legal target the die would hit: prefer one this die can
// destroy, then larger ship types, then higher damage. This implements the
// non-player damage-assignment rule ("destroy your largest ships first; else
// inflict as much damage as possible, largest first") and is also used to
// auto-resolve player dice with no meaningful choice.
size_t ChooseBestTarget(const ::State& state, uint8_t attacker_id,
                        ShipType attacker_type, uint8_t roll, DieColor color) {
    size_t best_idx = SIZE_MAX;
    int best_score = -1;
    const int damage = DamageForDie(roll, color);
    for (size_t i = 0; i < state.unit_registry.size(); ++i) {
        if (!IsLegalDieTarget(state, attacker_id, i)) continue;
        const Unit& target = state.unit_registry[i];
        if (!DieHitsTarget(state, attacker_id, attacker_type, roll, color,
                           target)) {
            continue;
        }
        const int remaining_hull =
            HullForPlayer(state, target.player_id, target.type) - target.damage;
        const int score = (damage > remaining_hull ? 1000 : 0) +
                          static_cast<int>(target.type) * 10 + damage;
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    return best_idx;
}

// Apply a hitting die's damage to `target_idx`. Does NOT apply self-damage
// (the caller applies that once at roll time). Returns true if the die hit.
//
// Antimatter Splitter: a hitting red (antimatter) die may divide its damage
// among several ships. We model this as a greedy split — destroy the chosen
// primary, then carry leftover damage to the next-best legal hittable target,
// largest first. Manual point-by-point distribution is not exposed as separate
// actions; this keeps the tech mechanically active without inflating the tree.
bool ApplyHitDamage(::State& state, uint8_t attacker_id, ShipType attacker_type,
                    uint8_t roll, DieColor color, size_t target_idx) {
    if (!IsLegalDieTarget(state, attacker_id, target_idx)) return false;
    Unit& target = state.unit_registry[target_idx];
    if (!DieHitsTarget(state, attacker_id, attacker_type, roll, color, target)) {
        return false;
    }
    const int damage = DamageForDie(roll, color);
    if (damage <= 0) return false;

    const bool splitter =
        attacker_id != NPC_PLAYER_ID && attacker_id < state.players.size() &&
        color == kAntimatterDie &&
        state.players[attacker_id].has_tech(TechBit::ANTIMATTER_SPLITTER);
    if (!splitter) {
        ApplyDamageToUnit(state, target, damage, attacker_id);
        return true;
    }

    int pool = damage;
    int need = HullForPlayer(state, target.player_id, target.type) -
               target.damage + 1;
    if (need < 1) need = 1;
    int give = std::min(pool, need);
    ApplyDamageToUnit(state, target, give, attacker_id);
    pool -= give;
    while (pool > 0) {
        const size_t next =
            ChooseBestTarget(state, attacker_id, attacker_type, roll, color);
        if (next == SIZE_MAX) break;
        Unit& t2 = state.unit_registry[next];
        int need2 = HullForPlayer(state, t2.player_id, t2.type) - t2.damage + 1;
        if (need2 < 1) need2 = 1;
        const int give2 = std::min(pool, need2);
        ApplyDamageToUnit(state, t2, give2, attacker_id);
        pool -= give2;
        if (give2 < need2) break;  // can't destroy this one; rest is wasted
    }
    return true;
}

// Drain destroyed units out of the registry at the end of a battle.
void FlushDestroyedShips(::State& state) {
    FixedVector<Unit, 128> kept;
    for (const Unit& u : state.unit_registry) {
        if (u.sector_id == kGraveyardSectorId) continue;
        kept.push_back(u);
    }
    state.unit_registry = kept;
    state.combat_state.destroyed_ships_size = 0;
}

// Build the initiative timeline for the current battle. Excludes any group
// with zero alive ships in the active sector.
void RebuildInitiativeTimeline(::State& state) {
    CombatState& cs = state.combat_state;
    cs.initiative_size = 0;
    cs.initiative_idx = 0;
    const uint16_t sector = cs.active_sector_id;
    const uint8_t defender = cs.current_defender_id;

    std::array<std::pair<uint8_t, ShipType>, 32> seen;
    uint8_t seen_size = 0;
    for (const Unit& u : state.unit_registry) {
        if (u.sector_id != sector) continue;
        if (!IsCurrentPairParticipant(cs, u.player_id)) continue;
        bool found = false;
        for (uint8_t k = 0; k < seen_size; ++k) {
            if (seen[k].first == u.player_id && seen[k].second == u.type) {
                found = true;
                break;
            }
        }
        if (found) continue;
        if (seen_size < seen.size()) {
            seen[seen_size++] = {u.player_id, u.type};
        }
    }
    for (uint8_t k = 0; k < seen_size; ++k) {
        InitiativeGroup g{};
        g.player_id = seen[k].first;
        g.type = seen[k].second;
        g.is_npc = (g.player_id == NPC_PLAYER_ID);
        g.initiative = static_cast<int8_t>(std::min<int>(
            127, InitiativeForPlayer(state, g.player_id, g.type)));
        g.alive_count = CountAliveInSector(state, g.player_id, g.type, sector);
        g.destroyed_count = 0;
        g.destroyed = (g.alive_count == 0);
        g.retreating = false;
        if (cs.initiative_size < kMaxInitiativeGroups) {
            cs.initiative_timeline[cs.initiative_size++] = g;
        }
    }
    // Sort: higher initiative first; defender wins ties.
    std::sort(cs.initiative_timeline.begin(),
              cs.initiative_timeline.begin() + cs.initiative_size,
              [&](const InitiativeGroup& a, const InitiativeGroup& b) {
                  if (a.initiative != b.initiative) {
                      return a.initiative > b.initiative;
                  }
                  if (a.player_id == defender) return true;
                  if (b.player_id == defender) return false;
                  return a.player_id < b.player_id;
              });
}

bool CompleteRetreatIfReady(::State& state, uint8_t player_id, ShipType type);
bool StartNextPairInActiveSector(::State& state);

// Move the initiative cursor past any group with zero alive ships. Returns
// true when a non-exhausted group is found at the cursor.
bool AdvanceToNextAliveGroup(::State& state) {
    CombatState& cs = state.combat_state;
    while (cs.initiative_idx < cs.initiative_size) {
        InitiativeGroup& g = cs.initiative_timeline[cs.initiative_idx];
        g.alive_count = CountAliveInSector(state, g.player_id, g.type,
                                           cs.active_sector_id);
        if (g.alive_count > 0 &&
            CompleteRetreatIfReady(state, g.player_id, g.type)) {
            ++cs.initiative_idx;
            continue;
        }
        if (g.alive_count > 0) {
            cs.active_ship_type = g.type;
            return true;
        }
        ++cs.initiative_idx;
    }
    return false;
}

uint8_t CountParticipantsWithShips(const ::State& state, uint16_t sector_id) {
    std::array<uint8_t, kMaxParticipantsPerBattle> participants{};
    uint8_t size = 0;
    for (const Unit& u : state.unit_registry) {
        if (u.sector_id != sector_id) continue;
        bool seen = false;
        for (uint8_t i = 0; i < size; ++i) {
            if (participants[i] == u.player_id) {
                seen = true;
                break;
            }
        }
        if (!seen && size < participants.size()) {
            participants[size++] = u.player_id;
        }
    }
    return size;
}

uint8_t CountCurrentPairParticipantsWithShips(const ::State& state) {
    const CombatState& cs = state.combat_state;
    uint8_t count = 0;
    if (cs.current_attacker_id != kNoPlayer &&
        CountAliveInSectorAnyType(state, cs.current_attacker_id,
                                  cs.active_sector_id) > 0) {
        ++count;
    }
    if (cs.current_defender_id != kNoPlayer &&
        CountAliveInSectorAnyType(state, cs.current_defender_id,
                                  cs.active_sector_id) > 0) {
        ++count;
    }
    return count;
}

uint8_t FindParticipantIndex(const CombatSectorInfo& battle, uint8_t player_id) {
    for (uint8_t i = 0; i < battle.participant_count; ++i) {
        if (battle.participant_ids[i] == player_id) return i;
    }
    return kNoPlayer;
}

bool IsRetreatingGroup(const CombatState& cs, uint8_t player_id, ShipType type,
                       uint8_t* index = nullptr) {
    for (uint8_t i = 0; i < cs.retreating_group_count; ++i) {
        if (cs.retreating_players[i] == player_id &&
            cs.retreating_types[i] == type) {
            if (index) *index = i;
            return true;
        }
    }
    return false;
}

bool AllAliveShipsAreRetreating(const ::State& state, uint8_t player_id) {
    const CombatState& cs = state.combat_state;
    bool has_ship = false;
    for (const Unit& u : state.unit_registry) {
        if (u.player_id != player_id || u.sector_id != cs.active_sector_id) {
            continue;
        }
        has_ship = true;
        if (!IsRetreatingGroup(cs, player_id, u.type)) return false;
    }
    return has_ship;
}

void AddRetreatingGroup(::State& state, uint8_t player_id, ShipType type,
                        uint16_t destination_sector_id) {
    CombatState& cs = state.combat_state;
    if (!IsRetreatingGroup(cs, player_id, type) &&
        cs.retreating_group_count < kMaxRetreatingGroups) {
        const uint8_t i = cs.retreating_group_count++;
        cs.retreating_players[i] = player_id;
        cs.retreating_types[i] = type;
        cs.retreating_destinations[i] = destination_sector_id;
        cs.retreating_rounds[i] = cs.engagement_round;
    }
    // Retreat penalty: if all of a player's remaining ships are retreating they
    // forfeit the participation reputation tile.
    if (AllAliveShipsAreRetreating(state, player_id)) {
        const auto& battle = cs.battle_queue[cs.current_battle_idx];
        const uint8_t participant_idx = FindParticipantIndex(battle, player_id);
        if (participant_idx < kMaxParticipantsPerBattle) {
            cs.reputation_retreat_penalty_mask[participant_idx] = 1;
        }
    }
}

bool CompleteRetreatIfReady(::State& state, uint8_t player_id, ShipType type) {
    CombatState& cs = state.combat_state;
    uint8_t idx = 0;
    if (!IsRetreatingGroup(cs, player_id, type, &idx)) return false;
    if (cs.engagement_round <= cs.retreating_rounds[idx]) return false;
    const uint16_t destination = cs.retreating_destinations[idx];
    for (size_t i = 0; i < state.unit_registry.size(); ++i) {
        Unit& u = state.unit_registry[i];
        if (u.player_id == player_id && u.type == type &&
            u.sector_id == cs.active_sector_id) {
            u.sector_id = destination;
            u.arrival_order = state.AllocateArrivalOrder();
        }
    }
    for (uint8_t i = idx + 1; i < cs.retreating_group_count; ++i) {
        cs.retreating_players[i - 1] = cs.retreating_players[i];
        cs.retreating_types[i - 1] = cs.retreating_types[i];
        cs.retreating_destinations[i - 1] = cs.retreating_destinations[i];
        cs.retreating_rounds[i - 1] = cs.retreating_rounds[i];
    }
    --cs.retreating_group_count;
    return true;
}

// Transient sector_id -> coordinate index. Built on demand (a single galaxy
// scan) and lives only on the stack, so it never bloats State / Clone. Sector
// ids are < 395 (see sectors.h).
constexpr int kSectorIdLimit = 400;
struct SectorCoordIndex {
    std::array<int16_t, kSectorIdLimit> q;
    std::array<int16_t, kSectorIdLimit> r;
    void Build(const ::State& state) {
        q.fill(INT16_MIN);
        for (int qq = -GALAXY_RADIUS; qq <= GALAXY_RADIUS; ++qq) {
            for (int rr = -GALAXY_RADIUS; rr <= GALAXY_RADIUS; ++rr) {
                if (!in_galaxy_bounds(qq, rr)) continue;
                const uint16_t sid = state.galaxy.at(qq, rr).sector_id;
                if (sid != 0 && sid < kSectorIdLimit) {
                    q[sid] = static_cast<int16_t>(qq);
                    r[sid] = static_cast<int16_t>(rr);
                }
            }
        }
    }
    Sector* Get(::State& state, uint16_t sector_id) const {
        if (sector_id >= kSectorIdLimit || q[sector_id] == INT16_MIN) {
            return nullptr;
        }
        return &state.galaxy.at(q[sector_id], r[sector_id]);
    }
};

bool StartNextPairInActiveSector(::State& state) {
    CombatState& cs = state.combat_state;
    const auto participants =
        CollectSectorParticipants(state, cs.active_sector_id);
    if (participants.size() < 2) return false;
    auto present = [&](uint8_t p) {
        return std::find(participants.begin(), participants.end(), p) !=
               participants.end();
    };

    // Prefer the battle's precomputed defender (NPC / sector controller) when
    // it is still present; this reuses identify_combat_sectors' decision
    // instead of re-deriving it from the galaxy.
    uint8_t defender = kNoPlayer;
    const auto& battle = cs.battle_queue[cs.current_battle_idx];
    if (battle.defender_idx < battle.participant_count) {
        const uint8_t pref = battle.participant_ids[battle.defender_idx];
        if (present(pref)) defender = pref;
    }
    if (defender == kNoPlayer) {
        for (uint8_t p : participants) {
            if (p == NPC_PLAYER_ID) { defender = p; break; }
        }
    }
    if (defender == kNoPlayer) {
        uint32_t earliest = UINT32_MAX;
        for (uint8_t p : participants) {
            for (const Unit& u : state.unit_registry) {
                if (u.player_id == p && u.sector_id == cs.active_sector_id &&
                    u.arrival_order < earliest) {
                    earliest = u.arrival_order;
                    defender = p;
                }
            }
        }
    }

    // Attacker: the latest entrant among the remaining non-defenders.
    uint8_t attacker = kNoPlayer;
    uint32_t latest = 0;
    for (uint8_t p : participants) {
        if (p == defender) continue;
        for (const Unit& u : state.unit_registry) {
            if (u.player_id == p && u.sector_id == cs.active_sector_id &&
                u.arrival_order >= latest) {
                latest = u.arrival_order;
                attacker = p;
            }
        }
    }
    if (attacker == kNoPlayer || defender == kNoPlayer) return false;
    cs.current_attacker_id = attacker;
    cs.current_defender_id = defender;
    cs.initiative_idx = 0;
    cs.engagement_round = 0;
    cs.pending_player = kNoPlayer;
    cs.pending_target_count = 0;
    cs.pending_die_count = 0;
    cs.pending_target_group_player = kNoPlayer;
    RebuildInitiativeTimeline(state);
    cs.phase = CombatState::Phase::missile_phase;
    return true;
}

// Reputation value per destroyed ship type.
int ReputationValueFor(ShipType t) {
    switch (t) {
        case ShipType::INTERCEPTOR:
        case ShipType::STARBASE:
        case ShipType::ANCIENT:
            return 1;
        case ShipType::CRUISER:
        case ShipType::GUARDIAN:
            return 2;
        case ShipType::DREADNOUGHT:
        case ShipType::GCDS:
            return 3;
    }
    return 0;
}

void DrawOneReputationTile(::State& state, ReputationTiles value) {
    CombatState& cs = state.combat_state;
    // Remove one instance of `value` from the bag (order is irrelevant — draws
    // are sampled, so unselected tiles are simply appended back later).
    for (size_t i = 0; i < state.reputation_tiles.size(); ++i) {
        if (state.reputation_tiles[i] == value) {
            for (size_t j = i + 1; j < state.reputation_tiles.size(); ++j) {
                state.reputation_tiles[j - 1] = state.reputation_tiles[j];
            }
            state.reputation_tiles.pop_back();
            break;
        }
    }
    if (cs.drawn_tiles_size < kMaxReputationDraw) {
        cs.drawn_tiles[cs.drawn_tiles_size++] = value;
    }
}

// Compute the hex cells neighboring the active sector that are legal retreat
// destinations: wormhole-connected, controlled by retreating_player, and
// containing no opponent ships (Wormhole Generator relaxes connectivity).
void ComputeLegalRetreatDestinations(const ::State& state,
                                     uint8_t retreating_player,
                                     std::array<uint16_t, 6>& out,
                                     uint8_t& out_size) {
    out_size = 0;
    const uint16_t sector = state.combat_state.active_sector_id;
    const HexCoord src = state.galaxy.FindSectorCoord(sector);
    if (src.q == -128) return;
    const Sector& src_sec = state.galaxy.at(src.q, src.r);
    const SectorDefinition* src_def = get_sector_definition(src_sec.sector_id);
    if (src_def == nullptr) return;
    const uint8_t src_edges =
        rotate_edge_mask(src_def->wormholes_mask, src_sec.rotation);
    const bool wormhole_generator =
        retreating_player < state.players.size() &&
        state.players[retreating_player].has_tech(TechBit::WORMHOLE_GENERATOR);

    for (int d = 0; d < 6; ++d) {
        const int nq = src.q + HEX_DIRECTIONS[d].first;
        const int nr = src.r + HEX_DIRECTIONS[d].second;
        if (!in_galaxy_bounds(nq, nr)) continue;
        const Sector& ns = state.galaxy.at(nq, nr);
        if (ns.sector_id == 0) continue;
        if (ns.owner_id != retreating_player) continue;  // must Control it
        const SectorDefinition* ndef = get_sector_definition(ns.sector_id);
        if (ndef == nullptr) continue;
        const uint8_t n_edges = rotate_edge_mask(ndef->wormholes_mask, ns.rotation);
        const bool my_edge = has_edge(src_edges, d);
        const bool their_edge = has_edge(n_edges, (d + 3) % 6);
        const bool connected =
            wormhole_generator ? (my_edge || their_edge) : (my_edge && their_edge);
        if (!connected) continue;
        bool has_opponent = false;
        for (const Unit& u : state.unit_registry) {
            if (u.sector_id == ns.sector_id && u.player_id != retreating_player) {
                has_opponent = true;
                break;
            }
        }
        if (has_opponent) continue;
        if (out_size < out.size()) out[out_size++] = ns.sector_id;
    }
}

bool identify_combat_sectors(const ::State& state,
                             std::array<CombatSectorInfo, kMaxCombatBattles>& out,
                             uint8_t& out_size) {
    out_size = 0;
    SectorCoordIndex index;
    index.Build(state);

    std::array<uint16_t, 64> candidate_ids;
    uint8_t cand_size = 0;
    for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
        for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
            if (!in_galaxy_bounds(q, r)) continue;
            const Sector& s = state.galaxy.at(q, r);
            if (s.sector_id == 0) continue;
            auto participants = CollectSectorParticipants(state, s.sector_id);
            if (participants.size() < 2) continue;
            if (cand_size < candidate_ids.size()) {
                candidate_ids[cand_size++] = s.sector_id;
            }
        }
    }
    // Resolve battles in descending sector-number order.
    std::sort(candidate_ids.begin(), candidate_ids.begin() + cand_size,
              [](uint16_t a, uint16_t b) { return a > b; });
    for (uint8_t i = 0; i < cand_size && out_size < kMaxCombatBattles; ++i) {
        CombatSectorInfo info{};
        info.sector_id = candidate_ids[i];
        auto participants = CollectSectorParticipants(state, info.sector_id);
        info.participant_count = static_cast<uint8_t>(participants.size());
        for (uint8_t k = 0; k < info.participant_count; ++k) {
            info.participant_ids[k] = participants[k];
            uint32_t latest = 0;
            for (const Unit& u : state.unit_registry) {
                if (u.sector_id == info.sector_id &&
                    u.player_id == participants[k]) {
                    if (u.arrival_order > latest) latest = u.arrival_order;
                }
            }
            info.latest_arrival[k] = latest;
        }
        // Sort participants ascending by entry order (earliest first).
        for (uint8_t a = 0; a < info.participant_count; ++a) {
            for (uint8_t b = static_cast<uint8_t>(a + 1);
                 b < info.participant_count; ++b) {
                if (info.latest_arrival[b] < info.latest_arrival[a]) {
                    std::swap(info.latest_arrival[a], info.latest_arrival[b]);
                    std::swap(info.participant_ids[a], info.participant_ids[b]);
                }
            }
        }
        // Defender: NPC if present; else the player who Controls the sector;
        // else the earliest entrant.
        bool defender_found = false;
        for (uint8_t k = 0; k < info.participant_count; ++k) {
            if (IsNPC(info.participant_ids[k])) {
                info.defender_idx = k;
                defender_found = true;
                break;
            }
        }
        if (!defender_found) {
            const Sector* sec = index.Get(const_cast<::State&>(state),
                                          info.sector_id);
            if (sec != nullptr) {
                for (uint8_t k = 0; k < info.participant_count; ++k) {
                    if (sec->owner_id == info.participant_ids[k]) {
                        info.defender_idx = k;
                        defender_found = true;
                        break;
                    }
                }
            }
        }
        if (!defender_found) {
            uint8_t min_idx = 0;
            for (uint8_t k = 1; k < info.participant_count; ++k) {
                if (info.latest_arrival[k] < info.latest_arrival[min_idx]) {
                    min_idx = k;
                }
            }
            info.defender_idx = min_idx;
        }
        out[out_size++] = info;
    }
    return out_size > 0;
}

void begin_combat_phase(::State& state) {
    state.current_phase = RoundPhase::COMBAT;
    state.combat_state.Reset();
    state.combat_state.phase = CombatState::Phase::determine_battles;
    identify_combat_sectors(state, state.combat_state.battle_queue,
                            state.combat_state.battle_queue_size);
    if (state.combat_state.battle_queue_size == 0) {
        // No ship battles, but combat still has population, influence,
        // discovery, and repair steps.
        state.combat_state.phase = CombatState::Phase::attack_population;
        return;
    }
    state.combat_state.current_battle_idx = 0;
    state.combat_state.active_sector_id =
        state.combat_state.battle_queue[0].sector_id;
    StartNextPairInActiveSector(state);
}

// ----------------------------------------------------------------------
// Dice volleys
// ----------------------------------------------------------------------

void WeaponDicePerColor(const ::State& state, uint8_t attacker_id,
                        ShipType type, bool missiles,
                        uint8_t (&dice_per_color)[5]) {
    std::fill(std::begin(dice_per_color), std::end(dice_per_color), 0);
    if (attacker_id == NPC_PLAYER_ID) {
        const NPC* row = NpcRow(state, type);
        if (row != nullptr) {
            const DieColor color = missiles ? row->missile : row->cannon;
            if (color != DieColor::NONE) {
                dice_per_color[static_cast<int>(color)] =
                    missiles ? row->missile_amount : row->cannon_amount;
            }
        }
    } else {
        const auto& stats = state.players[attacker_id]
                                .blueprints[static_cast<size_t>(type)]
                                .total_stats;
        for (int i = 0; i < 5; ++i) {
            dice_per_color[i] = missiles ? stats.missiles[i] : stats.cannons[i];
        }
    }
}

uint8_t CountOccupiedSlots(uint16_t mask) {
    uint8_t count = 0;
    while (mask != 0) {
        mask &= static_cast<uint16_t>(mask - 1);
        ++count;
    }
    return count;
}

// Queue all dice for one (player, ship-type) group firing missiles or cannons.
// Values are left 0 (unrolled); each is rolled by a chance node. Returns the
// number of dice queued (0 if the group has no dice of the requested kind).
int SetupVolley(::State& state, uint8_t attacker_id, ShipType type,
                bool missiles, bool pop_attack) {
    CombatState& cs = state.combat_state;
    uint8_t dice_per_color[5] = {0, 0, 0, 0, 0};
    WeaponDicePerColor(state, attacker_id, type, missiles, dice_per_color);
    const int n_ships =
        pop_attack ? 1
                   : CountAliveInSector(state, attacker_id, type,
                                        cs.active_sector_id);
    cs.pending_die_count = 0;
    cs.pending_die_index = 0;
    for (int ship = 0; ship < n_ships; ++ship) {
        for (int c = 0; c < 5; ++c) {
            for (int d = 0; d < dice_per_color[c]; ++d) {
                if (cs.pending_die_count >= kMaxPendingDice) break;
                cs.pending_die_values[cs.pending_die_count] = 0;  // unrolled
                cs.pending_die_colors[cs.pending_die_count] =
                    static_cast<uint8_t>(c);
                ++cs.pending_die_count;
            }
        }
    }
    if (cs.pending_die_count == 0) {
        cs.pending_target_group_player = kNoPlayer;
        return 0;
    }
    cs.pending_target_group_player = attacker_id;
    cs.pending_target_group_type = type;
    cs.pending_dice_are_missiles = missiles;
    cs.pending_dice_pop_attack = pop_attack;
    return cs.pending_die_count;
}

// Called when the last die of a volley is resolved.
void OnVolleyComplete(::State& state) {
    CombatState& cs = state.combat_state;
    const bool pop_attack = cs.pending_dice_pop_attack;
    cs.pending_target_group_player = kNoPlayer;
    cs.pending_die_count = 0;
    cs.pending_die_index = 0;
    cs.pending_dice_are_missiles = false;
    cs.pending_dice_pop_attack = false;
    if (pop_attack) {
        // Population bombardment continues in the attack_population phase, which
        // owns the initiative-independent unit cursor.
        return;
    }
    ++cs.initiative_idx;
    if (cs.phase == CombatState::Phase::engagement_firing) {
        cs.phase = CombatState::Phase::choose_engagement_action;
    }
}

void AdvanceDie(::State& state) {
    CombatState& cs = state.combat_state;
    ++cs.pending_die_index;
    if (cs.pending_die_index >= cs.pending_die_count) {
        OnVolleyComplete(state);
    }
}

void ResolveCombatDie(::State& state, uint8_t face) {
    CombatState& cs = state.combat_state;
    if (cs.pending_target_group_player == kNoPlayer ||
        cs.pending_die_index >= cs.pending_die_count) {
        return;
    }
    const uint8_t i = cs.pending_die_index;
    cs.pending_die_values[i] = face;
    const uint8_t attacker = cs.pending_target_group_player;
    const ShipType type = cs.pending_target_group_type;
    const DieColor color = static_cast<DieColor>(cs.pending_die_colors[i]);

    // Rift self-damage applies regardless of how / whether the die is assigned.
    ApplyRiftSelfDamage(state, attacker, type, SelfDamageForDie(face, color));

    if (cs.pending_dice_pop_attack) {
        if (DieHitsShield(state, attacker, type, face, color, /*shield=*/0)) {
            Sector* sec = state.galaxy.FindSectorById(cs.pop_attack_sector_id);
            const int cap = sec ? CountOccupiedSlots(sec->occupied_slots_mask) : 0;
            const int total = std::min<int>(
                cap, cs.pop_attack_damage_remaining + DamageForDie(face, color));
            cs.pop_attack_damage_remaining = static_cast<uint8_t>(total);
        }
        AdvanceDie(state);
        return;
    }

    const int damage = DamageForDie(face, color);
    const uint8_t hittable =
        CountHittableTargets(state, attacker, type, face, color);
    // A player decision is only worth a node when the die deals damage and the
    // player has a genuine target choice (two or more ships it could hit).
    // NPC fire, misses, and forced single-target hits resolve automatically.
    const bool player_choice =
        attacker != NPC_PLAYER_ID && damage > 0 && hittable >= 2;
    if (player_choice) {
        return;  // leave the rolled die for the firing player to assign
    }
    if (damage > 0 && hittable >= 1) {
        const size_t best = ChooseBestTarget(state, attacker, type, face, color);
        if (best != SIZE_MAX) {
            ApplyHitDamage(state, attacker, type, face, color, best);
        }
    }
    AdvanceDie(state);
}

void ApplyPlayerDieTarget(::State& state, size_t target_idx) {
    CombatState& cs = state.combat_state;
    if (cs.pending_target_group_player == kNoPlayer ||
        cs.pending_die_index >= cs.pending_die_count) {
        return;
    }
    const uint8_t i = cs.pending_die_index;
    const uint8_t face = cs.pending_die_values[i];
    if (face == 0) return;  // not rolled yet
    const uint8_t attacker = cs.pending_target_group_player;
    const ShipType type = cs.pending_target_group_type;
    const DieColor color = static_cast<DieColor>(cs.pending_die_colors[i]);
    // Self-damage was already applied at roll time; only apply target damage.
    ApplyHitDamage(state, attacker, type, face, color, target_idx);
    AdvanceDie(state);
}

// ----------------------------------------------------------------------
// State machine
// ----------------------------------------------------------------------

// Pick the next battle. Resets engagement bookkeeping.
void AdvanceToNextBattle(::State& state) {
    FlushDestroyedShips(state);
    CombatState& cs = state.combat_state;
    cs.destroyed_ships_size = 0;
    cs.engagement_round = 0;
    cs.reputation_drawn_mask.fill(0);
    cs.reputation_retreat_penalty_mask.fill(0);
    cs.retreating_group_count = 0;
    cs.current_attacker_id = kNoPlayer;
    cs.current_defender_id = kNoPlayer;
    cs.tile_select_player = kNoPlayer;
    cs.rep_draw_target = 0;
    cs.drawn_tiles_size = 0;
    ++cs.current_battle_idx;
    if (cs.current_battle_idx >= cs.battle_queue_size) {
        cs.phase = CombatState::Phase::attack_population;
        return;
    }
    cs.active_sector_id = cs.battle_queue[cs.current_battle_idx].sector_id;
    StartNextPairInActiveSector(state);
}

bool StepCombat(::State& state) {
    CombatState& cs = state.combat_state;
    switch (cs.phase) {
        case CombatState::Phase::inactive:
            return false;
        case CombatState::Phase::determine_battles:
            return false;  // already done in begin_combat_phase
        case CombatState::Phase::missile_phase: {
            // Missiles fire once, by ship type, in initiative order.
            while (cs.initiative_idx < cs.initiative_size) {
                const InitiativeGroup& g =
                    cs.initiative_timeline[cs.initiative_idx];
                cs.active_ship_type = g.type;
                if (SetupVolley(state, g.player_id, g.type, /*missiles=*/true,
                                /*pop_attack=*/false) > 0) {
                    return false;  // dice rolled via chance; idx advances on done
                }
                ++cs.initiative_idx;
            }
            cs.engagement_round = 1;
            cs.initiative_idx = 0;
            cs.phase = CombatState::Phase::choose_engagement_action;
            return false;
        }
        case CombatState::Phase::choose_engagement_action: {
            if (!AdvanceToNextAliveGroup(state)) {
                if (CountCurrentPairParticipantsWithShips(state) > 1 &&
                    cs.engagement_round < kMaxEngagementRounds) {
                    ++cs.engagement_round;
                    RebuildInitiativeTimeline(state);
                    return false;
                }
                if (CountCurrentPairParticipantsWithShips(state) > 1 &&
                    cs.engagement_round >= kMaxEngagementRounds &&
                    cs.current_attacker_id != kNoPlayer) {
                    // Stalemate breaker: the attacker's ships are destroyed.
                    for (size_t i = 0; i < state.unit_registry.size(); ++i) {
                        Unit& u = state.unit_registry[i];
                        if (u.player_id == cs.current_attacker_id &&
                            u.sector_id == cs.active_sector_id) {
                            RecordDestroyedShip(state, u.player_id, u.type,
                                                cs.current_defender_id);
                            u.sector_id = kGraveyardSectorId;
                        }
                    }
                }
                if (CountParticipantsWithShips(state, cs.active_sector_id) > 1 &&
                    StartNextPairInActiveSector(state)) {
                    return false;
                }
                cs.phase = CombatState::Phase::select_reputation_tile;
                return false;
            }
            const InitiativeGroup& g = cs.initiative_timeline[cs.initiative_idx];
            if (g.is_npc) {
                // NPCs never retreat; they always attack.
                cs.active_ship_type = g.type;
                cs.phase = CombatState::Phase::engagement_firing;
                return false;
            }
            cs.pending_player = g.player_id;
            cs.active_ship_type = g.type;
            ComputeLegalRetreatDestinations(state, g.player_id,
                                            cs.retreat_destinations,
                                            cs.retreat_destinations_size);
            return true;
        }
        case CombatState::Phase::engagement_firing: {
            if (cs.initiative_idx < cs.initiative_size) {
                const InitiativeGroup& g =
                    cs.initiative_timeline[cs.initiative_idx];
                cs.active_ship_type = g.type;
                if (SetupVolley(state, g.player_id, g.type, /*missiles=*/false,
                                /*pop_attack=*/false) > 0) {
                    return false;  // dice rolled via chance; OnVolleyComplete
                                   // advances idx and returns to choose phase
                }
                ++cs.initiative_idx;
            }
            cs.phase = CombatState::Phase::choose_engagement_action;
            return false;
        }
        case CombatState::Phase::select_reputation_tile: {
            if (cs.tile_select_player != kNoPlayer) {
                // A draw / selection is in flight; handled by the chance and
                // decision layers, not by stepping.
                return false;
            }
            const auto& b = cs.battle_queue[cs.current_battle_idx];
            // Players draw in order of entry; participant_ids is sorted ascending
            // by arrival, so the earliest entrant (the defender) draws first.
            uint8_t next_player = kNoPlayer;
            uint8_t next_k = 0;
            for (uint8_t k = 0; k < b.participant_count; ++k) {
                const uint8_t pid = b.participant_ids[k];
                if (pid == NPC_PLAYER_ID) continue;
                if (cs.reputation_drawn_mask[k]) continue;
                next_player = pid;
                next_k = k;
                break;
            }
            if (next_player == kNoPlayer) {
                AdvanceToNextBattle(state);
                return false;
            }
            // 1 tile for participating (unless every ship retreated) + tiles for
            // each destroyed opponent ship, capped at five.
            int earned = cs.reputation_retreat_penalty_mask[next_k] ? 0 : 1;
            for (uint8_t i = 0; i < cs.destroyed_ships_size; ++i) {
                const auto& rec = cs.destroyed_ships[i];
                if (rec.destroyed_by != next_player) continue;
                earned += ReputationValueFor(rec.type) * rec.count;
            }
            cs.reputation_drawn_mask[next_k] = 1;  // processed
            cs.reputation_earned = static_cast<uint8_t>(std::min<int>(earned, 255));
            const uint8_t target = static_cast<uint8_t>(std::min<int>(
                std::min<int>(kMaxReputationDraw, earned),
                static_cast<int>(state.reputation_tiles.size())));
            if (target == 0) return false;  // nothing to draw; next participant
            cs.tile_select_player = next_player;
            cs.rep_draw_target = target;
            cs.drawn_tiles_size = 0;
            return false;  // draws happen via chance, then the player selects
        }
        case CombatState::Phase::attack_population: {
            if (cs.pop_attack_damage_remaining > 0) return true;  // assign cubes
            while (cs.pop_attack_unit_index < state.unit_registry.size()) {
                Unit& attacker = state.unit_registry[cs.pop_attack_unit_index];
                ++cs.pop_attack_unit_index;
                if (attacker.sector_id == kGraveyardSectorId) continue;
                if (attacker.player_id == NPC_PLAYER_ID) continue;
                Sector* sector = state.galaxy.FindSectorById(attacker.sector_id);
                if (sector == nullptr || sector->owner_id == 255) continue;
                if (sector->owner_id == attacker.player_id) continue;
                if (sector->occupied_slots_mask == 0) continue;
                cs.pop_attack_sector_id = sector->sector_id;
                cs.pop_attack_player = attacker.player_id;
                cs.pop_attack_owner = sector->owner_id;
                cs.pop_attack_damage_remaining = 0;
                // Neutron Bombs destroy all population automatically (no roll).
                if (attacker.player_id < state.players.size() &&
                    state.players[attacker.player_id].has_tech(
                        TechBit::NEUTRON_BOMBS)) {
                    cs.pop_attack_damage_remaining =
                        CountOccupiedSlots(sector->occupied_slots_mask);
                    return true;
                }
                if (SetupVolley(state, attacker.player_id, attacker.type,
                                /*missiles=*/false, /*pop_attack=*/true) > 0) {
                    return false;  // bombardment dice rolled via chance
                }
            }
            cs.pop_attack_unit_index = 0;
            cs.phase = CombatState::Phase::influence_sectors;
            return false;
        }
        case CombatState::Phase::influence_sectors: {
            if (cs.influence_uncontrolled_size == 0 &&
                cs.influence_scan_index == 0 &&
                cs.influence_decision_player == kNoPlayer) {
                for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
                    for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
                        if (!in_galaxy_bounds(q, r)) continue;
                        Sector& s = state.galaxy.at(q, r);
                        if (s.sector_id == 0) continue;
                        if (s.owner_id == 255) continue;  // already unowned
                        if (s.occupied_slots_mask != 0) continue;
                        bool has_ship = false;
                        for (const Unit& u : state.unit_registry) {
                            if (u.sector_id == s.sector_id) {
                                has_ship = true;
                                break;
                            }
                        }
                        if (!has_ship) continue;
                        if (s.owner_id < state.players.size() &&
                            state.players[s.owner_id].disks_on_sectors > 0) {
                            state.players[s.owner_id].disks_on_sectors--;
                        }
                        s.owner_id = 255;
                        if (cs.influence_uncontrolled_size < kMaxGalaxyCells) {
                            cs.influence_uncontrolled_sectors[
                                cs.influence_uncontrolled_size++] = s.sector_id;
                        }
                    }
                }
            }
            if (cs.influence_uncontrolled_size == 0) {
                cs.phase = CombatState::Phase::discovery_award;
                return false;
            }

            while (cs.influence_scan_index < cs.influence_uncontrolled_size) {
                const uint16_t sid =
                    cs.influence_uncontrolled_sectors[cs.influence_scan_index];
                Sector* sector = state.galaxy.FindSectorById(sid);
                if (sector == nullptr || sector->owner_id != 255) {
                    ++cs.influence_scan_index;
                    continue;
                }
                for (uint8_t p = cs.influence_turn_order_index;
                     p < MAX_PLAYERS; ++p) {
                    const uint8_t player = state.turn_order[p];
                    if (player >= MAX_PLAYERS) break;
                    if (player >= state.players.size()) continue;
                    if (state.players[player].eliminated) continue;
                    if (state.players[player].available_influence_discs() == 0) {
                        continue;
                    }
                    for (const Unit& u : state.unit_registry) {
                        if (u.sector_id == sid && u.player_id == player) {
                            cs.influence_decision_player = player;
                            cs.influence_decision_sector = sid;
                            cs.influence_turn_order_index = p;
                            return true;
                        }
                    }
                }
                ++cs.influence_scan_index;
                cs.influence_turn_order_index = 0;
            }
            cs.influence_decision_player = kNoPlayer;
            cs.influence_decision_sector = 0;
            cs.influence_uncontrolled_size = 0;
            cs.influence_scan_index = 0;
            cs.influence_turn_order_index = 0;
            cs.phase = CombatState::Phase::discovery_award;
            return false;
        }
        case CombatState::Phase::discovery_award: {
            for (int q = -GALAXY_RADIUS; q <= GALAXY_RADIUS; ++q) {
                for (int r = -GALAXY_RADIUS; r <= GALAXY_RADIUS; ++r) {
                    if (!in_galaxy_bounds(q, r)) continue;
                    const Sector& s = state.galaxy.at(q, r);
                    if (s.sector_id == 0) continue;
                    if (!s.discovery_tile_present) continue;
                    for (uint8_t p = 0; p < state.players.size(); ++p) {
                        for (const Unit& u : state.unit_registry) {
                            if (u.sector_id == s.sector_id && u.player_id == p) {
                                cs.discovery_decision_sector = s.sector_id;
                                cs.discovery_decision_player = p;
                                return true;
                            }
                        }
                    }
                }
            }
            cs.phase = CombatState::Phase::repair;
            return false;
        }
        case CombatState::Phase::repair: {
            for (size_t i = 0; i < state.unit_registry.size(); ++i) {
                state.unit_registry[i].damage = 0;
            }
            state.combat_state.Reset();
            state.combat_state.phase = CombatState::Phase::inactive;
            return false;
        }
    }
    return false;
}

void advance_combat_state(::State& state) { StepCombat(state); }

}  // namespace open_spiel::eclipse
