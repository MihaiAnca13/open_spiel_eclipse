//
// Diplomacy implementation. See diplomacy.h for the rulebook mapping.
//

#include "open_spiel/games/eclipse/systems/actions/diplomacy.h"

#include <array>
#include <bitset>

#include "open_spiel/games/eclipse/state.h"
#include "open_spiel/games/eclipse/galaxy.h"
#include "open_spiel/games/eclipse/sectors.h"
#include "open_spiel/games/eclipse/species.h"
#include "open_spiel/games/eclipse/tech.h"
#include "open_spiel/games/eclipse/systems/actions/bonus.h"

namespace open_spiel::eclipse
{
    namespace
    {
        constexpr uint8_t NO_SLOT = 255;
        constexpr uint8_t NO_PLAYER = 255;

        constexpr uint8_t kPopTrackMax = 12;

        bool increment_track(::Player& player, PopTrack track) {
            switch (track) {
                case PopTrack::MONEY: {
                    if (player.resources.gold_prod >= kPopTrackMax) return false;
                    ++player.resources.gold_prod;
                    return true;
                }
                case PopTrack::SCIENCE: {
                    if (player.resources.science_prod >= kPopTrackMax) return false;
                    ++player.resources.science_prod;
                    return true;
                }
                case PopTrack::MATERIALS: {
                    if (player.resources.materials_prod >= kPopTrackMax) return false;
                    ++player.resources.materials_prod;
                    return true;
                }
            }
            return false;
        }

        bool sector_controlled_by(const ::State& state, uint8_t player_id, uint16_t sector_id) {
            const Sector* s = state.galaxy.FindSectorById(sector_id);
            return s != nullptr && s->owner_id == player_id;
        }

        bool sector_has_opponent_ship(const ::State& state, uint8_t opponent_id, uint16_t sector_id) {
            if (sector_id == 0) return false;
            for (const Unit& u : state.unit_registry) {
                if (u.player_id == opponent_id && u.sector_id == sector_id &&
                    u.type != ShipType::GCDS && u.type != ShipType::STARBASE) {
                    return true;
                }
            }
            return false;
        }

        bool has_diplomacy_wormhole_inner(const ::State& state, uint8_t a, uint8_t b) {
            for (int qa = -GALAXY_RADIUS; qa <= GALAXY_RADIUS; ++qa) {
                for (int ra = -GALAXY_RADIUS; ra <= GALAXY_RADIUS; ++ra) {
                    if (!in_galaxy_bounds(qa, ra)) continue;
                    const Sector& sa = state.galaxy.at(qa, ra);
                    if (sa.sector_id == 0 || sa.owner_id != a) continue;
                    const SectorDefinition* da = get_sector_definition(sa.sector_id);
                    if (da == nullptr) continue;
                    const uint8_t a_edges = rotate_edge_mask(da->wormholes_mask, sa.rotation);

                    const bool a_has_portal = sa.player_warp_portal_vp != 0 || da->has_warp_portal;
                    if (a_has_portal && player_has_warp_portal_anchor(state, b)) {
                        return true;
                    }

                    for (uint8_t dir = 0; dir < 6; ++dir) {
                        const int qb = qa + HEX_DIRECTIONS[dir].first;
                        const int rb = ra + HEX_DIRECTIONS[dir].second;
                        if (!in_galaxy_bounds(qb, rb)) continue;
                        const Sector& sb = state.galaxy.at(qb, rb);
                        if (sb.sector_id == 0 || sb.owner_id != b) continue;
                        const SectorDefinition* db = get_sector_definition(sb.sector_id);
                        if (db == nullptr) continue;
                        const uint8_t b_edges = rotate_edge_mask(db->wormholes_mask, sb.rotation);

                        if (has_edge(a_edges, dir) && has_edge(b_edges, (dir + 3) % 6)) {
                            if (!sector_has_opponent_ship(state, b, sa.sector_id) &&
                                !sector_has_opponent_ship(state, a, sb.sector_id)) {
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }
    }  // namespace

    // ── Slot helpers ─────────────────────────────────────────────────────────

    bool slot_is_returnable(const ReputationSlot& slot) {
        if (slot.holds_ambassador) return false;
        if (!slot_kind_holds_rep(slot.kind)) return false;
        return true;
    }

    uint8_t find_free_ambassador_slot(const ::Player& player) {
        for (size_t i = 0; i < player.reputation_track.size(); ++i) {
            const ReputationSlot& s = player.reputation_track[i];
            if (!s.holds_ambassador && slot_kind_holds_ambassador(s.kind)) {
                return static_cast<uint8_t>(i);
            }
        }
        return NO_SLOT;
    }

    uint8_t find_free_rep_slot(const ::Player& player) {
        for (size_t i = player.reputation_track.size(); i > 0; --i) {
            const size_t idx = i - 1;
            const ReputationSlot& s = player.reputation_track[idx];
            if (!s.holds_ambassador && slot_kind_holds_rep(s.kind)) {
                return static_cast<uint8_t>(idx);
            }
        }
        return NO_SLOT;
    }

    bool has_free_ambassador_slot(const ::Player& player) {
        return find_free_ambassador_slot(player) != NO_SLOT;
    }

    bool has_freeable_ambassador_slot(const ::Player& player) {
        if (has_free_ambassador_slot(player)) return true;
        for (size_t i = 0; i < player.reputation_track.size(); ++i) {
            if (slot_is_returnable(player.reputation_track[i])) return true;
        }
        return false;
    }

    bool has_ambassador_from(const ::Player& player, uint8_t from_id) {
        for (const auto& slot : player.reputation_track) {
            if (slot.holds_ambassador && slot.ambassador_from == from_id) return true;
        }
        return false;
    }

    bool has_diplomacy_wormhole_connection(const ::State& state, uint8_t a, uint8_t b) {
        if (a >= state.players.size() || b >= state.players.size()) return false;
        return has_diplomacy_wormhole_inner(state, a, b);
    }

    // ── Rulebook p.14: "no sector anywhere has co-located ships" ──────────────
    // Diplomatic Relations are not allowed to be proposed between players if
    // either player's Ships are present in a Sector Controlled by or containing
    // a Ship from the other player.
    bool sector_has_co_located_players(const ::State& state,
                                        uint8_t a, uint8_t b) {
        for (const Unit& u : state.unit_registry) {
            if (u.sector_id == 0) continue;
            if (u.player_id == a) {
                if (sector_controlled_by(state, b, u.sector_id) ||
                    sector_has_opponent_ship(state, b, u.sector_id)) {
                    return true;
                }
            }
            if (u.player_id == b) {
                if (sector_controlled_by(state, a, u.sector_id) ||
                    sector_has_opponent_ship(state, a, u.sector_id)) {
                    return true;
                }
            }
        }
        return false;
    }

    bool can_propose_diplomacy(const ::State& state, uint8_t proposer, uint8_t partner) {
        if (proposer >= state.players.size() || partner >= state.players.size()) return false;
        if (proposer == partner) return false;
        const ::Player& p = state.players[proposer];
        const ::Player& q = state.players[partner];
        if (p.eliminated || q.eliminated) return false;
        if (state.players.size() < 4) return false;
        if (p.traitor_held || q.traitor_held) return false;
        if (!has_freeable_ambassador_slot(p)) return false;
        if (!has_freeable_ambassador_slot(q)) return false;
        for (size_t i = 0; i < p.reputation_track.size(); ++i) {
            if (p.reputation_track[i].holds_ambassador &&
                p.reputation_track[i].ambassador_from == partner) {
                return false;
            }
        }
        for (size_t i = 0; i < q.reputation_track.size(); ++i) {
            if (q.reputation_track[i].holds_ambassador &&
                q.reputation_track[i].ambassador_from == proposer) {
                return false;
            }
        }
        if (!has_diplomacy_wormhole_connection(state, proposer, partner)) return false;
        // Fix #5: scan ALL sectors for co-located ships (rulebook p.14).
        if (sector_has_co_located_players(state, proposer, partner)) return false;
        return true;
    }

    // ── Fix #1: partner accept/decline ────────────────────────────────────────
    // Rulebook p.14: "If either player declines the proposed Diplomatic
    // Relations, the current player simply continues their Action."

    bool begin_diplomacy(::State& state, uint8_t proposer, uint8_t partner) {
        if (!can_propose_diplomacy(state, proposer, partner)) return false;

        state.diplomacy_state = DiplomacyState{};
        state.diplomacy_state.player_id = proposer;
        state.diplomacy_state.partner_id = partner;
        state.diplomacy_state.phase = DiplomacyState::Phase::choose_accept;
        return true;
    }

    bool execute_diplomacy_accept(::State& state) {
        DiplomacyState& ds = state.diplomacy_state;
        if (ds.phase != DiplomacyState::Phase::choose_accept) return false;
        if (ds.player_id >= state.players.size() || ds.partner_id >= state.players.size()) {
            ds = DiplomacyState{};
            return false;
        }

        // Move to formation: check slot availability, same logic as old
        // begin_diplomacy body.
        if (!has_free_ambassador_slot(state.players[ds.player_id])) {
            ds.phase = DiplomacyState::Phase::choose_rearrange;
            ds.rearrange_side = 0;
        } else if (!has_free_ambassador_slot(state.players[ds.partner_id])) {
            ds.phase = DiplomacyState::Phase::choose_rearrange;
            ds.rearrange_side = 1;
        } else {
            ds.phase = DiplomacyState::Phase::choose_pop_track;
            ds.pop_track_side = 0;
        }
        return true;
    }

    void execute_diplomacy_decline(::State& state) {
        state.diplomacy_state = DiplomacyState{};
    }

    bool execute_return_rep_to_bag(::State& state, uint8_t player_id, uint8_t slot_idx) {
        DiplomacyState& ds = state.diplomacy_state;
        if (ds.phase != DiplomacyState::Phase::choose_rearrange) return false;
        const uint8_t expected = (ds.rearrange_side == 0) ? ds.player_id : ds.partner_id;
        if (player_id != expected) return false;
        if (player_id >= state.players.size()) return false;
        ::Player& player = state.players[player_id];
        if (slot_idx >= player.reputation_track.size()) return false;

        ReputationSlot& slot = player.reputation_track[slot_idx];
        if (!slot_is_returnable(slot)) return false;

        state.reputation_tiles.push_back(slot.rep_value);
        slot.rep_value = ReputationTiles::NONE;
        slot.holds_ambassador = false;
        slot.ambassador_from = NO_PLAYER;
        slot.pending_track_choice = false;

        if (has_free_ambassador_slot(state.players[ds.player_id]) &&
            has_free_ambassador_slot(state.players[ds.partner_id])) {
            ds.phase = DiplomacyState::Phase::choose_pop_track;
            ds.pop_track_side = 0;
            ds.rearrange_side = NO_PLAYER;
        } else {
            ds.rearrange_side = 1 - ds.rearrange_side;
            if (!has_freeable_ambassador_slot(state.players[ds.player_id]) &&
                !has_freeable_ambassador_slot(state.players[ds.partner_id])) {
                ds = DiplomacyState{};
                return false;
            }
        }
        return true;
    }

    bool execute_swap_rep_slots(::State& state, uint8_t player_id,
                                uint8_t from_slot, uint8_t to_slot) {
        DiplomacyState& ds = state.diplomacy_state;
        if (ds.phase != DiplomacyState::Phase::choose_rearrange) return false;
        const uint8_t expected = (ds.rearrange_side == 0) ? ds.player_id : ds.partner_id;
        if (player_id != expected) return false;
        if (player_id >= state.players.size()) return false;
        ::Player& player = state.players[player_id];
        if (from_slot >= player.reputation_track.size()) return false;
        if (to_slot >= player.reputation_track.size()) return false;
        if (from_slot == to_slot) return false;

        ReputationSlot& from = player.reputation_track[from_slot];
        ReputationSlot& to = player.reputation_track[to_slot];

        if (from.holds_ambassador || to.holds_ambassador) return false;
        if (from.kind != ReputationSlotKind::AMBASSADOR_OR_REP) return false;
        if (to.kind != ReputationSlotKind::REP_ONLY) return false;

        std::swap(from.rep_value, to.rep_value);

        if (has_free_ambassador_slot(state.players[ds.player_id]) &&
            has_free_ambassador_slot(state.players[ds.partner_id])) {
            ds.phase = DiplomacyState::Phase::choose_pop_track;
            ds.pop_track_side = 0;
            ds.rearrange_side = NO_PLAYER;
        } else {
            ds.rearrange_side = 1 - ds.rearrange_side;
        }
        return true;
    }

    bool execute_diplomacy_pick_track(::State& state, uint8_t player_id, PopTrack track) {
        DiplomacyState& ds = state.diplomacy_state;
        if (ds.phase != DiplomacyState::Phase::choose_pop_track) return false;
        const uint8_t expected = (ds.pop_track_side == 0) ? ds.player_id : ds.partner_id;
        if (player_id != expected) return false;
        if (player_id >= state.players.size()) return false;
        ::Player& player = state.players[player_id];
        if (track == PopTrack::MONEY && player.resources.gold_prod == 0) return false;
        if (track == PopTrack::SCIENCE && player.resources.science_prod == 0) return false;
        if (track == PopTrack::MATERIALS && player.resources.materials_prod == 0) return false;

        if (track == PopTrack::MONEY) --player.resources.gold_prod;
        else if (track == PopTrack::SCIENCE) --player.resources.science_prod;
        else if (track == PopTrack::MATERIALS) --player.resources.materials_prod;

        if (ds.pop_track_side == 0) {
            ds.pop_track_side = 1;
            return true;
        }
        return commit_diplomacy_formation(state);
    }

    bool commit_diplomacy_formation(::State& state) {
        DiplomacyState& ds = state.diplomacy_state;
        if (ds.phase != DiplomacyState::Phase::choose_pop_track) return false;
        if (ds.player_id >= state.players.size() || ds.partner_id >= state.players.size()) return false;

        const uint8_t proposer_slot = find_free_ambassador_slot(state.players[ds.player_id]);
        const uint8_t partner_slot = find_free_ambassador_slot(state.players[ds.partner_id]);
        if (proposer_slot == NO_SLOT || partner_slot == NO_SLOT) {
            ::Player& p = state.players[ds.player_id];
            ::Player& q = state.players[ds.partner_id];
            if (p.resources.gold_prod < kPopTrackMax) ++p.resources.gold_prod;
            else if (p.resources.science_prod < kPopTrackMax) ++p.resources.science_prod;
            else if (p.resources.materials_prod < kPopTrackMax) ++p.resources.materials_prod;
            if (q.resources.gold_prod < kPopTrackMax) ++q.resources.gold_prod;
            else if (q.resources.science_prod < kPopTrackMax) ++q.resources.science_prod;
            else if (q.resources.materials_prod < kPopTrackMax) ++q.resources.materials_prod;
            ds = DiplomacyState{};
            return false;
        }

        // Write Ambassador Tiles. Slot kind is never mutated (per rulebook p.14).
        ReputationSlot& ps = state.players[ds.player_id].reputation_track[proposer_slot];
        ps.holds_ambassador = true;
        ps.ambassador_from = ds.partner_id;
        ps.rep_value = ReputationTiles::NONE;
        ps.pending_track_choice = false;

        ReputationSlot& qs = state.players[ds.partner_id].reputation_track[partner_slot];
        qs.holds_ambassador = true;
        qs.ambassador_from = ds.player_id;
        qs.rep_value = ReputationTiles::NONE;
        qs.pending_track_choice = false;

        if (state.players[ds.player_id].ambassador_tiles_held < 255) {
            ++state.players[ds.player_id].ambassador_tiles_held;
        }
        if (state.players[ds.partner_id].ambassador_tiles_held < 255) {
            ++state.players[ds.partner_id].ambassador_tiles_held;
        }

        ds = DiplomacyState{};
        return true;
    }

    // ── Fix #2, #3, #4: Traitor Tile transfer, full ship scan, multi-partner ──
    bool break_all_diplomacy_for(::State& state, uint8_t aggressor) {
        if (aggressor >= state.players.size()) return false;
        ::Player& agg = state.players[aggressor];
        if (agg.eliminated) return false;

        // Fix #3: Scan ALL of the aggressor's ships (rulebook p.14: "at the
        // end of your Action") for co-location with any diplomatic partner.
        std::array<bool, MAX_PLAYERS> partners_to_break{};
        for (uint8_t p = 0; p < state.players.size(); ++p) {
            if (p == aggressor) continue;
            if (state.players[p].eliminated) continue;
            if (!has_ambassador_from(agg, p)) continue;

            bool co_located = false;
            for (const Unit& u : state.unit_registry) {
                if (u.player_id != aggressor) continue;
                if (u.sector_id == 0) continue;
                if (sector_controlled_by(state, p, u.sector_id) ||
                    sector_has_opponent_ship(state, p, u.sector_id)) {
                    co_located = true;
                    break;
                }
            }
            if (co_located) partners_to_break[p] = true;
        }

        bool any_broken = false;
        for (uint8_t p = 0; p < state.players.size(); ++p) {
            if (!partners_to_break[p]) continue;

            for (size_t i = 0; i < agg.reputation_track.size(); ++i) {
                ReputationSlot& s = agg.reputation_track[i];
                if (s.holds_ambassador && s.ambassador_from == p) {
                    s.holds_ambassador = false;
                    s.ambassador_from = NO_PLAYER;
                    s.rep_value = ReputationTiles::NONE;
                    s.pending_track_choice = true;
                    if (agg.ambassador_tiles_held > 0) --agg.ambassador_tiles_held;
                    if (agg.ambassador_tiles_pending_return < 255) {
                        ++agg.ambassador_tiles_pending_return;
                    }
                    break;
                }
            }

            ::Player& partner = state.players[p];
            for (size_t i = 0; i < partner.reputation_track.size(); ++i) {
                ReputationSlot& s = partner.reputation_track[i];
                if (s.holds_ambassador && s.ambassador_from == aggressor) {
                    s.holds_ambassador = false;
                    s.ambassador_from = NO_PLAYER;
                    s.rep_value = ReputationTiles::NONE;
                    s.pending_track_choice = true;
                    if (partner.ambassador_tiles_held > 0) --partner.ambassador_tiles_held;
                    if (partner.ambassador_tiles_pending_return < 255) {
                        ++partner.ambassador_tiles_pending_return;
                    }
                    break;
                }
            }

            any_broken = true;
        }

        if (any_broken) {
            // Fix #2: clear previous holder before assigning (rulebook p.15:
            // "receives the Traitor Tile from its previous holder or the supply").
            for (auto& player : state.players) {
                if (player.id != aggressor) {
                    player.traitor_held = false;
                }
            }
            agg.traitor_held = true;

            // Fix #4: enqueue ALL pending returns. Find the first player
            // (aggressor or partner) with a pending_track_choice.
            state.diplomacy_state = DiplomacyState{};
            state.diplomacy_state.phase = DiplomacyState::Phase::choose_return_track;
            // Scan all players for the first pending return. Set return_side=0
            // for the owner of the first pending slot.
            uint8_t first_pending = NO_PLAYER;
            for (uint8_t p = 0; p < state.players.size(); ++p) {
                for (const auto& slot : state.players[p].reputation_track) {
                    if (slot.pending_track_choice) {
                        first_pending = p;
                        break;
                    }
                }
                if (first_pending != NO_PLAYER) break;
            }
            if (first_pending != NO_PLAYER) {
                state.diplomacy_state.player_id = first_pending;
                state.diplomacy_state.return_side = 0;
                // partner_id is unused in the multi-partner model; we scan
                // globally on every resolve.
                state.diplomacy_state.partner_id = NO_PLAYER;
            } else {
                state.diplomacy_state = DiplomacyState{};
            }
        }
        return any_broken;
    }

    // ── Fix #4: multi-partner aware return-track resolution ───────────────────
    bool execute_choose_return_track(::State& state, uint8_t player_id, PopTrack track) {
        DiplomacyState& ds = state.diplomacy_state;
        if (ds.phase != DiplomacyState::Phase::choose_return_track) return false;
        if (player_id >= state.players.size()) return false;
        ::Player& player = state.players[player_id];

        int pending_idx = -1;
        for (size_t i = 0; i < player.reputation_track.size(); ++i) {
            if (player.reputation_track[i].pending_track_choice) {
                pending_idx = static_cast<int>(i);
                break;
            }
        }
        if (pending_idx < 0) {
            ds = DiplomacyState{};
            return false;
        }

        if (!increment_track(player, track)) return false;
        ReputationSlot& slot = player.reputation_track[pending_idx];
        slot.pending_track_choice = false;
        if (player.ambassador_tiles_pending_return > 0) {
            --player.ambassador_tiles_pending_return;
        }

        // Scan all players for remaining pending slots to decide next actor.
        uint8_t next_pending = NO_PLAYER;
        for (uint8_t p = 0; p < state.players.size(); ++p) {
            for (const auto& s : state.players[p].reputation_track) {
                if (s.pending_track_choice) {
                    next_pending = p;
                    break;
                }
            }
            if (next_pending != NO_PLAYER) break;
        }

        if (next_pending != NO_PLAYER) {
            ds.player_id = next_pending;
            ds.return_side = 0;
            ds.partner_id = NO_PLAYER;
        } else {
            ds = DiplomacyState{};
        }
        return true;
    }

    uint8_t pending_return_track_player(const ::State& state) {
        if (state.diplomacy_state.phase != DiplomacyState::Phase::choose_return_track) {
            return NO_PLAYER;
        }
        return state.diplomacy_state.player_id;
    }
}  // namespace open_spiel::eclipse
