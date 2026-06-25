//
// Diplomacy: forming, rearranging, and breaking Diplomatic Relations
// (rulebook p.14-15). Player↔Player relations only;
//
// This is a free bonus action (like Trade / Colony Ship): taking or
// resolving a Diplomacy proposal does NOT consume an influence disc and
// does NOT advance the turn. It may be initiated at any point during any
// of the proposer's Actions.
//
// Breakage: when the Move / Combat action end leaves the aggressor with
// a Ship in a Sector where a partner has a Ship or Control, the relation
// is broken and the Traitor Tile is assigned to the aggressor. The
// returned Pop Cubes are NOT auto-placed on a track; each side picks the
// track at the start of their next Action via the deferred
// choose_return_track sub-action.
//

#ifndef ECLIPSE_DIPLOMACY_H
#define ECLIPSE_DIPLOMACY_H

#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>

#include "open_spiel/games/eclipse/types.h"

// Forward declarations to avoid circular dependency
struct State;
struct Player;
enum class PopTrack : uint8_t;

// ── Diplomacy sub-state machine ──────────────────────────────────────────────
// Models a free bonus action (like Trade / Colony Ship) that any player may
// take during their own Action to form or rearrange Diplomatic Relations.
// Also drives the deferred return-track choice that follows a break.
struct DiplomacyState {
    enum class Phase : uint8_t {
        inactive            = 0,
        choose_partner      = 1,   // proposer picks a partner (currently unused)
        choose_rearrange    = 2,   // either side frees a Reputation Track slot
        choose_pop_track    = 3,   // giver picks which Pop Track the cube comes from
        choose_return_track = 4,   // one side of a break picks the Pop Track for the returned cube
        choose_accept       = 5,   // partner accepts or declines a proposal (rulebook p.14)
    };

    Phase phase = Phase::inactive;
    uint8_t player_id = 255;
    uint8_t partner_id = 255;
    // choose_rearrange: which side is acting (0 = proposer, 1 = partner). 255 = none.
    uint8_t rearrange_side = 255;
    // choose_pop_track: 0 = proposer is choosing, 1 = partner is choosing.
    uint8_t pop_track_side = 0;
    // choose_return_track: 0 = aggressor is choosing, 1 = victim is choosing.
    uint8_t return_side = 255;
    // Formation: track currently being picked (default MONEY for serialization
    // stability). 0=Money, 1=Science, 2=Materials. Stored as uint8_t to avoid
    // requiring the full PopTrack enum at this header's include depth.
    uint8_t selected_track = 0;
};

NLOHMANN_JSON_SERIALIZE_ENUM(DiplomacyState::Phase, {
    {DiplomacyState::Phase::inactive,            "inactive"},
    {DiplomacyState::Phase::choose_partner,      "choose_partner"},
    {DiplomacyState::Phase::choose_rearrange,    "choose_rearrange"},
    {DiplomacyState::Phase::choose_pop_track,    "choose_pop_track"},
    {DiplomacyState::Phase::choose_return_track, "choose_return_track"},
    {DiplomacyState::Phase::choose_accept,       "choose_accept"},
});

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DiplomacyState,
    phase, player_id, partner_id, rearrange_side, pop_track_side, return_side, selected_track);

namespace open_spiel::eclipse
{
    // ReputationSlot, ReputationSlotKind, ReputationTiles are brought in via
    // types.h. PopTrack is forward-declared and only used by-value in this
    // header, so the full definition in bonus.h is not required.

    // Canonical 5-slot Reputation Track layout (rulebook p.14):
    //   2× "May hold either an Ambassador or a Reputation Tile"
    //   1× "May hold only an Ambassador Tile"
    //   2× "May hold only a Reputation Tile"
    constexpr uint8_t kReputationTrackSlots = 5;
    constexpr ReputationSlotKind kCanonicalSlotKinds[kReputationTrackSlots] = {
        ReputationSlotKind::AMBASSADOR_OR_REP,
        ReputationSlotKind::AMBASSADOR_OR_REP,
        ReputationSlotKind::AMBASSADOR_ONLY,
        ReputationSlotKind::REP_ONLY,
        ReputationSlotKind::REP_ONLY,
    };

    // True if `slot` (a Reputation Track slot index) can hold an Ambassador tile.
    inline bool slot_kind_holds_ambassador(ReputationSlotKind k) {
        return k == ReputationSlotKind::AMBASSADOR_OR_REP ||
               k == ReputationSlotKind::AMBASSADOR_ONLY;
    }

    // True if `slot` can hold a Reputation tile.
    inline bool slot_kind_holds_rep(ReputationSlotKind k) {
        return k == ReputationSlotKind::AMBASSADOR_OR_REP ||
               k == ReputationSlotKind::REP_ONLY;
    }

    // Returns the index of the first free slot that can hold an Ambassador,
    // or 255 if none.
    uint8_t find_free_ambassador_slot(const ::Player& player);

    // Returns the index of the highest-index free Reputation-capable slot, or 255.
    uint8_t find_free_rep_slot(const ::Player& player);

    // True if the player has any free slot that can hold an Ambassador tile
    // (AMBASSADOR_OR_REP or AMBASSADOR_ONLY).
    bool has_free_ambassador_slot(const ::Player& player);

    // True if the player can free a slot for an Ambassador by either:
    //   - returning a Reputation tile to the bag (from any Rep-capable slot), or
    //   - swapping a Reputation tile from an AMBASSADOR_OR_REP slot into a free
    //     REP_ONLY slot.
    bool has_freeable_ambassador_slot(const ::Player& player);

    // Returns true if `a` and `b` have a wormhole connection between Sectors
    // they each Control. Includes Warp Portal adjacency (rulebook p.7).
    // Excludes the Wormhole Generator shortcut — the rulebook disallows it
    // specifically for Diplomatic Relations.
    bool has_diplomacy_wormhole_connection(const ::State& state, uint8_t a, uint8_t b);

    // All Diplomacy formation preconditions (rulebook p.14):
    //   - 4+ player game
    //   - proposer and partner are different, alive, both have empty or
    //     freeable Reputation Track slots
    //   - neither holds the Traitor Tile
    //   - neither holds the other's Ambassador Tile
    //   - a wormhole (or warp portal) connection joins Sectors they Control
    //   - the connecting sectors contain no opposing player's ships
    //   - no sector anywhere has co-located ships from both players
    bool can_propose_diplomacy(const ::State& state, uint8_t proposer, uint8_t partner);

    // Begin a Diplomacy formation. Validates can_propose_diplomacy, then
    // transitions diplomacy_state to choose_accept for the partner to
    // accept or decline. Returns false if the proposal cannot proceed at
    // all (preconditions not met / rejected).
    bool begin_diplomacy(::State& state, uint8_t proposer, uint8_t partner);

    // Formation sub-action: proposer picks a Pop Track for their cube, then
    // the partner picks a Pop Track for theirs. Each call consumes one side.
    bool execute_diplomacy_pick_track(::State& state, uint8_t player_id, PopTrack track);

    // Formation sub-action: return a Reputation tile to the bag to free a
    // slot for an Ambassador. Validated against the current rearrange_side.
    bool execute_return_rep_to_bag(::State& state, uint8_t player_id, uint8_t slot_idx);

    // Formation sub-action: swap a Reputation tile from an AMBASSADOR_OR_REP
    // slot to a free REP_ONLY slot to free the AMBASSADOR_OR_REP slot.
    bool execute_swap_rep_slots(::State& state, uint8_t player_id,
                                uint8_t from_slot, uint8_t to_slot);

    // Finalize the formation after both track picks are in: exchange
    // Ambassador Tiles and Pop Cubes, and clear the DiplomacyState. Called
    // automatically by execute_diplomacy_pick_track when the second side
    // commits; may also be called directly to commit after both tracks
    // have been picked.
    bool commit_diplomacy_formation(::State& state);

    // Partner accepts the proposal (called from ApplyDiplomacySubAction).
    // Transitions to formation (choose_rearrange or choose_pop_track) and
    // sets proposer as the acting player for the next sub-phase.
    bool execute_diplomacy_accept(::State& state);

    // Partner declines the proposal. Clears diplomacy state; proposer
    // resumes their Action.
    void execute_diplomacy_decline(::State& state);

    // Break: called by AdvanceTurn when an Act of Aggression is detected.
    // Scans ALL of the aggressor's ships (rulebook p.14: "at the end of
    // your Action") for co-location with any diplomatic partner's ships or
    // control. For each partner:
    //   - clear aggressor's Reputation Track slot for that partner
    //   - clear partner's Reputation Track slot for aggressor
    //   - clear the previous Traitor Tile holder's traitor_held
    //   - mark the aggressor as the new Traitor Tile holder
    //   - enqueue deferred return-track choices (aggressor first)
    // Returns true if at least one relation was broken.
    bool break_all_diplomacy_for(::State& state, uint8_t aggressor);

    // Resolves the deferred return-track choice for the slot currently
    // awaiting it. Called by execute_choose_return_track with the track
    // chosen by the player. Returns false if the choice is not pending
    // or the track is invalid (track must currently have room — i.e. < 12).
    bool execute_choose_return_track(::State& state, uint8_t player_id, PopTrack track);

    // Returns the player_id whose return-track choice is currently pending,
    // or 255 if none. Used by LegalActions to gate the action set.
    uint8_t pending_return_track_player(const ::State& state);

    // Returns true if the slot holds a returnable Reputation tile (a
    // Reputation tile in a Rep-capable slot that is not also holding an
    // Ambassador).
    bool slot_is_returnable(const ::open_spiel::eclipse::ReputationSlot& slot);
} // namespace open_spiel::eclipse

#endif // ECLIPSE_DIPLOMACY_H
