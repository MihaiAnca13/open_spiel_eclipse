import json
import random
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from . import eclipse_ui_native

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── existing setup endpoints ────────────────────────────────────────────────

@app.post("/setup/pre-choice")
async def setup_pre_choice(
    config: dict = Body(..., description="UI-driven setup configuration."),
):
    snapshot_json_str = eclipse_ui_native.initialize_pre_choice(json.dumps(config))
    return json.loads(snapshot_json_str)


@app.post("/setup/finalize")
async def setup_finalize(
    snapshot: dict = Body(..., description="Stage 1 setup snapshot."),
    player_choices: list = Body(
        ..., description="Final player choices [{'species': 'Planta', 'is_ai': false}, ...]"
    ),
):
    finalized_snapshot_json_str = eclipse_ui_native.finalize_game_setup(
        json.dumps(snapshot), json.dumps(player_choices)
    )
    return json.loads(finalized_snapshot_json_str)


@app.get("/metadata")
async def metadata():
    metadata_json_str = eclipse_ui_native.get_game_metadata()
    return json.loads(metadata_json_str)


# ─── lobby ───────────────────────────────────────────────────────────────────

TERRAN_SPECIES = "Terran Factions"
TERRAN_STARTING_SECTORS = [221, 223, 225, 227, 229, 231]


def _random_seed() -> int:
    return random.randint(0, 2**31 - 1)


@dataclass
class Seat:
    state: Literal["empty", "human", "ai"] = "empty"
    player_id: Optional[str] = None
    player_name: Optional[str] = None
    species: str = TERRAN_SPECIES
    # Last human who held this seat. Kept when the seat is freed on disconnect
    # so the original player auto-reclaims it on return (while still empty).
    last_player_id: Optional[str] = None


@dataclass
class Lobby:
    host_player_id: Optional[str] = None
    num_players: int = 2
    seats: List[Seat] = field(default_factory=lambda: [Seat(), Seat()])
    difficulty: str = "Easy"
    rng_seed: int = field(default_factory=_random_seed)
    phase: str = "waiting"  # "waiting" | "setup" | "started"
    stage1_snapshot: Optional[dict] = None
    snapshot: Optional[dict] = None
    # Canonical EclipseState::Serialize blob — the source of truth for gameplay
    # (the UI-facing `snapshot` is derived from this after each action).
    game_blob: Optional[dict] = None
    picker_order: List[int] = field(default_factory=list)   # seat indices, human only
    current_picker_idx: int = 0
    connections: list = field(default_factory=list)
    conn_player_ids: dict = field(default_factory=dict)  # websocket -> player_id

    @property
    def started(self) -> bool:
        return self.phase == "started"


lobby = Lobby()


# ─── gameplay helpers ─────────────────────────────────────────────────────────

PASS_ACTION = 0


def _legal_info(blob: dict) -> dict:
    """Decision facing the current player: {current_player, is_terminal,
    legal_actions, action_strings}."""
    return json.loads(eclipse_ui_native.legal_actions(json.dumps(blob)))


def _apply(blob: dict, action_id: int) -> dict:
    """Apply one action and auto-resolve chance, returning the new blob."""
    return json.loads(eclipse_ui_native.apply_action(json.dumps(blob), action_id))


def _all_humans_passed(blob: dict) -> bool:
    players = blob.get("state", {}).get("players", [])
    for idx, seat in enumerate(lobby.seats):
        if seat.state == "human" and idx < len(players):
            if not players[idx].get("has_passed", False):
                return False
    return True


def _autoplay_ai(max_steps: int = 1000) -> None:
    """Resolve AI/NPC seats until it is a human's turn (or the game ends).
    An AI passes when every human has already passed (so the round can end);
    otherwise it picks a uniform-random legal action (incl. explore sub-steps)."""
    for _ in range(max_steps):
        if lobby.game_blob is None:
            return
        info = _legal_info(lobby.game_blob)
        if info["is_terminal"]:
            return
        cur = info["current_player"]
        if cur is None or cur < 0 or cur >= len(lobby.seats):
            return
        if lobby.seats[cur].state != "ai":
            return  # human's turn (or mid-explore human) — hand control back
        legal = info["legal_actions"]
        if not legal:
            return
        if PASS_ACTION in legal and _all_humans_passed(lobby.game_blob):
            action = PASS_ACTION
        else:
            action = random.choice(legal)
        lobby.game_blob = _apply(lobby.game_blob, action)


def serialize_lobby(lby: Lobby) -> dict:
    d: dict = {
        "host_player_id": lby.host_player_id,
        "num_players": lby.num_players,
        "seats": [
            {
                "state": s.state,
                "player_id": s.player_id,
                "player_name": s.player_name,
                "species": s.species,
                "last_player_id": s.last_player_id,
            }
            for s in lby.seats
        ],
        "difficulty": lby.difficulty,
        "rng_seed": lby.rng_seed,
        "phase": lby.phase,
        "picker_order": lby.picker_order,
        "current_picker_idx": lby.current_picker_idx,
    }
    if lby.phase in ("setup", "started") and lby.stage1_snapshot:
        d["stage1_snapshot"] = lby.stage1_snapshot
    if lby.phase == "started" and lby.snapshot:
        snapshot = dict(lby.snapshot)
        if lby.game_blob is not None:
            snapshot["state"] = lby.game_blob["state"]
            info = _legal_info(lby.game_blob)
            snapshot["legal_actions"] = info["legal_actions"]
            snapshot["action_strings"] = info["action_strings"]
            snapshot["current_player"] = info["current_player"]
            snapshot["is_terminal"] = info["is_terminal"]
        d["snapshot"] = snapshot
    return d


async def broadcast_lobby() -> None:
    msg = {"type": "lobby_state", "lobby": serialize_lobby(lobby)}
    dead = []
    for ws in list(lobby.connections):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in lobby.connections:
            lobby.connections.remove(ws)


@app.get("/lobby")
async def get_lobby():
    return serialize_lobby(lobby)


@app.post("/lobby/join")
async def lobby_join(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    player_name: str = body.get("player_name") or "Player"

    # Reconnect: player already has a seat
    for seat in lobby.seats:
        if seat.player_id == player_id:
            return serialize_lobby(lobby)

    # Auto-reclaim: a seat we held was freed (disconnect) and is still empty.
    for seat in lobby.seats:
        if seat.state == "empty" and seat.last_player_id == player_id:
            seat.state = "human"
            seat.player_id = player_id
            seat.player_name = player_name
            await broadcast_lobby()
            return serialize_lobby(lobby)

    if lobby.started:
        raise HTTPException(status_code=409, detail="Game already started")

    if not any(s.state == "empty" for s in lobby.seats):
        raise HTTPException(status_code=409, detail="Lobby is full")

    for seat in lobby.seats:
        if seat.state == "empty":
            seat.state = "human"
            seat.player_id = player_id
            seat.player_name = player_name
            seat.last_player_id = player_id
            if lobby.host_player_id is None:
                lobby.host_player_id = player_id
            break

    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/seat/{idx}/species")
async def lobby_seat_species(idx: int, body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    species: str = body.get("species", "")

    if idx < 0 or idx >= len(lobby.seats):
        raise HTTPException(status_code=404, detail="Seat not found")

    seat = lobby.seats[idx]
    is_host = player_id == lobby.host_player_id
    is_owner = seat.player_id == player_id

    if seat.state == "ai":
        if not is_host:
            raise HTTPException(status_code=403, detail="Only host can change AI species")
    elif seat.state == "human":
        if not is_owner:
            raise HTTPException(status_code=403, detail="Can only change your own species")
        # Humans pick species only during setup, only on their own pick turn.
        if lobby.phase != "setup":
            raise HTTPException(status_code=400, detail="Species are chosen during setup")
        if lobby.current_picker_idx >= len(lobby.picker_order):
            raise HTTPException(status_code=403, detail="All species picks are done")
        if lobby.picker_order[lobby.current_picker_idx] != idx:
            raise HTTPException(status_code=403, detail="Not your turn to pick species")
        seat.species = species
        lobby.current_picker_idx += 1
        await broadcast_lobby()
        return serialize_lobby(lobby)
    else:
        raise HTTPException(status_code=400, detail="Cannot set species on empty seat")

    seat.species = species
    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/seat/{idx}/set_ai")
async def lobby_seat_set_ai(idx: int, body: dict = Body(...)):
    player_id: str = body.get("player_id", "")

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can mark seats as AI")

    if idx < 0 or idx >= len(lobby.seats):
        raise HTTPException(status_code=404, detail="Seat not found")

    seat = lobby.seats[idx]
    seat.state = "ai"
    seat.player_id = None
    seat.player_name = None

    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/seat/{idx}/set_empty")
async def lobby_seat_set_empty(idx: int, body: dict = Body(...)):
    """Host reverts a seat (AI or human) back to empty so a player can join."""
    player_id: str = body.get("player_id", "")

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can open a seat")

    if idx < 0 or idx >= len(lobby.seats):
        raise HTTPException(status_code=404, detail="Seat not found")

    if lobby.phase != "waiting":
        raise HTTPException(status_code=400, detail="Cannot open seats after game is initialized")

    seat = lobby.seats[idx]
    seat.state = "empty"
    seat.player_id = None
    seat.player_name = None
    seat.species = TERRAN_SPECIES
    # Host explicitly opened this seat — drop the old owner's reclaim hold.
    seat.last_player_id = None

    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/claim-seat/{idx}")
async def lobby_claim_seat(idx: int, body: dict = Body(...)):
    """Claim a specific seat — used when a player's session expired and they need to reconnect."""
    player_id: str = body.get("player_id", "")
    player_name: str = body.get("player_name") or "Player"

    if idx < 0 or idx >= len(lobby.seats):
        raise HTTPException(status_code=404, detail="Seat not found")

    # Player must not already have a seat
    for seat in lobby.seats:
        if seat.player_id == player_id:
            raise HTTPException(status_code=400, detail="Already have a seat")

    seat = lobby.seats[idx]
    seat.state = "human"
    seat.player_id = player_id
    seat.player_name = player_name
    seat.last_player_id = player_id
    # Keep existing species

    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/num_players")
async def lobby_num_players(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    num = body.get("num_players", 0)

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can change player count")

    if not isinstance(num, int) or num < 2 or num > 6:
        raise HTTPException(status_code=400, detail="num_players must be 2-6")

    current = len(lobby.seats)
    if num > current:
        for _ in range(num - current):
            lobby.seats.append(Seat())
    elif num < current:
        lobby.seats = lobby.seats[:num]

    lobby.num_players = num
    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/difficulty")
async def lobby_difficulty(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    difficulty: str = body.get("difficulty", "")

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can change difficulty")

    lobby.difficulty = difficulty
    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/seed")
async def lobby_seed(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    rng_seed = body.get("rng_seed", 42)

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can change seed")

    if not isinstance(rng_seed, int):
        raise HTTPException(status_code=400, detail="rng_seed must be an integer")

    lobby.rng_seed = rng_seed
    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/initialize")
async def lobby_initialize(body: dict = Body(...)):
    """Stage 1: resolve randomness (turn order, tech market, sectors). No species needed yet."""
    player_id: str = body.get("player_id", "")

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can initialize the game")

    if any(s.state == "empty" for s in lobby.seats):
        raise HTTPException(status_code=400, detail="All seats must be filled before initializing")

    if lobby.phase != "waiting":
        raise HTTPException(status_code=400, detail="Game already initialized")

    staged_players = [{"species": s.species, "is_ai": s.state == "ai"} for s in lobby.seats]
    setup_config = {
        "players": lobby.num_players,
        "rng_seed": lobby.rng_seed,
        "npc_difficulty": lobby.difficulty,
        "staged_players": staged_players,
    }

    stage1_str = eclipse_ui_native.initialize_pre_choice(json.dumps(setup_config))
    stage1 = json.loads(stage1_str)

    # Compute picker order: reverse of turn order, human seats only
    raw_turn_order: List[int] = [p for p in stage1["state"]["turn_order"] if p != 255]
    human_seat_set = {i for i, s in enumerate(lobby.seats) if s.state == "human"}
    picker_order = [i for i in reversed(raw_turn_order) if i in human_seat_set]

    lobby.stage1_snapshot = stage1
    lobby.picker_order = picker_order
    lobby.current_picker_idx = 0
    lobby.phase = "setup"

    await broadcast_lobby()
    return serialize_lobby(lobby)


@app.post("/lobby/finalize")
async def lobby_finalize(body: dict = Body(...)):
    """Stage 2: apply species choices, spawn galaxy map, start game."""
    player_id: str = body.get("player_id", "")

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can finalize the game")

    if lobby.phase != "setup":
        raise HTTPException(status_code=400, detail="Must initialize before finalizing")

    if not lobby.stage1_snapshot:
        raise HTTPException(status_code=400, detail="No stage 1 snapshot found")

    player_choices = [{"species": s.species, "is_ai": s.state == "ai"} for s in lobby.seats]
    finalized_str = eclipse_ui_native.finalize_game_setup(
        json.dumps(lobby.stage1_snapshot), json.dumps(player_choices)
    )
    finalized = json.loads(finalized_str)

    lobby.snapshot = finalized
    # Canonical gameplay blob (EclipseState::Serialize format). pending=0 and an
    # empty rng_state mean DeserializeState reseeds from the config rng_seed.
    lobby.game_blob = {
        "setup_config": finalized["config"],
        "state": finalized["state"],
        "pending_random_event": 0,
        "rng_state": "",
    }
    lobby.phase = "started"

    # If turn order opens on AI seats, resolve them up front so the first human
    # to act sees a live decision.
    _autoplay_ai()

    await broadcast_lobby()
    return {"lobby": serialize_lobby(lobby), "snapshot": serialize_lobby(lobby).get("snapshot")}


@app.post("/game/action")
async def game_action(body: dict = Body(...)):
    """Apply a gameplay action for the seat whose turn it is, then auto-resolve
    any following AI seats. Broadcasts the new state to all clients."""
    if lobby.phase != "started" or lobby.game_blob is None:
        raise HTTPException(status_code=400, detail="Game not started")

    seat = body.get("seat", body.get("player_id"))
    action_id = body.get("action_id")
    if not isinstance(seat, int) or not isinstance(action_id, int):
        raise HTTPException(status_code=400, detail="seat and action_id are required ints")

    info = _legal_info(lobby.game_blob)
    if info["is_terminal"]:
        raise HTTPException(status_code=400, detail="Game is over")
    if info["current_player"] != seat:
        raise HTTPException(status_code=403, detail="Not your turn")
    if seat < 0 or seat >= len(lobby.seats) or lobby.seats[seat].state != "human":
        raise HTTPException(status_code=403, detail="Not a controllable human seat")
    if action_id not in info["legal_actions"]:
        raise HTTPException(status_code=400, detail="Illegal action")

    lobby.game_blob = _apply(lobby.game_blob, action_id)
    _autoplay_ai()

    await broadcast_lobby()
    return serialize_lobby(lobby).get("snapshot")


@app.post("/lobby/reset")
async def lobby_reset(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can reset the lobby")

    lobby.host_player_id = None
    lobby.num_players = 2
    lobby.seats = [Seat(), Seat()]
    lobby.difficulty = "Easy"
    lobby.rng_seed = _random_seed()
    lobby.phase = "waiting"
    lobby.stage1_snapshot = None
    lobby.snapshot = None
    lobby.game_blob = None
    lobby.picker_order = []
    lobby.current_picker_idx = 0

    await broadcast_lobby()
    return serialize_lobby(lobby)


def _free_seat_on_disconnect(player_id: str) -> bool:
    """Free a disconnected player's seat (waiting phase only) so others see it
    open. Keeps last_player_id so the player auto-reclaims it on return. No-op
    if the player still has another live connection, or mid-setup/started."""
    if not player_id or lobby.phase != "waiting":
        return False
    if player_id in lobby.conn_player_ids.values():
        return False  # still connected on another tab
    for seat in lobby.seats:
        if seat.player_id == player_id:
            seat.state = "empty"
            seat.player_id = None
            seat.player_name = None
            seat.species = TERRAN_SPECIES
            seat.last_player_id = player_id
            return True
    return False


@app.websocket("/ws")
async def ws_lobby(websocket: WebSocket, player_id: str = ""):
    await websocket.accept()
    lobby.connections.append(websocket)
    lobby.conn_player_ids[websocket] = player_id
    await websocket.send_json({"type": "lobby_state", "lobby": serialize_lobby(lobby)})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in lobby.connections:
            lobby.connections.remove(websocket)
        lobby.conn_player_ids.pop(websocket, None)
        if _free_seat_on_disconnect(player_id):
            await broadcast_lobby()
