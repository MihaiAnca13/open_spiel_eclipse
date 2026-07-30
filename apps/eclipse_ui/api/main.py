import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import eclipse_ui_native

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── sector tile art ─────────────────────────────────────────────────────────
# Serve the raw sector tile images and a sector_id -> filename manifest so the
# UI can paint real art inside each hex (debug aid for the explore action).

SECTORS_DIR = Path(__file__).resolve().parent.parent / "data" / "sectors"
TECH_DIR = Path(__file__).resolve().parent.parent / "data" / "tech"
SHIPS_DIR = Path(__file__).resolve().parent.parent / "data" / "ships"
DISCOVERY_DIR = Path(__file__).resolve().parent.parent / "data" / "discovery"
MINOR_SPECIES_DIR = Path(__file__).resolve().parent.parent / "data" / "minor_species"


def _build_sector_manifest() -> dict[str, str]:
    """Map sector_id -> image filename by parsing the 3-digit id out of each name.

    Trap: `sector_394_simeis_147.png` has two 3-digit groups; the id is the one
    right after the `sector_` prefix (394). All other prefixes carry the id as
    the trailing group (`galactic_center_001`->1, `starting_procyon_221`->221).
    """
    manifest: dict[str, str] = {}
    for p in sorted(SECTORS_DIR.glob("*.png")):
        m = re.match(r"^sector_(\d{3})_", p.name) or re.search(r"(\d{3})\.png$", p.name)
        if not m:
            print(f"[sector manifest] could not parse sector_id from {p.name}")
            continue
        manifest[str(int(m.group(1)))] = p.name
    return manifest


SECTOR_MANIFEST = _build_sector_manifest()

LAYOUTS_FILE = Path(__file__).resolve().parent.parent / "data" / "sector_layouts.json"


def _load_sector_layouts() -> dict:
    if LAYOUTS_FILE.exists():
        import json as _json
        return _json.loads(LAYOUTS_FILE.read_text())
    return {}


SECTOR_LAYOUTS = _load_sector_layouts()

app.mount("/assets/sectors", StaticFiles(directory=str(SECTORS_DIR)), name="sectors")
app.mount("/assets/tech", StaticFiles(directory=str(TECH_DIR)), name="tech")
app.mount("/assets/ships", StaticFiles(directory=str(SHIPS_DIR)), name="ships")
app.mount("/assets/discovery", StaticFiles(directory=str(DISCOVERY_DIR)), name="discovery")
app.mount("/assets/minor_species", StaticFiles(directory=str(MINOR_SPECIES_DIR)), name="minor_species")


@app.get("/sectors/manifest")
async def sectors_manifest() -> dict[str, str]:
    return SECTOR_MANIFEST


@app.get("/sectors/layouts")
async def sectors_layouts() -> dict:
    return SECTOR_LAYOUTS


# ─── existing setup endpoints ────────────────────────────────────────────────

@app.post("/setup/pre-choice")
async def setup_pre_choice(
    config: dict = Body(..., description="UI-driven setup configuration."),
):
    config = _normalize_setup_config(config)
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


def _warped_universe_supported(num_players: int) -> bool:
    return 3 <= num_players <= 5


def _normalize_setup_config(config: dict) -> dict:
    normalized = dict(config)
    warped_universe = bool(normalized.get("warped_universe", False))
    players = normalized.get("players")
    if isinstance(players, int) and not _warped_universe_supported(players):
        warped_universe = False
    normalized["warped_universe"] = warped_universe
    return normalized


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
    warped_universe: bool = False
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


def _require_debug_host(player_id: str) -> None:
    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can use debug state controls")
    if lobby.phase != "started" or lobby.game_blob is None:
        raise HTTPException(status_code=400, detail="Game not started")


def _validate_debug_game_blob(blob: dict, expected_players: int | None = None) -> int:
    if not isinstance(blob, dict):
        raise HTTPException(status_code=400, detail="game_blob must be an object")

    for key in ("setup_config", "state", "pending_random_event"):
        if key not in blob:
            raise HTTPException(status_code=400, detail=f"game_blob missing {key}")

    setup_config = blob["setup_config"]
    if not isinstance(setup_config, dict):
        raise HTTPException(status_code=400, detail="game_blob.setup_config must be an object")
    blob["setup_config"] = _normalize_setup_config(setup_config)
    setup_config = blob["setup_config"]

    players = setup_config.get("players")
    if not isinstance(players, int):
        raise HTTPException(status_code=400, detail="game_blob.setup_config.players must be an integer")
    if players < 2 or players > 6:
        raise HTTPException(status_code=400, detail="game_blob.setup_config.players must be 2-6")
    if expected_players is not None and players != expected_players:
        raise HTTPException(
            status_code=400,
            detail=f"Loaded state has {players} players, current lobby has {expected_players}",
        )

    if not isinstance(blob["state"], dict):
        raise HTTPException(status_code=400, detail="game_blob.state must be an object")
    if not isinstance(blob["pending_random_event"], int):
        raise HTTPException(status_code=400, detail="game_blob.pending_random_event must be an integer")

    try:
        _legal_info(blob)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid game state blob: {exc}") from exc
    return players


def _snapshot_from_game_blob(blob: dict) -> dict:
    return {
        "config": blob["setup_config"],
        "state": blob["state"],
        "finalized": True,
    }


def _seat_species(blob: dict, player_idx: int) -> str:
    state_players = blob.get("state", {}).get("players", [])
    if player_idx < len(state_players):
        species = state_players[player_idx].get("species_id")
        if isinstance(species, str) and species:
            return species

    staged_players = blob.get("setup_config", {}).get("staged_players", [])
    if player_idx < len(staged_players):
        species = staged_players[player_idx].get("species")
        if isinstance(species, str) and species:
            return species

    return TERRAN_SPECIES


def _core_player_is_ai(blob: dict, player_idx: int) -> bool:
    state_players = blob.get("state", {}).get("players", [])
    if player_idx < len(state_players):
        return bool(state_players[player_idx].get("is_ai", False))
    staged_players = blob.get("setup_config", {}).get("staged_players", [])
    if player_idx < len(staged_players):
        return bool(staged_players[player_idx].get("is_ai", False))
    return False


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
        "warped_universe": lby.warped_universe,
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
    if lobby.phase == "started":
        if seat.state == "ai":
            raise HTTPException(status_code=400, detail="Cannot claim an AI seat")
        if seat.player_id is not None and seat.player_id != player_id:
            raise HTTPException(status_code=409, detail="Seat is already claimed")

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
    if not _warped_universe_supported(num):
        lobby.warped_universe = False
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


@app.post("/lobby/warped_universe")
async def lobby_warped_universe(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    warped_universe = body.get("warped_universe", False)

    if player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can change Warped Universe")

    if not isinstance(warped_universe, bool):
        raise HTTPException(status_code=400, detail="warped_universe must be a boolean")

    if warped_universe and not _warped_universe_supported(lobby.num_players):
        raise HTTPException(status_code=400, detail="Warped Universe is supported for 3-5 players")

    lobby.warped_universe = warped_universe
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
        "warped_universe": lobby.warped_universe,
    }
    setup_config = _normalize_setup_config(setup_config)

    stage1_str = eclipse_ui_native.initialize_pre_choice(json.dumps(setup_config))
    stage1 = json.loads(stage1_str)

    # Compute picker order: reverse of turn order, human seats only
    raw_turn_order: List[int] = [p for p in stage1["state"]["turn_order"] if p != 255]
    human_seat_set = {i for i, s in enumerate(lobby.seats) if s.state == "human"}
    picker_order = [i for i in reversed(raw_turn_order) if i in human_seat_set]

    lobby.stage1_snapshot = stage1
    lobby.warped_universe = bool(stage1["config"].get("warped_universe", False))
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


@app.post("/debug/state/dump")
async def debug_state_dump(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    _require_debug_host(player_id)
    return {"game_blob": lobby.game_blob}


@app.post("/debug/state/load")
async def debug_state_load(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    _require_debug_host(player_id)

    game_blob = body.get("game_blob")
    _validate_debug_game_blob(game_blob, expected_players=lobby.num_players)

    lobby.game_blob = game_blob
    lobby.snapshot = _snapshot_from_game_blob(game_blob)
    lobby.warped_universe = bool(game_blob["setup_config"].get("warped_universe", False))
    for idx, seat in enumerate(lobby.seats):
        seat.species = _seat_species(game_blob, idx)
    _autoplay_ai()

    await broadcast_lobby()
    return serialize_lobby(lobby).get("snapshot")


@app.post("/debug/state/start")
async def debug_state_start(body: dict = Body(...)):
    player_id: str = body.get("player_id", "")
    player_name: str = body.get("player_name") or "Player"

    if lobby.phase != "waiting":
        raise HTTPException(status_code=400, detail="Can only start from state before setup")
    if lobby.host_player_id is not None and player_id != lobby.host_player_id:
        raise HTTPException(status_code=403, detail="Only host can start from debug state")

    game_blob = body.get("game_blob")
    players = _validate_debug_game_blob(game_blob)
    host_seat_idx = next(
        (idx for idx in range(players) if not _core_player_is_ai(game_blob, idx)),
        0,
    )

    seats: list[Seat] = []
    for idx in range(players):
        is_ai = _core_player_is_ai(game_blob, idx)
        seat = Seat(
            state="ai" if is_ai else "human",
            species=_seat_species(game_blob, idx),
        )
        if idx == host_seat_idx:
            seat.state = "human"
            seat.player_id = player_id
            seat.player_name = player_name
            seat.last_player_id = player_id
        seats.append(seat)

    lobby.host_player_id = player_id
    lobby.num_players = players
    lobby.seats = seats
    lobby.difficulty = game_blob["state"].get("gcds_difficulty", lobby.difficulty)
    lobby.rng_seed = int(game_blob["setup_config"].get("rng_seed", lobby.rng_seed))
    lobby.warped_universe = bool(game_blob["setup_config"].get("warped_universe", False))
    lobby.phase = "started"
    lobby.stage1_snapshot = None
    lobby.snapshot = _snapshot_from_game_blob(game_blob)
    lobby.game_blob = game_blob
    lobby.picker_order = []
    lobby.current_picker_idx = 0

    _autoplay_ai()

    await broadcast_lobby()
    serialized = serialize_lobby(lobby)
    return {
        "lobby": serialized,
        "snapshot": serialized.get("snapshot"),
        "seat": host_seat_idx,
    }


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
    lobby.warped_universe = False
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
