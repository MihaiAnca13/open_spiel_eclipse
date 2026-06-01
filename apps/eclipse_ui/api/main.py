import json

from fastapi import Body, FastAPI
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
