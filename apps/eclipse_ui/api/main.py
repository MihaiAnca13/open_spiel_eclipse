import json
from fastapi import FastAPI, Body
from . import eclipse_ui_native

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins, perfect for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/setup/pre-choice")
async def setup_pre_choice(seed: int = 42, num_players: int = 4, difficulty: str = "Easy"):
    # Call C++ Stage 1 setup (returns a JSON string)
    state_json_str = eclipse_ui_native.initialize_pre_choice(seed, num_players, difficulty)
    # Parse back into a Python dict so FastAPI can return standard JSON
    return json.loads(state_json_str)

@app.post("/setup/finalize")
async def setup_finalize(
    state: dict = Body(..., description="The current Stage 1 State dict"),
    player_choices: list = Body(..., description="List of chosen player configurations [{'species': 'Planta', 'is_ai': false}, ...]")
):
    # Convert inputs to strings to cross the C++ boundary
    state_str = json.dumps(state)
    choices_str = json.dumps(player_choices)
    
    # Call C++ Stage 2 setup (returns updated JSON string)
    updated_state_json_str = eclipse_ui_native.finalize_game_setup(state_str, choices_str)
    
    # Parse and return updated state as dict
    return json.loads(updated_state_json_str)
