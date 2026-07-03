from __future__ import annotations

import asyncio

from api.main import setup_finalize, setup_pre_choice


async def _main() -> None:
    config = {
        "players": 4,
        "rng_seed": 42,
        "npc_difficulty": "Easy",
        "staged_players": [
            {"species": "Terran Factions", "is_ai": False},
            {"species": "Planta", "is_ai": True},
            {"species": "Orion Hegemony", "is_ai": True},
            {"species": "Hydran Progress", "is_ai": True},
        ],
    }
    payload = await setup_pre_choice(config=config)
    assert "state" in payload
    assert len(payload["state"]["players"]) == 4
    assert payload["config"]["rng_seed"] == 42
    assert payload["config"]["warped_universe"] is False

    warped_config = {**config, "players": 3, "warped_universe": True}
    warped_config["staged_players"] = config["staged_players"][:3]
    warped_payload = await setup_pre_choice(config=warped_config)
    assert warped_payload["config"]["warped_universe"] is True

    choices = [
        {"species": "Terran Factions", "is_ai": False},
        {"species": "Planta", "is_ai": True},
        {"species": "Orion Hegemony", "is_ai": True},
        {"species": "Hydran Progress", "is_ai": True},
    ]
    finalized_payload = await setup_finalize(snapshot=payload, player_choices=choices)
    assert finalized_payload["state"]["players"][0]["species_id"] == "Terran Factions"
    assert finalized_payload["finalized"] is True

    print("Validated eclipse_ui_native import and FastAPI setup endpoints.")


if __name__ == "__main__":
    asyncio.run(_main())
