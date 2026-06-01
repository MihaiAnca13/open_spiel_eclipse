from __future__ import annotations

import asyncio

from api.main import setup_finalize, setup_pre_choice


async def _main() -> None:
    payload = await setup_pre_choice(seed=42, num_players=4, difficulty="Easy")
    assert "players" in payload
    assert len(payload["players"]) == 4

    choices = [
        {"species": "Terran Factions", "is_ai": False},
        {"species": "Planta", "is_ai": True},
        {"species": "Orion Hegemony", "is_ai": True},
        {"species": "Hydran Progress", "is_ai": True},
    ]
    finalized_payload = await setup_finalize(state=payload, player_choices=choices)
    assert finalized_payload["players"][0]["species_id"] == "Terran Factions"

    print("Validated eclipse_ui_native import and FastAPI setup endpoints.")


if __name__ == "__main__":
    asyncio.run(_main())
