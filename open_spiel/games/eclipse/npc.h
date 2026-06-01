//
// Created by Mihai on 28/05/2026.
//

#ifndef ECLIPSE_NPC_H
#define ECLIPSE_NPC_H
#include "dice.h"
#include <cstdint>
#include <nlohmann/json.hpp>

enum class NPCType { GCDS, GUARDIAN, ANCIENT };

enum class NPCDifficulty {EASY, MEDIUM, HARD};

NLOHMANN_JSON_SERIALIZE_ENUM(NPCDifficulty, {
    {NPCDifficulty::EASY, "Easy"},
    {NPCDifficulty::MEDIUM, "Medium"},
    {NPCDifficulty::HARD, "Hard"}
});

struct NPC {
    NPCType type;
    NPCDifficulty difficulty;
    DieColor cannon, missile;
    uint8_t cannon_amount, missile_amount;
    uint8_t computer;
    uint8_t shield;
    uint8_t hull;
    uint8_t initiative;
};

static const NPC NPC_TABLE[] = {
    { NPCType::GCDS, NPCDifficulty::EASY, DieColor::YELLOW, DieColor::NONE, 4, 0, 2, 0 , 1, 0},
    { NPCType::GCDS, NPCDifficulty::MEDIUM, DieColor::RED, DieColor::YELLOW, 1, 4, 2, 0, 3, 2 },
    { NPCType::GCDS, NPCDifficulty::HARD, DieColor::ORANGE, DieColor::NONE, 2, 0, 2, 2, 4, 3 },
    { NPCType::GUARDIAN, NPCDifficulty::EASY, DieColor::YELLOW, DieColor::NONE, 3, 0, 2, 0, 2, 3 },
    { NPCType::GUARDIAN, NPCDifficulty::MEDIUM, DieColor::ORANGE, DieColor::NONE, 2, 0, 1, 1, 3, 2 },
    { NPCType::GUARDIAN, NPCDifficulty::HARD, DieColor::RED, DieColor::ORANGE, 1, 2, 1, 0, 3, 1 },
    { NPCType::ANCIENT, NPCDifficulty::EASY, DieColor::YELLOW, DieColor::NONE, 2, 0, 1, 0, 1, 2 },
    { NPCType::ANCIENT, NPCDifficulty::MEDIUM, DieColor::YELLOW, DieColor::NONE, 1, 0, 2, 0, 1, 3 },
    { NPCType::ANCIENT, NPCDifficulty::HARD, DieColor::ORANGE, DieColor::NONE, 1, 0, 1, 0, 2, 1 }
};

#endif //ECLIPSE_NPC_H
