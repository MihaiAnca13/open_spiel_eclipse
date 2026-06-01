//
// Created by Mihai on 24/05/2026.
//

#ifndef ECLIPSE_ACTIONS_H
#define ECLIPSE_ACTIONS_H
#include <cstdint>
#include <nlohmann/json.hpp>

struct ActionActivations {
    uint8_t explore;
    uint8_t research;
    uint8_t upgrade;
    uint8_t build;
    uint8_t move;
    uint8_t influence;
};


enum class ActionType : uint8_t {
    EXPLORE, RESEARCH, UPGRADE, BUILD, MOVE, INFLUENCE, PASS, REACTION
};

NLOHMANN_JSON_SERIALIZE_ENUM( ActionType, {
    {ActionType::EXPLORE, "EXPLORE"},
    {ActionType::RESEARCH, "RESEARCH"},
    {ActionType::UPGRADE, "UPGRADE"},
    {ActionType::BUILD, "BUILD"},
    {ActionType::MOVE, "MOVE"},
    {ActionType::INFLUENCE, "INFLUENCE"},
    {ActionType::PASS, "PASS"},
    {ActionType::REACTION, "REACTION"}
})

#endif //ECLIPSE_ACTIONS_H
