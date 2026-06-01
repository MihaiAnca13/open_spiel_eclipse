//
// Created by Mihai on 25/05/2026.
//

#ifndef ECLIPSE_DICE_H
#define ECLIPSE_DICE_H
#include <cstdint>

constexpr uint8_t DIE_DAMAGE[] = {1, 2, 3, 4};

enum class DieColor : uint8_t {
    YELLOW = 0,
    ORANGE = 1,
    BLUE = 2,
    RED = 3,
    PURPLE = 4, // Rift
    NONE = 5
};

struct DieResult {
    uint8_t value; // 1 always miss, 6 always hit
    DieColor color;      // YELLOW, ORANGE, BLUE, RED, or PURPLE
};

struct RiftCannonFace {
    uint8_t damage;
    uint8_t self_damage;
};

constexpr RiftCannonFace RIFT_CANNON_FACES[6] = {
    {0, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 1}, {0, 1}
};

#endif //ECLIPSE_DICE_H
