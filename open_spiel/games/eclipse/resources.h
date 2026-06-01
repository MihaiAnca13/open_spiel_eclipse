//
// Created by Mihai on 28/05/2026.
//

#ifndef ECLIPSE_RESOURCES_H
#define ECLIPSE_RESOURCES_H

#include <cstdint>
#include <nlohmann/json.hpp>

struct Resources {
    uint8_t gold, science, materials;
    uint8_t gold_prod, science_prod, materials_prod;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Resources, gold, science, materials, gold_prod, science_prod, materials_prod);

enum class Species : uint8_t {
    ERIDANI_EMPIRE = 0,
    HYDRAN_PROGRESS = 1,
    PLANTA = 2,
    DESCENDANTS_OF_DRACO = 3,
    MECHANEMA = 4,
    ORION_HEGEMONY = 5,
    TERRAN_FACTIONS = 6
};

NLOHMANN_JSON_SERIALIZE_ENUM(Species, {
    {Species::ERIDANI_EMPIRE, "Eridani Empire"},
    {Species::HYDRAN_PROGRESS, "Hydran Progress"},
    {Species::PLANTA, "Planta"},
    {Species::DESCENDANTS_OF_DRACO, "Descendants of Draco"},
    {Species::MECHANEMA, "Mechanema"},
    {Species::ORION_HEGEMONY, "Orion Hegemony"},
    {Species::TERRAN_FACTIONS, "Terran Factions"}
});

#endif //ECLIPSE_RESOURCES_H
