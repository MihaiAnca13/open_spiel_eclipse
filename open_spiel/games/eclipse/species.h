//
// Created by Mihai on 25/05/2026.
//

#ifndef ECLIPSE_SPECIES_H
#define ECLIPSE_SPECIES_H
#include "resources.h"
#include "actions.h"
#include "tech.h"
#include "types.h"

struct SpeciesReputationTrack {
    uint8_t count = 0;
    open_spiel::eclipse::ReputationSlotKind slots[5] = {
        open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
        open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
        open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
        open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
        open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP
    };
};

struct SpeciesData {
    const char* name;
    Resources starting_resources;
    ActionActivations activations;
    int starting_sector;
    int starting_disk_penalty; // Eridani specific
    uint64_t starting_techs;   // Use your TechBit enum
    int trade_rate;            // Standard is 3, Hegemony is 4
    uint8_t starting_colony_ships;
    SpeciesReputationTrack reputation_track;
};

// TODO: verify all info here is correct
static const SpeciesData SPECIES_TABLE[] = {
    // ERIDANI EMPIRE: High money, low disks, strong start
    {
        "Eridani Empire",
        {26, 2, 4},        // Money, Science, Materials
        {1, 1, 2, 2, 2, 2}, // Activations
        222, 2,            // Sector, 2-disk penalty
        static_cast<uint64_t>(TechBit::GAUSS_SHIELD) | static_cast<uint64_t>(TechBit::FUSION_DRIVE) | static_cast<uint64_t>(TechBit::PLASMA_CANNON),
        3, 3,               // Trade Rate, Colony Ships
        {
            4,
            {
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP
            }
        }
    },
    // HYDRAN PROGRESS: High science, extra research activation
    {
        "Hydran Progress",
        {2, 6, 2},         // Starting resources
        {1, 2, 2, 2, 2, 2}, // 2 Research activations
        224, 0,
        static_cast<uint64_t>(TechBit::ADVANCED_LABS),
        3, 3,
        {
            4,
            {
                open_spiel::eclipse::ReputationSlotKind::REP_ONLY,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP
            }
        }
    },
    // PLANTA: High explore activation, weak ships
    {
        "Planta",
        {2, 3, 4},
        {2, 1, 2, 2, 2, 2}, // 2 Explore activations
        226, 0,
        static_cast<uint64_t>(TechBit::STARBASE),
        3, 4,               // 4 colony ships
        {
            4,
            {
                open_spiel::eclipse::ReputationSlotKind::REP_ONLY,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP
            }
        }
    },
    // DESCENDANTS OF DRACO: High explore and movement
    {
        "Descendants of Draco",
        {2, 4, 3},
        {1, 1, 2, 2, 2, 2}, // Note: Special Explore rule is logic-based, not count-based
        228, 0,
        static_cast<uint64_t>(TechBit::FUSION_DRIVE),
        3, 3,
        {
            4,
            {
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP
            }
        }
    },
    // MECHANEMA: Cheap building, high upgrade/build activations
    {
        "Mechanema",
        {3, 3, 4},
        {1, 1, 3, 3, 2, 2}, // 3 Upgrade, 3 Build activations
        230, 0,
        static_cast<uint64_t>(TechBit::POSITRON_COMPUTER),
        3, 3,
        {
            4,
            {
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP
            }
        }
    },
    // ORION HEGEMONY: Strong military, starting Cruiser
    {
        "Orion Hegemony",
        {3, 3, 4},
        {1, 1, 2, 2, 2, 2},
        232, 0,
        static_cast<uint64_t>(TechBit::NEUTRON_BOMBS) | static_cast<uint64_t>(TechBit::GAUSS_SHIELD),
        4, 3,               // Trade Rate 4
        {
            5,
            {
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_ONLY
            }
        }
    },
    // TERRAN FACTIONS: Versatile, trade rate 2
    {
        "Terran Factions",
        {3, 3, 4},
        {1, 1, 2, 2, 2, 2},
        221, 0,            // Example Starting Sector (varies by faction)
        static_cast<uint64_t>(TechBit::STARBASE),
        2, 3,               // Trade Rate 2
        {
            5,
            {
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_ONLY,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP,
                open_spiel::eclipse::ReputationSlotKind::AMBASSADOR_OR_REP
            }
        }
    }
};



#endif //ECLIPSE_SPECIES_H
