add_library(eclipse_game OBJECT
  eclipse.cc
  observation.cpp
  systems/setup.cpp
  systems/combat.cpp
  systems/upkeep.cpp
  systems/scoring.cpp
  systems/actions/bonus.cpp
  systems/actions/explore.cpp
  systems/actions/research.cpp
  systems/actions/build.cpp
  systems/actions/diplomacy.cpp
  systems/actions/minor_species_diplomacy.cpp
  systems/actions/influence.cpp
  systems/actions/upgrade.cpp
  systems/actions/move.cpp
  warped_universe/warped_universe.cpp
)

target_include_directories(eclipse_game PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
target_link_libraries(eclipse_game PUBLIC open_spiel_core)
