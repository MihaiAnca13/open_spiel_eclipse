set(ECLIPSE_UI_NATIVE_SOURCES
  api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/eclipse.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/setup.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/combat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/actions/bonus.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/actions/explore.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/actions/research.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/actions/build.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/actions/influence.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/actions/upgrade.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/games/eclipse/systems/actions/move.cpp
)

add_library(eclipse_ui_native MODULE ${ECLIPSE_UI_NATIVE_SOURCES})

find_package(Python3 COMPONENTS Interpreter Development Development.Module REQUIRED)

target_include_directories(eclipse_ui_native PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../pybind11/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/json/include
)

target_link_directories(eclipse_ui_native PRIVATE ${Python3_LIBRARY_DIRS})
if (TARGET Python3::Module)
  target_link_libraries(eclipse_ui_native PRIVATE Python3::Module)
endif()

set_target_properties(eclipse_ui_native PROPERTIES
  PREFIX ""
  OUTPUT_NAME "eclipse_ui_native"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../api"
)

if (WIN32)
  set_target_properties(eclipse_ui_native PROPERTIES SUFFIX ".pyd")
endif()
