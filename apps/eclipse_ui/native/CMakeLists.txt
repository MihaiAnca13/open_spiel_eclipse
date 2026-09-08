# This directory is added by open_spiel/CMakeLists.txt, which is also where the
# module picks up open_spiel_core and absl (via the directory-scoped
# link_libraries() there). Configuring it as its own top-level project instead
# links a module full of undefined open_spiel::* symbols, which only surfaces as
# an ImportError when Python dlopens it. Fail at configure time instead.
if (NOT TARGET open_spiel_core)
  message(FATAL_ERROR
    "eclipse_ui_native must be configured as part of the OpenSpiel build. "
    "Build it from the repo-root build tree:\n"
    "  cmake --build build --target eclipse_ui_native")
endif()

add_library(eclipse_ui_native MODULE api.cpp)
target_link_libraries(eclipse_ui_native PRIVATE eclipse_game)

find_package(Python3 COMPONENTS Interpreter Development Development.Module REQUIRED)

target_include_directories(eclipse_ui_native PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../open_spiel/abseil-cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../pybind11/include
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
