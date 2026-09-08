cmake_minimum_required(VERSION 3.17)
project(open_spiel_eclipse)

set(OPEN_SPIEL_ECLIPSE_BUILD ON CACHE BOOL
    "Build only the Eclipse and FCM OpenSpiel integrations")

add_subdirectory(open_spiel)
