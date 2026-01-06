
set(SHARED_SOURCES
        src/DoublePendulumSimulation.cpp
        src/DoublePendulumTester.cpp
)

add_library(DoublePendulumTestLib STATIC ${SHARED_SOURCES})
target_include_directories(DoublePendulumTestLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/Projects)