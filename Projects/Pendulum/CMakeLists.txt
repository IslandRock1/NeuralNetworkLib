
set(SHARED_SOURCES
        src/PendulumSimulation.cpp
        src/PendulumTester.cpp
)

add_library(PendulumTestLib STATIC ${SHARED_SOURCES})
target_include_directories(PendulumTestLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/Projects)