
set(SHARED_SOURCES
        src/PendulumSimulation.cpp
        src/Tester.cpp
)

add_library(PendulumTestLib STATIC ${SHARED_SOURCES})
target_include_directories(PendulumTestLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)