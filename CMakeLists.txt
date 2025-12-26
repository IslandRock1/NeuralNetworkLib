cmake_minimum_required(VERSION 4.0)
project(NeuralNetworkLib)

set(CMAKE_CXX_STANDARD 20)

set(SHARED_SOURCES
        src/ActivationFunctions.cpp
        src/Node.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/TrainingLib.cpp
)

add_library(NeuralNetworkLib STATIC ${SHARED_SOURCES})
target_include_directories(NeuralNetworkLib PUBLIC ${CMAKE_SOURCE_DIR}/include)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11)
if(pybind11_FOUND)
    pybind11_add_module(NeuralNetworkPython src/pybind.cpp)
    target_link_libraries(NeuralNetworkPython PUBLIC NeuralNetworkLib)
endif()

add_executable(Test Testing/test.cpp)
target_link_libraries(Test NeuralNetworkLib)