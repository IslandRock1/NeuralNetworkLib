cmake_minimum_required(VERSION 4.0)
project(NeuralNetworkLib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(Projects/Pendulum)
add_subdirectory(Projects/DoublePendulum)
add_subdirectory(Projects/MNIST_Digits)
add_subdirectory(Protobuf)

include(FetchContent)
FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.1
)
FetchContent_MakeAvailable(Eigen3)

set(SHARED_SOURCES
        src/ActivationFunctions.cpp
        src/Node.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/TrainingLib.cpp
        src/NeuralNetworkCuda.cu
)

add_library(NeuralNetworkLib STATIC ${SHARED_SOURCES})
set_target_properties(NeuralNetworkLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(NeuralNetworkLib PUBLIC MNISTTestLib PendulumTestLib DoublePendulumTestLib neuralnetwork_msg Eigen3::Eigen)
target_include_directories(NeuralNetworkLib PUBLIC ${CMAKE_SOURCE_DIR}/include)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11)
if(pybind11_FOUND)
    pybind11_add_module(NeuralNetworkPython src/pybind.cpp)
    target_link_libraries(NeuralNetworkPython PUBLIC NeuralNetworkLib)

    set_target_properties(NeuralNetworkPython PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Testing
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Testing
    )
endif()

add_executable(Test Testing/test.cpp)
set_target_properties(Test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(Test NeuralNetworkLib)

add_executable(TestCuda Testing/testCuda.cpp)
set_target_properties(TestCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(TestCuda NeuralNetworkLib)