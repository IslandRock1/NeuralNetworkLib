
set(SHARED_SOURCES
        src/MNISTDigitTester.cpp
)

add_library(MNISTTestLib STATIC ${SHARED_SOURCES})
target_include_directories(MNISTTestLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/Projects)
target_compile_definitions(MNISTTestLib PRIVATE DATA_PATH="${CMAKE_SOURCE_DIR}/Projects/MNIST_Digits/data")