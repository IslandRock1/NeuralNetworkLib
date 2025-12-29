
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG v33.2
)
FetchContent_MakeAvailable(protobuf)

set(generatedDir "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generatedDir}")

set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/NeuralNetwork.proto")
set(GENERATED_SRC "${generatedDir}/NeuralNetwork.pb.cc")
set(GENERATED_HDR "${generatedDir}/NeuralNetwork.pb.h")

add_custom_command(
        OUTPUT ${GENERATED_SRC} ${GENERATED_HDR}
        COMMAND $<TARGET_FILE:protobuf::protoc>
        --cpp_out=${generatedDir}
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROTO_FILE}
        DEPENDS ${PROTO_FILE} protobuf::protoc
        COMMENT "Generating protobuf sources"
)

add_library(neuralnetwork_msg STATIC ${GENERATED_SRC})
target_include_directories(neuralnetwork_msg PUBLIC ${generatedDir})
target_link_libraries(neuralnetwork_msg PUBLIC protobuf::libprotobuf)
