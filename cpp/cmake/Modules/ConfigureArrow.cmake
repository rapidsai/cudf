set(ARROW_ROOT ${CMAKE_BINARY_DIR}/arrow)

configure_file("${CMAKE_SOURCE_DIR}/cmake/Templates/Arrow.CMakeLists.txt.cmake"
               "${ARROW_ROOT}/CMakeLists.txt")

file(MAKE_DIRECTORY "${ARROW_ROOT}/build")
file(MAKE_DIRECTORY "${ARROW_ROOT}/install")

execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE ARROW_CONFIG
    WORKING_DIRECTORY ${ARROW_ROOT})

if(ARROW_CONFIG)
    message(FATAL_ERROR "Configuring Arrow failed: " ${ARROW_CONFIG})
endif(ARROW_CONFIG)

# Parallel builds cause Travis to run out of memory
unset(PARALLEL_CMAKE_BUILD)            
if (NOT ENV{TRAVIS})
    set(PARALLEL_CMAKE_BUILD --parallel)
    message("Enabling Parallel CMake build")
else()
    message("Disabling Parallel CMake build on Travis")
endif (NOT ENV{TRAVIS})

execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${PARALLEL_CMAKE_BUILD} ..
    RESULT_VARIABLE ARROW_BUILD
    WORKING_DIRECTORY ${ARROW_ROOT}/build)

if(ARROW_BUILD)
    message(FATAL_ERROR "Building Arrow failed: " ${ARROW_BUILD})
endif(ARROW_BUILD)

set(ARROW_GENERATED_IPC_DIR 
    "${ARROW_ROOT}/build/src/arrow/ipc")

configure_file(${ARROW_GENERATED_IPC_DIR}/File_generated.h ${CMAKE_SOURCE_DIR}/include/cudf/ipc_generated/File_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Message_generated.h ${CMAKE_SOURCE_DIR}/include/cudf/ipc_generated/Message_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Schema_generated.h ${CMAKE_SOURCE_DIR}/include/cudf/ipc_generated/Schema_generated.h COPYONLY)
configure_file(${ARROW_GENERATED_IPC_DIR}/Tensor_generated.h ${CMAKE_SOURCE_DIR}/include/cudf/ipc_generated/Tensor_generated.h COPYONLY)

message(STATUS "Arrow installed here: " ${ARROW_ROOT}/install)
set(ARROW_LIBRARY_DIR "${ARROW_ROOT}/install/lib")
set(ARROW_INCLUDE_DIR "${ARROW_ROOT}/install/include")

set(FLATBUFFERS_ROOT "${ARROW_ROOT}/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install")

message(STATUS "FlatBuffers installed here: " ${FLATBUFFERS_ROOT})
set(FLATBUFFERS_INCLUDE_DIR "${FLATBUFFERS_ROOT}/include")
set(FLATBUFFERS_LIBRARY_DIR "${FLATBUFFERS_ROOT}/lib")

add_definitions(-DARROW_METADATA_V4)
add_definitions(-DARROW_VERSION=1000)



