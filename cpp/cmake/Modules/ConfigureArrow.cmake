set(ARROW_ROOT ${CMAKE_BINARY_DIR}/arrow)

if(NOT CMAKE_CXX11_ABI)
    message(STATUS "ARROW: Disabling the GLIBCXX11 ABI")
    set(ARROW_CMAKE_ARGS " -DARROW_WITH_LZ4=OFF"
                         " -DARROW_WITH_ZSTD=OFF"
                         " -DARROW_WITH_BROTLI=OFF"
                         " -DARROW_WITH_SNAPPY=OFF"
                         " -DARROW_WITH_ZLIB=OFF"
                         " -DARROW_BUILD_STATIC=ON"
                         " -DARROW_BUILD_SHARED=OFF"
                         " -DARROW_BOOST_USE_SHARED=ON"
                         " -DARROW_BUILD_TESTS=OFF"
                         " -DARROW_TEST_MEMCHECK=OFF"
                         " -DARROW_BUILD_BENCHMARKS=OFF"
                         " -DARROW_IPC=ON"
                         " -DARROW_COMPUTE=OFF"
                         " -DARROW_GPU=OFF"
                         " -DARROW_JEMALLOC=OFF"
                         " -DARROW_BOOST_VENDORED=OFF"
                         " -DARROW_PYTHON=OFF"
                         " -DARROW_TENSORFLOW=ON") # enable old ABI for C/C++
elseif(CMAKE_CXX11_ABI)
    set(ARROW_CMAKE_ARGS " -DARROW_WITH_LZ4=OFF"
                         " -DARROW_WITH_ZSTD=OFF"
                         " -DARROW_WITH_BROTLI=OFF"
                         " -DARROW_WITH_SNAPPY=OFF"
                         " -DARROW_WITH_ZLIB=OFF"
                         " -DARROW_BUILD_STATIC=ON"
                         " -DARROW_BUILD_SHARED=OFF"
                         " -DARROW_BOOST_USE_SHARED=ON"
                         " -DARROW_BUILD_TESTS=OFF"
                         " -DARROW_TEST_MEMCHECK=OFF"
                         " -DARROW_BUILD_BENCHMARKS=OFF"
                         " -DARROW_IPC=ON"
                         " -DARROW_COMPUTE=OFF"
                         " -DARROW_GPU=OFF"
                         " -DARROW_JEMALLOC=OFF"
                         " -DARROW_BOOST_VENDORED=OFF"
                         " -DARROW_PYTHON=OFF"
                         " -DARROW_TENSORFLOW=OFF") # enable old ABI for C/C++
endif(NOT CMAKE_CXX11_ABI)

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
unset(PARALLEL_BUILD)            
if($ENV{TRAVIS})
    if(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        message(STATUS "ARROW BUILD: Disabling Parallel CMake build on Travis")
    else()
        set(PARALLEL_BUILD --parallel)
        message(STATUS "ARROW BUILD: Using $ENV{CMAKE_BUILD_PARALLEL_LEVEL} build jobs on Travis")
    endif(NOT DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
else()
    set(PARALLEL_BUILD --parallel)
    message(STATUS "ARROW BUILD: Enabling Parallel CMake build")
endif($ENV{TRAVIS})

execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${PARALLEL_BUILD} ..
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

find_library(ARROW_LIB arrow
             NO_DEFAULT_PATH
             HINTS "${ARROW_LIBRARY_DIR}")

if(ARROW_LIB)
    message(STATUS "Arrow library: " ${ARROW_LIB})
    set(ARROW_FOUND TRUE)
endif(ARROW_LIB)

set(FLATBUFFERS_ROOT "${ARROW_ROOT}/build/flatbuffers_ep-prefix/src/flatbuffers_ep-install")

message(STATUS "FlatBuffers installed here: " ${FLATBUFFERS_ROOT})
set(FLATBUFFERS_INCLUDE_DIR "${FLATBUFFERS_ROOT}/include")
set(FLATBUFFERS_LIBRARY_DIR "${FLATBUFFERS_ROOT}/lib")

add_definitions(-DARROW_METADATA_V4)
add_definitions(-DARROW_VERSION=1000)






