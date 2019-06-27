set(GBENCH_ROOT "${CMAKE_BINARY_DIR}/googlebenchmark")

set(GBENCH_CMAKE_ARGS " -DCMAKE_BUILD_TYPE=Release")
                     #" -Dgtest_build_samples=ON" 
                     #" -DCMAKE_VERBOSE_MAKEFILE=ON")

if(NOT CMAKE_CXX11_ABI)
    message(STATUS "GBENCH: Disabling the GLIBCXX11 ABI")
    list(APPEND GBENCH_CMAKE_ARGS " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
    list(APPEND GBENCH_CMAKE_ARGS " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
elseif(CMAKE_CXX11_ABI)
    message(STATUS "GBENCH: Enabling the GLIBCXX11 ABI")
    list(APPEND GBENCH_CMAKE_ARGS " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1")
    list(APPEND GBENCH_CMAKE_ARGS " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1")
endif(NOT CMAKE_CXX11_ABI)

configure_file("${CMAKE_SOURCE_DIR}/cmake/Templates/GoogleBenchmark.CMakeLists.txt.cmake"
               "${GBENCH_ROOT}/CMakeLists.txt")

file(MAKE_DIRECTORY "${GBENCH_ROOT}/build")
file(MAKE_DIRECTORY "${GBENCH_ROOT}/install")

execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
                RESULT_VARIABLE GBENCH_CONFIG
                WORKING_DIRECTORY ${GBENCH_ROOT})

if(GBENCH_CONFIG)
    message(FATAL_ERROR "Configuring Google Benchmark failed: " ${GBENCH_CONFIG})
endif(GBENCH_CONFIG)

set(PARALLEL_BUILD -j)
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "${PARALLEL_BUILD}${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(STATUS "GBENCH BUILD: Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(STATUS "GBENCH BUILD: Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(STATUS "GBENCH BUILD: Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

execute_process(COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
                RESULT_VARIABLE GBENCH_BUILD
                WORKING_DIRECTORY ${GBENCH_ROOT}/build)

if(GBENCH_BUILD)
    message(FATAL_ERROR "Building Google Benchmark failed: " ${GBENCH_BUILD})
endif(GBENCH_BUILD)

message(STATUS "Google Benchmark installed here: " ${GBENCH_ROOT}/install)
set(GBENCH_INCLUDE_DIR "${GBENCH_ROOT}/install/include")
set(GBENCH_LIBRARY_DIR "${GBENCH_ROOT}/install/lib" "${GBENCH_ROOT}/install/lib64")
set(GBENCH_FOUND TRUE)

