set(GTEST_ROOT "${CMAKE_BINARY_DIR}/googletest")

set(GTEST_CMAKE_ARGS "")

configure_file("${CMAKE_SOURCE_DIR}/cmake/Templates/GoogleTest.CMakeLists.txt.cmake"
               "${GTEST_ROOT}/CMakeLists.txt")

file(MAKE_DIRECTORY "${GTEST_ROOT}/build")
file(MAKE_DIRECTORY "${GTEST_ROOT}/install")

execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
                RESULT_VARIABLE GTEST_CONFIG
                WORKING_DIRECTORY ${GTEST_ROOT})

if(GTEST_CONFIG)
    message(FATAL_ERROR "Configuring GoogleTest failed: " ${GTEST_CONFIG})
endif(GTEST_CONFIG)

set(PARALLEL_BUILD -j)
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "${PARALLEL_BUILD}${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(VERBOSE "GTEST BUILD: Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(VERBOSE "GTEST BUILD: Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(VERBOSE "GTEST BUILD: Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

execute_process(COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
                RESULT_VARIABLE GTEST_BUILD
                WORKING_DIRECTORY ${GTEST_ROOT}/build)

if(GTEST_BUILD)
    message(FATAL_ERROR "Building GoogleTest failed: " ${GTEST_BUILD})
endif(GTEST_BUILD)

message(VERBOSE "GoogleTest installed here: " ${GTEST_ROOT}/install)
set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/install/include")
set(GTEST_LIBRARY_DIR "${GTEST_ROOT}/install/lib")
set(GTEST_FOUND TRUE)
