set(GTEST_ROOT "${CMAKE_BINARY_DIR}/googletest")

set(GTEST_CMAKE_ARGS " -Dgtest_build_samples=ON" 
                     " -DCMAKE_VERBOSE_MAKEFILE=ON"
                     " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"      # enable old ABI for C/C++
                     " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")   # enable old ABI for C/C++


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

# Parallel builds cause Travis to run out of memory
unset(PARALLEL_CMAKE_BUILD)            
if(DEFINED ENV{TRAVIS})
    set(PARALLEL_CMAKE_BUILD --parallel [$ENV{CPU_COUNT}])
    message("Enabling Parallel CMake build on Travis with $ENV{CPU_COUNT} CPUs")
else()
    set(PARALLEL_CMAKE_BUILD --parallel)
    message("Enabling Parallel CMake build")
endif(DEFINED ENV{TRAVIS})

execute_process(COMMAND ${CMAKE_COMMAND} --build ${PARALLEL_CMAKE_BUILD} ..
                RESULT_VARIABLE GTEST_BUILD
                WORKING_DIRECTORY ${GTEST_ROOT}/build)

if(GTEST_BUILD)
    message(FATAL_ERROR "Building GoogleTest failed: " ${GTEST_BUILD})
endif(GTEST_BUILD)

message(STATUS "GoogleTest installed here: " ${GTEST_ROOT}/install)
set(GTEST_INCLUDE_DIR "${GTEST_ROOT}/install/include")
set(GTEST_LIBRARY_DIR "${GTEST_ROOT}/install/lib")
set(GTEST_FOUND TRUE)

