set(NVSTRINGS_ROOT "${CMAKE_BINARY_DIR}/nvstrings")

configure_file("${CMAKE_SOURCE_DIR}/cmake/Templates/NVStrings.CMakeLists.txt.cmake"
               "${NVSTRINGS_ROOT}/CMakeLists.txt")

execute_process(COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
                RESULT_VARIABLE NVSTRINGS_CONFIG
                WORKING_DIRECTORY ${NVSTRINGS_ROOT})

if(NVSTRINGS_CONFIG)
    message(FATAL_ERROR "Configuring NVStrings failed: " ${NVSTRINGS_CONFIG})
endif(NVSTRINGS_CONFIG)

set(PARALLEL_BUILD -j)
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "${PARALLEL_BUILD}${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(STATUS "NVSTRINGS BUILD: Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(STATUS "NVSTRINGS BUILD: Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(STATUS "NVSTRINGS BUILD: Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

execute_process(COMMAND ${CMAKE_COMMAND} --build . -- ${PARALLEL_BUILD}
                RESULT_VARIABLE NVSTRINGS_BUILD
                WORKING_DIRECTORY ${NVSTRINGS_ROOT})

if(NVSTRINGS_BUILD)
    message(FATAL_ERROR "Building NVStrings failed: " ${NVSTRINGS_BUILD})
endif(NVSTRINGS_BUILD)

message(STATUS "NVStrings built here: " ${NVSTRINGS_ROOT})
set(NVSTRINGS_LIBRARY_DIR "${NVSTRINGS_ROOT}")
set(NVSTRINGS_BUILT TRUE)
