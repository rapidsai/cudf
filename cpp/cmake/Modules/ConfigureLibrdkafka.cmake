set(LIBRDKAFKA_ROOT ${CMAKE_BINARY_DIR}/librdkafka)

configure_file("${CMAKE_SOURCE_DIR}/cmake/Templates/Librdkafka.CMakeLists.txt.cmake"
        "${LIBRDKAFKA_ROOT}/CMakeLists.txt")

file(MAKE_DIRECTORY "${LIBRDKAFKA_ROOT}/build")
file(MAKE_DIRECTORY "${LIBRDKAFKA_ROOT}/install")

execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE LIBRDKAFKA_CONFIG
        WORKING_DIRECTORY ${LIBRDKAFKA_ROOT})

if(LIBRDKAFKA_CONFIG)
    message(FATAL_ERROR "Configuring librdkafka failed: " ${LIBRDKAFKA_CONFIG})
endif(LIBRDKAFKA_CONFIG)

set(PARALLEL_BUILD -j)
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "${PARALLEL_BUILD}${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(STATUS "LIBRDKAFKA BUILD: Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(STATUS "LIBRDKAFKA BUILD: Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(STATUS "LIBRDKAFKA BUILD: Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

execute_process(
        COMMAND ${CMAKE_COMMAND} --build .. -- ${PARALLEL_BUILD}
        RESULT_VARIABLE LIBRDKAFKA_BUILD
        WORKING_DIRECTORY ${LIBRDKAFKA_ROOT}/build)

if(LIBRDKAFKA_BUILD)
    message(FATAL_ERROR "Building librdkafka failed: " ${LIBRDKAFKA_BUILD})
endif(LIBRDKAFKA_BUILD)


message(STATUS "LIBRDKAFKA installed here: " ${LIBRDKAFKA_ROOT}/install)
set(LIBRDKAFKA_LIBRARY_DIR "${LIBRDKAFKA_ROOT}/install/lib")
set(LIBRDKAFKA_INCLUDE_DIR "${LIBRDKAFKA_ROOT}/install/include")

find_library(LIBRDKAFKA_LIB rdkafka
        NO_DEFAULT_PATH
        HINTS "${LIBRDKAFKA_LIBRARY_DIR}")

if(LIBRDKAFKA_LIB)
    message(STATUS "librdkafka library: " ${LIBRDKAFKA_LIB})
    set(LIBRDKAFKA_FOUND TRUE)
endif(LIBRDKAFKA_LIB)

file(INSTALL ${LIBRDKAFKA_INCLUDE_DIR}/librdkafka DESTINATION include/librdkafka)

install(DIRECTORY ${LIBRDKAFKA_INCLUDE_DIR}/librdkafka
        DESTINATION include/librdkafka)

