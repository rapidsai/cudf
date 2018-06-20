# Download and unpack arrow at configure time
configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Arrow.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/CMakeLists.txt COPYONLY)

execute_process(
    COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/
)

if(result)
    message(FATAL_ERROR "CMake step for arrow failed: ${result}")
endif()

execute_process(
    COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/
)

if(result)
    message(FATAL_ERROR "Build step for arrow failed: ${result}")
endif()

# Locate the Arrow package.
# Requires that you build with:
#   -DARROW_ROOT:PATH=/path/to/arrow_install_dir
set(ARROW_ROOT ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/arrow-download/arrow-prefix/src/arrow-install/usr/local/)
message(STATUS "ARROW_ROOT: " ${ARROW_ROOT})
