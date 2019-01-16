#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# BEGIN macros

macro(CONFIGURE_BOOST_EXTERNAL_PROJECT)
    set(ENV{CFLAGS} "${CMAKE_C_FLAGS}")
    set(ENV{CXXFLAGS} "${CMAKE_CXX_FLAGS}")

    # Download and unpack Boost at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Boost.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for Boost failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for Boost failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

if (BOOST_INSTALL_DIR)
    message(STATUS "BOOST_INSTALL_DIR defined, it will use vendor version from ${BOOST_INSTALL_DIR}")
    set(BOOST_ROOT "${BOOST_INSTALL_DIR}")
else()
    message(STATUS "BOOST_INSTALL_DIR not defined, it will be built from sources")
    CONFIGURE_BOOST_EXTERNAL_PROJECT()
    set(BOOST_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/boost-install/")
endif()

message(STATUS "BOOST_ROOT: ${BOOST_ROOT}")

set(Boost_USE_STATIC_LIBS ON) # only find static libs
find_package(Boost REQUIRED COMPONENTS regex system filesystem)
set_package_properties(Boost PROPERTIES TYPE REQUIRED
    PURPOSE " Boost."
    URL "https://Boost. org")

set(BOOST_INCLUDEDIR ${BOOST_ROOT}/include/)

include_directories(${BOOST_INCLUDEDIR} ${BOOST_INCLUDE_DIR})
link_directories(${BOOST_ROOT}/lib/)

# END MAIN #
