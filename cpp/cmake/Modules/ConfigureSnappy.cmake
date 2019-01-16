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

macro(CONFIGURE_SNAPPY_EXTERNAL_PROJECT)
    # NOTE percy c.gonzales if you want to pass other RAL CMAKE_CXX_FLAGS into this dependency add it by harcoding
    set(ENV{CFLAGS} "-D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -O2")
    set(ENV{CXXFLAGS} "-D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -O2")

    # Download and unpack Snappy at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Snappy.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/snappy-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/snappy-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for Snappy failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/snappy-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for Snappy failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

if (SNAPPY_INSTALL_DIR)
    message(STATUS "SNAPPY_INSTALL_DIR defined, it will use vendor version from ${SNAPPY_INSTALL_DIR}")
    set(SNAPPY_ROOT "${SNAPPY_INSTALL_DIR}")
else()
    message(STATUS "SNAPPY_INSTALL_DIR not defined, it will be built from sources")
    CONFIGURE_SNAPPY_EXTERNAL_PROJECT()
    set(SNAPPY_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/snappy-install/")
endif()

set(SNAPPY_HOME ${SNAPPY_ROOT})
find_package(Snappy REQUIRED)
set_package_properties(Snappy PROPERTIES TYPE REQUIRED
    PURPOSE " Snappy."
    URL "https://Snappy. org")

set(SNAPPY_INCLUDEDIR ${SNAPPY_ROOT}/include/)

include_directories(${SNAPPY_INCLUDEDIR} ${SNAPPY_INCLUDE_DIR})
link_directories(${SNAPPY_ROOT}/lib/)

# END MAIN #
