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

macro(CONFIGURE_LZ4_EXTERNAL_PROJECT)
    # NOTE percy c.gonzales if you want to pass other RAL CMAKE_CXX_FLAGS into this dependency add it by harcoding
    set(ENV{CFLAGS} "-D_GLIBCXX_USE_CXX11_ABI=0 -O3 -fPIC")
    set(ENV{CXXFLAGS} "-D_GLIBCXX_USE_CXX11_ABI=0 -O3 -fPIC")
    set(ENV{PREFIX} "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/lz4-install")

    # Download and unpack Lz4 at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Lz4.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/lz4-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/lz4-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for Lz4 failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/lz4-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for Lz4 failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

# NOTE since parquet and Lz4 are in the same repo is safe to pass the lz4 installation dir here
set(LZ4_INSTALL_DIR ${LZ4_INSTALL_DIR})

if (LZ4_INSTALL_DIR)
    message(STATUS "LZ4_INSTALL_DIR defined, it will use vendor version from ${LZ4_INSTALL_DIR}")
    set(LZ4_ROOT "${LZ4_INSTALL_DIR}")
else()
    message(STATUS "LZ4_INSTALL_DIR not defined, it will be built from sources")
    CONFIGURE_LZ4_EXTERNAL_PROJECT()
    set(LZ4_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/lz4-install/")
endif()

set(LZ4_HOME ${LZ4_ROOT})
find_package(Lz4 REQUIRED)
set_package_properties(Lz4 PROPERTIES TYPE REQUIRED
    PURPOSE "Apache Lz4."
    URL "https://Lz4.apache.org")

set(LZ4_INCLUDEDIR ${LZ4_ROOT}/include/)

include_directories(${LZ4_INCLUDEDIR} ${LZ4_INCLUDE_DIR})
link_directories(${LZ4_ROOT}/lib/)

# END MAIN #
