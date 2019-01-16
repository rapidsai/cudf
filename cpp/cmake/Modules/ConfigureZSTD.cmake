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

macro(CONFIGURE_ZSTD_EXTERNAL_PROJECT)
    # NOTE percy c.gonzales if you want to pass other RAL CMAKE_CXX_FLAGS into this dependency add it by harcoding
    # NOTE build with CMAKE_POSITION_INDEPENDENT_CODE (akka -fPIC)
    set(ZSTD_CMAKE_ARGS
        " -DZSTD_BUILD_STATIC=ON"
        " -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
        " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
        " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
    )

    # Download and unpack ZSTD at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/ZSTD.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/zstd-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/zstd-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for ZSTD failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/zstd-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for ZSTD failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

# NOTE since parquet and ZSTD are in the same repo is safe to pass the zstd installation dir here
set(ZSTD_INSTALL_DIR ${ZSTD_INSTALL_DIR})

if (ZSTD_INSTALL_DIR)
    message(STATUS "ZSTD_INSTALL_DIR defined, it will use vendor version from ${ZSTD_INSTALL_DIR}")
    set(ZSTD_ROOT "${ZSTD_INSTALL_DIR}")
else()
    message(STATUS "ZSTD_INSTALL_DIR not defined, it will be built from sources")
    CONFIGURE_ZSTD_EXTERNAL_PROJECT()
    set(ZSTD_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/zstd-install/")
endif()

set(ZSTD_HOME ${ZSTD_ROOT})
find_package(ZSTD REQUIRED)
set_package_properties(ZSTD PROPERTIES TYPE REQUIRED
    PURPOSE "Apache ZSTD."
    URL "https://ZSTD.apache.org")

set(ZSTD_INCLUDEDIR ${ZSTD_ROOT}/include/)

include_directories(${ZSTD_INCLUDEDIR} ${ZSTD_INCLUDE_DIR})
link_directories(${ZSTD_ROOT}/lib/)

# END MAIN #
