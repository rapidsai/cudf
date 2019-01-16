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

macro(CONFIGURE_BROTLI_EXTERNAL_PROJECT)
    # NOTE percy c.gonzales if you want to pass other RAL CMAKE_CXX_FLAGS into this dependency add it by harcoding
    set(BROTLI_CMAKE_ARGS " -DBUILD_SHARED_LIBS=OFF"
                          " -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
                          " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
                          " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")

    # Download and unpack Brotli at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Brotli.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/brotli-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/brotli-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for Brotli failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/brotli-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for Brotli failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

if (BROTLI_INSTALL_DIR)
    message(STATUS "BROTLI_INSTALL_DIR defined, it will use vendor version from ${BROTLI_INSTALL_DIR}")
    set(BROTLI_ROOT "${BROTLI_INSTALL_DIR}")
else()
    message(STATUS "BROTLI_INSTALL_DIR not defined, it will be built from sources")
    CONFIGURE_BROTLI_EXTERNAL_PROJECT()
    set(BROTLI_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/brotli-install/")
endif()

set(BROTLI_HOME ${BROTLI_ROOT})
find_package(Brotli REQUIRED)
set_package_properties(Brotli PROPERTIES TYPE REQUIRED
    PURPOSE " Brotli."
    URL "https://Brotli. org")

set(BROTLI_INCLUDEDIR ${BROTLI_ROOT}/include/)

include_directories(${BROTLI_INCLUDEDIR} ${BROTLI_INCLUDE_DIR})
link_directories(${BROTLI_ROOT}/lib/)

# END MAIN #
