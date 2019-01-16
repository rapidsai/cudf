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

macro(CONFIGURE_THRIFT_EXTERNAL_PROJECT)
    # NOTE percy c.gonzales if you want to pass other RAL CMAKE_CXX_FLAGS into this dependency add it by harcoding
    # NOTE build with CMAKE_POSITION_INDEPENDENT_CODE (akka -fPIC)
    set(THRIFT_CMAKE_ARGS " -DBUILD_SHARED_LIBS=OFF"
                        " -DBUILD_TESTING=OFF"
                        " -DBUILD_EXAMPLES=OFF"
                        " -DBUILD_TUTORIALS=OFF"
                        " -DWITH_QT4=OFF"
                        " -DWITH_C_GLIB=OFF"
                        " -DWITH_JAVA=OFF"
                        " -DWITH_PYTHON=OFF"
                        " -DWITH_HASKELL=OFF"
                        " -DWITH_CPP=ON"
                        " -DWITH_STATIC_LIB=ON"
                        " -DWITH_LIBEVENT=OFF"
                        " -DCMAKE_POSITION_INDEPENDENT_CODE=ON"
                        " -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"      # enable old ABI for C/C++
                        " -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"    # enable old ABI for C/C++
                        " -DBOOST_ROOT=${BOOST_ROOT}")

    # Download and unpack Thrift at configure time
    configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/Thrift.CMakeLists.txt.cmake ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/thrift-download/CMakeLists.txt)

    execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/thrift-download/
    )

    if(result)
        message(FATAL_ERROR "CMake step for Thrift failed: ${result}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . -- -j8
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/thrift-download/
    )

    if(result)
        message(FATAL_ERROR "Build step for Thrift failed: ${result}")
    endif()
endmacro()

# END macros

# BEGIN MAIN #

# NOTE since parquet and Thrift are in the same repo is safe to pass the thrift installation dir here
set(THRIFT_INSTALL_DIR ${Thrift_INSTALL_DIR})

if (THRIFT_INSTALL_DIR)
    message(STATUS "THRIFT_INSTALL_DIR defined, it will use vendor version from ${THRIFT_INSTALL_DIR}")
    set(THRIFT_ROOT "${THRIFT_INSTALL_DIR}")
else()
    message(STATUS "THRIFT_INSTALL_DIR not defined, it will be built from sources")
    CONFIGURE_THRIFT_EXTERNAL_PROJECT()
    set(THRIFT_ROOT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/thirdparty/thrift-install/")
endif()

set(THRIFT_HOME ${THRIFT_ROOT})
find_package(Thrift REQUIRED)
set_package_properties(Thrift PROPERTIES TYPE REQUIRED
    PURPOSE "Apache Thrift."
    URL "https://Thrift.apache.org")

set(THRIFT_INCLUDEDIR ${THRIFT_ROOT}/include/)

include_directories(${THRIFT_INCLUDEDIR} ${THRIFT_INCLUDE_DIR})
link_directories(${THRIFT_ROOT}/lib/)

# END MAIN #
