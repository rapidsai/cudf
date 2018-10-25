#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
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

#pass the dependency libraries as optional arguments using ${ARGN}
#NOTE the order of libraries matter, so try to link first with the most high level lib
function(configure_test TEST_NAME Tests_SRCS)
    message(STATUS "${TEST_NAME} will link against: gdf arrow")

    cuda_add_executable(${TEST_NAME} ${Tests_SRCS})
    target_link_libraries(${TEST_NAME} gmock_main gmock GTest::GTest gdf arrow)
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

    #register the target as CMake test so we can use ctest for this executable
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

## Main ##

enable_testing()

# Configure unit tests with Google Test
# -------------------
message(STATUS "******** Configuring tests ********")

include_directories(${CMAKE_CURRENT_SOURCE_DIR};
                    ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_subdirectory(foo-sample)
add_subdirectory(datetime)
add_subdirectory(hashing)
add_subdirectory(join)
add_subdirectory(sqls)
add_subdirectory(hash_map)
add_subdirectory(groupby)
add_subdirectory(unaryops)
add_subdirectory(filterops_numeric)
add_subdirectory(quantiles)
add_subdirectory(column)
add_subdirectory(validops)
add_subdirectory(memory)
add_subdirectory(gdf)
add_subdirectory(csv)

message(STATUS "******** Tests are ready ********")
