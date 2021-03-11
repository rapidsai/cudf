#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

function(find_and_configure_gtest VERSION)
    # Find or install GoogleTest
    CPMFindPackage(NAME GTest
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/google/googletest.git
        GIT_TAG         release-${VERSION}
        GIT_SHALLOW     TRUE
        OPTIONS         "INSTALL_GTEST OFF"
        # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
        FIND_PACKAGE_ARGUMENTS "CONFIG")
    # Add GTest aliases if they don't already exist.
    # Assumes if GTest::gtest doesn't exist, the others don't either.
    # TODO: Is this always a valid assumption?
    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
        add_library(GTest::gmock ALIAS gmock)
        add_library(GTest::gtest_main ALIAS gtest_main)
        add_library(GTest::gmock_main ALIAS gmock_main)
    endif()
    # Make sure consumers of cudf can also see GTest::* targets
    fix_cmake_global_defaults(GTest::gtest)
    fix_cmake_global_defaults(GTest::gmock)
    fix_cmake_global_defaults(GTest::gtest_main)
    fix_cmake_global_defaults(GTest::gmock_main)
    if(GTest_ADDED)
        install(TARGETS gmock
                        gtest
                        gmock_main
                        gtest_main
            DESTINATION lib
            EXPORT cudf-targets)
    endif()
endfunction()

set(CUDF_MIN_VERSION_gtest 1.10.0)

find_and_configure_gtest(${CUDF_MIN_VERSION_gtest})
