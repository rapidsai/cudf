#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

project(cudf-GoogleTest)

include(ExternalProject)

ExternalProject_Add(GoogleTest
                    GIT_REPOSITORY    https://github.com/google/googletest.git
                    GIT_TAG           release-1.8.0
                    GIT_SHALLOW       true
                    SOURCE_DIR        "${GTEST_ROOT}/googletest"
                    BINARY_DIR        "${GTEST_ROOT}/build"
                    INSTALL_DIR       "${GTEST_ROOT}/install"
                    CMAKE_ARGS        ${GTEST_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${GTEST_ROOT}/install)
