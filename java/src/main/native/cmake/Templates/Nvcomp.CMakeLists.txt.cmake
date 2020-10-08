#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.12)

project(nvcomp)

include(ExternalProject)

ExternalProject_Add(nvcomp
    GIT_REPOSITORY  https://github.com/NVIDIA/nvcomp.git
    GIT_TAG         v1.1.0
    GIT_SHALLOW     true
    SOURCE_DIR      "${NVCOMP_ROOT}/nvcomp"
    BINARY_DIR      "${NVCOMP_ROOT}/build"
    INSTALL_DIR     "${NVCOMP_ROOT}/install"
    CMAKE_ARGS      ${NVCOMP_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${NVCOMP_ROOT}/install
    BUILD_COMMAND   ${CMAKE_COMMAND} --build . --target nvcomp
    INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Skipping nvcomp install step.")
