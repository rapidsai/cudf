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

# Min version set to newest boost in Ubuntu apt repositories
set(CUDF_MIN_VERSION_Boost 1.71.0)

# Don't look for a Boost CMake configuration file because it adds the
# `-DBOOST_ALL_NO_LIB` and `-DBOOST_FILESYSTEM_DYN_LINK` compile defs
set(Boost_NO_BOOST_CMAKE ON)

find_package(Boost ${CUDF_MIN_VERSION_Boost} QUIET MODULE COMPONENTS filesystem)

message(STATUS "Boost_FOUND: ${Boost_FOUND}")

if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found, please check your settings.")
endif()

message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")

list(APPEND CUDF_CXX_DEFINITIONS BOOST_NO_CXX14_CONSTEXPR)
list(APPEND CUDF_CUDA_DEFINITIONS BOOST_NO_CXX14_CONSTEXPR)
