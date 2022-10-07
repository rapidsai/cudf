# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

project(
  IDENTIFY_STREAM_USAGE
  VERSION 0.0.1
  LANGUAGES CXX
)

set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
add_library(identify_stream_usage SHARED identify_stream_usage.cpp)

set_target_properties(
  identify_stream_usage
  PROPERTIES # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
)
