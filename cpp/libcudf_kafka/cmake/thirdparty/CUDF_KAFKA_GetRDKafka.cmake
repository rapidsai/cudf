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

find_path(RDKAFKA_INCLUDE "librdkafka" HINTS "$ENV{RDKAFKA_ROOT}/include")
find_library(RDKAFKA++_LIBRARY "rdkafka++" HINTS "$ENV{RDKAFKA_ROOT}/lib" "$ENV{RDKAFKA_ROOT}/build")

if(RDKAFKA_INCLUDE AND RDKAFKA++_LIBRARY)
  add_library(rdkafka INTERFACE)
  target_link_libraries(rdkafka INTERFACE "${RDKAFKA++_LIBRARY}")
  target_include_directories(rdkafka INTERFACE "${RDKAFKA_INCLUDE}")
  add_library(RDKAFKA::RDKAFKA ALIAS rdkafka)
endif()