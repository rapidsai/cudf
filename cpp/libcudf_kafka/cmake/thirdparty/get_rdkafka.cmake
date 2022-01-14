# =============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

# This function finds rdkafka and sets any additional necessary environment variables.
function(get_RDKafka)
  rapids_find_generate_module(
    RDKAFKA
    HEADER_NAMES rdkafkacpp.h
    INCLUDE_SUFFIXES librdkafka
    LIBRARY_NAMES rdkafka++
    BUILD_EXPORT_SET cudf_kafka-exports
    INSTALL_EXPORT_SET cudf_kafka-exports
  )

  if(DEFINED ENV{RDKAFKA_ROOT})
    # Since this is inside a function the modification of CMAKE_PREFIX_PATH won't leak to other
    # callers/users
    list(APPEND CMAKE_PREFIX_PATH "$ENV{RDKAFKA_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{RDKAFKA_ROOT}/build")
  endif()

  rapids_find_package(
    RDKAFKA REQUIRED
    BUILD_EXPORT_SET cudf_kafka-exports
    INSTALL_EXPORT_SET cudf_kafka-exports
  )

endfunction()

get_RDKafka()
