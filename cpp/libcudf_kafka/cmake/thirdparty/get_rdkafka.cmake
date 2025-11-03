# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
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
