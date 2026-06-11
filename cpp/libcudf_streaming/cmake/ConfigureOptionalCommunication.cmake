# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

set(CUDF_STREAMING_HAVE_MPI OFF)
set(CUDF_STREAMING_HAVE_UCXX OFF)

if(BUILD_BENCHMARKS OR BUILD_EXAMPLES)
  rapids_find_package(
    MPI QUIET
    BUILD_EXPORT_SET cudf_streaming-exports
    INSTALL_EXPORT_SET cudf_streaming-exports
  )
  if(TARGET MPI::MPI_CXX)
    set(CUDF_STREAMING_HAVE_MPI ON)
  endif()
endif()

if(BUILD_BENCHMARKS)
  rapids_find_package(
    ucxx QUIET
    BUILD_EXPORT_SET cudf_streaming-exports
    INSTALL_EXPORT_SET cudf_streaming-exports
  )
  if(TARGET ucxx::ucxx)
    set(CUDF_STREAMING_HAVE_UCXX ON)
  endif()
endif()

set(CUDF_STREAMING_HAVE_COMM OFF)
if(CUDF_STREAMING_HAVE_MPI OR CUDF_STREAMING_HAVE_UCXX)
  set(CUDF_STREAMING_HAVE_COMM ON)
endif()

add_library(cudf_streaming_optional_communication INTERFACE)
target_compile_definitions(
  cudf_streaming_optional_communication
  INTERFACE $<$<BOOL:${CUDF_STREAMING_HAVE_MPI}>:CUDF_STREAMING_HAVE_MPI>
            $<$<BOOL:${CUDF_STREAMING_HAVE_UCXX}>:CUDF_STREAMING_HAVE_UCXX>
)
if(CUDF_STREAMING_HAVE_MPI)
  target_link_libraries(cudf_streaming_optional_communication INTERFACE MPI::MPI_CXX)
endif()
if(CUDF_STREAMING_HAVE_UCXX)
  target_link_libraries(cudf_streaming_optional_communication INTERFACE ucxx::ucxx)
endif()
add_library(cudf_streaming::optional_communication ALIAS cudf_streaming_optional_communication)

message(STATUS "CUDF_STREAMING: MPI support: ${CUDF_STREAMING_HAVE_MPI}")
message(STATUS "CUDF_STREAMING: UCXX support: ${CUDF_STREAMING_HAVE_UCXX}")

# Emit a standardized status message for targets disabled by unavailable optional dependencies.
function(cudf_streaming_skip_optional_target target reason)
  message(STATUS "CUDF_STREAMING: Skipping ${target}: ${reason}")
endfunction()
