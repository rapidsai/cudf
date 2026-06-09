# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds rapidsmpf and sets any additional necessary environment variables.
function(find_and_configure_rapidsmpf VERSION)
  rapids_cmake_parse_version(MAJOR_MINOR ${VERSION} major_minor)
  set(rapidsmpf_build_comm_support OFF)
  if(BUILD_BENCHMARKS)
    set(rapidsmpf_build_comm_support ON)
  endif()
  rapids_cpm_find(
    rapidsmpf ${VERSION}
    BUILD_EXPORT_SET cudf_streaming-exports
    INSTALL_EXPORT_SET cudf_streaming-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/rapidsmpf.git
    GIT_TAG "${RAPIDS_BRANCH}"
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "BUILD_MPI_SUPPORT ${rapidsmpf_build_comm_support}"
            "BUILD_UCXX_SUPPORT ${rapidsmpf_build_comm_support}"
            "BUILD_SLURM_SUPPORT OFF"
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
            "BUILD_EXAMPLES OFF"
  )
  if(BUILD_BENCHMARKS)
    set(rapidsmpf_missing_required_features OFF)
    if((DEFINED RAPIDSMPF_HAVE_MPI AND NOT RAPIDSMPF_HAVE_MPI) OR (DEFINED RAPIDSMPF_HAVE_UCXX
                                                                   AND NOT RAPIDSMPF_HAVE_UCXX)
    )
      set(rapidsmpf_missing_required_features ON)
    endif()

    if(TARGET rapidsmpf::rapidsmpf)
      get_target_property(
        rapidsmpf_compile_definitions rapidsmpf::rapidsmpf INTERFACE_COMPILE_DEFINITIONS
      )
      if(rapidsmpf_compile_definitions
         AND (rapidsmpf_compile_definitions MATCHES "\\$<\\$<BOOL:OFF>:RAPIDSMPF_HAVE_MPI>"
              OR rapidsmpf_compile_definitions MATCHES "\\$<\\$<BOOL:OFF>:RAPIDSMPF_HAVE_UCXX>")
      )
        set(rapidsmpf_missing_required_features ON)
      endif()
    endif()

    if(rapidsmpf_missing_required_features)
      message(
        FATAL_ERROR
          "libcudf_streaming benchmarks require rapidsmpf with MPI and UCXX support. "
          "Use -DCPM_DOWNLOAD_rapidsmpf=ON to build rapidsmpf from source, or disable BUILD_BENCHMARKS."
      )
    endif()
  endif()
endfunction()

set(CUDF_STREAMING_MIN_VERSION
    "${CUDF_STREAMING_VERSION_MAJOR}.${CUDF_STREAMING_VERSION_MINOR}.${CUDF_STREAMING_VERSION_PATCH}"
)
find_and_configure_rapidsmpf(${CUDF_STREAMING_MIN_VERSION})
