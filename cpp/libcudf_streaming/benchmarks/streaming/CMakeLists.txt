# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(RAPIDSMPF_HAVE_MPI AND RAPIDSMPF_HAVE_UCXX)
  add_executable(bench_streaming_shuffle "bench_streaming_shuffle.cpp")
  set_target_properties(
    bench_streaming_shuffle
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/benchmarks"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    bench_streaming_shuffle PRIVATE cudf_streaming rapidsmpf::rapidsmpf ucxx::ucxx MPI::MPI_CXX
                                    $<TARGET_NAME_IF_EXISTS:conda_env> bench_utils
  )
  install(
    TARGETS bench_streaming_shuffle
    COMPONENT benchmarking
    DESTINATION bin/benchmarks/libcudf_streaming
    EXCLUDE_FROM_ALL
  )

  add_subdirectory(ndsh)
endif()
