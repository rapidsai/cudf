cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_BENCHS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC}
                   "${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp")
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_BENCH_NAME} benchmark benchmark_main pthread cudf )
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
endfunction(ConfigureBench)

option(CMAKE_ENABLE_BENCHMARKS "Enable building cuDF benchmarks" OFF)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    "${CMAKE_SOURCE_DIR}/thirdparty/jitify"
                    "${CMAKE_SOURCE_DIR}/thirdparty/libcudacxx/include"
                    "${GTEST_INCLUDE_DIR}"
                    "${GBENCH_INCLUDE_DIR}"
                    "${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${RMM_INCLUDE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${GBENCH_LIBRARY_DIR}"
                 "${RMM_LIBRARY}")

###################################################################################################
# - gather benchmark ------------------------------------------------------------------------------

set(GATHER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/copying/gather_benchmark.cu")

ConfigureBench(GATHER_BENCH "${GATHER_BENCH_SRC}")

###################################################################################################
# - scatter benchmark -----------------------------------------------------------------------------

set(SCATTER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/copying/scatter_benchmark.cu")

ConfigureBench(SCATTER_BENCH "${SCATTER_BENCH_SRC}")

###################################################################################################
# - contiguous_split benchmark  -------------------------------------------------------------------

set(CONTIGUOUS_SPLIT_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/copying/contiguous_split_benchmark.cu")

ConfigureBench(CONTIGUOUS_SPLIT_BENCH "${CONTIGUOUS_SPLIT_BENCH_SRC}")

###################################################################################################
# - apply_boolean_mask benchmark ------------------------------------------------------------------------------

set(APPLY_BOOLEAN_MASK_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/apply_boolean_mask_benchmark.cu")

ConfigureBench(APPLY_BOOLEAN_MASK_BENCH "${APPLY_BOOLEAN_MASK_BENCH_SRC}")

# - join benchmark --------------------------------------------------------------------------------

set(JOIN_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/join/join_benchmark.cu")

ConfigureBench(JOIN_BENCH "${JOIN_BENCH_SRC}")

###################################################################################################
# - search benchmark ------------------------------------------------------------------------------

set(SEARCH_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/search/search_benchmark.cu")

ConfigureBench(SEARCH_BENCH "${SEARCH_BENCH_SRC}")


###################################################################################################
# - type_dispatcher benchmark ---------------------------------------------------------------------

set(TD_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/type_dispatcher/type_dispatcher_benchmark.cu")

ConfigureBench(TYPE_DISPATCHER_BENCH "${TD_BENCH_SRC}")

###################################################################################################
# - quantiles benchmark ---------------------------------------------------------------------------

set(LEGACY_QUANTILES_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/quantiles/legacy/group_quantiles_benchmark.cu")

ConfigureBench(LEGACY_QUANTILES_BENCH "${LEGACY_QUANTILES_BENCH_SRC}")

###################################################################################################
# - stack benchmark -------------------------------------------------------------------------------

set(LEGACY_STACK_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/reshape/legacy/stack_benchmark.cu")

ConfigureBench(LEGACY_STACK_BENCH "${LEGACY_STACK_BENCH_SRC}")
