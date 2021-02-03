#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(CUDF VERSION 0.19.0 LANGUAGES C CXX)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)
option(BUILD_SHARED_LIBS "Build cuDF shared libraries" ON)
option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
option(ARROW_STATIC_LIB "Build and statically link Arrow libraries" OFF)
option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(AUTO_DETECT_CUDA_ARCHITECTURES "Auto-detect CUDA architectures to build from the GPUs on the current system" OFF)
option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CUDA_ENABLE_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

message(STATUS "CUDF: Build with NVTX support: ${USE_NVTX}")
message(STATUS "CUDF: Configure CMake to build tests: ${BUILD_TESTS}")
message(STATUS "CUDF: Configure CMake to build (google) benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "CUDF: Build cuDF shared libraries: ${BUILD_SHARED_LIBS}")
message(STATUS "CUDF: Use a file cache for JIT compiled kernels: ${JITIFY_USE_CACHE}")
message(STATUS "CUDF: Build and statically link Arrow libraries: ${ARROW_STATIC_LIB}")
message(STATUS "CUDF: Build with per-thread default stream: ${PER_THREAD_DEFAULT_STREAM}")
message(STATUS "CUDF: Auto-detect CUDA architectures to build from the GPUs on the current system: ${AUTO_DETECT_CUDA_ARCHITECTURES}")
message(STATUS "CUDF: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNING}")
message(STATUS "CUDF: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler: ${CUDA_ENABLE_LINEINFO}")
message(STATUS "CUDF: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")
set(CUDF_BUILD_TESTS ${BUILD_TESTS})
set(CUDF_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})

set(CUDF_CXX_FLAGS "")
set(CUDF_CUDA_FLAGS "")
set(CUDF_CXX_DEFINITIONS "")
set(CUDF_CUDA_DEFINITIONS "")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "CUDF: Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING  "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

# Set RMM logging level
set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "CUDF: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

if (NOT CUDF_GENERATED_INCLUDE_DIR)
    set(CUDF_GENERATED_INCLUDE_DIR ${CUDF_BINARY_DIR})
endif()

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    message(STATUS "CUDF: Conda build detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "CUDF: Conda environment detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
include(cmake/Modules/ConfigureCUDA.cmake)

###################################################################################################
# - dependencies ----------------------------------------------------------------------------------

# find zlib
find_package(ZLIB REQUIRED)
# add third party dependencies using CPM
include(cmake/thirdparty/CUDF_GetCPM.cmake)
# find boost
include(cmake/thirdparty/CUDF_FindBoost.cmake)
# find jitify
include(cmake/thirdparty/CUDF_GetJitify.cmake)
# find thrust/cub
include(cmake/thirdparty/CUDF_GetThrust.cmake)
# find rmm
include(cmake/thirdparty/CUDF_GetRMM.cmake)
# find arrow
include(cmake/thirdparty/CUDF_GetArrow.cmake)
# find dlpack
include(cmake/thirdparty/CUDF_GetDLPack.cmake)
# find libcu++
include(cmake/thirdparty/CUDF_GetLibcudacxx.cmake)
# Stringify libcudf and libcudacxx headers used in JIT operations
include(cmake/Modules/StringifyJITHeaders.cmake)

###################################################################################################
# - add tests -------------------------------------------------------------------------------------

if(CUDF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

###################################################################################################
# - add benchmarks --------------------------------------------------------------------------------

if(CUDF_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf
    "${CUDF_SOURCE_DIR}/src/aggregation/aggregation.cpp"
    "${CUDF_SOURCE_DIR}/src/aggregation/aggregation.cu"
    "${CUDF_SOURCE_DIR}/src/aggregation/result_cache.cpp"
    "${CUDF_SOURCE_DIR}/src/ast/linearizer.cpp"
    "${CUDF_SOURCE_DIR}/src/ast/transform.cu"
    "${CUDF_SOURCE_DIR}/src/binaryop/binaryop.cpp"
    "${CUDF_SOURCE_DIR}/src/binaryop/compiled/binary_ops.cu"
    "${CUDF_SOURCE_DIR}/src/binaryop/jit/code/kernel.cpp"
    "${CUDF_SOURCE_DIR}/src/binaryop/jit/code/operation.cpp"
    "${CUDF_SOURCE_DIR}/src/binaryop/jit/code/traits.cpp"
    "${CUDF_SOURCE_DIR}/src/bitmask/null_mask.cu"
    "${CUDF_SOURCE_DIR}/src/column/column.cu"
    "${CUDF_SOURCE_DIR}/src/column/column_device_view.cu"
    "${CUDF_SOURCE_DIR}/src/column/column_factories.cpp"
    "${CUDF_SOURCE_DIR}/src/column/column_view.cpp"
    "${CUDF_SOURCE_DIR}/src/comms/ipc/ipc.cpp"
    "${CUDF_SOURCE_DIR}/src/copying/concatenate.cu"
    "${CUDF_SOURCE_DIR}/src/copying/contiguous_split.cu"
    "${CUDF_SOURCE_DIR}/src/copying/copy.cpp"
    "${CUDF_SOURCE_DIR}/src/copying/copy.cu"
    "${CUDF_SOURCE_DIR}/src/copying/copy_range.cu"
    "${CUDF_SOURCE_DIR}/src/copying/gather.cu"
    "${CUDF_SOURCE_DIR}/src/copying/get_element.cu"
    "${CUDF_SOURCE_DIR}/src/copying/sample.cu"
    "${CUDF_SOURCE_DIR}/src/copying/scatter.cu"
    "${CUDF_SOURCE_DIR}/src/copying/shift.cu"
    "${CUDF_SOURCE_DIR}/src/copying/slice.cpp"
    "${CUDF_SOURCE_DIR}/src/copying/split.cpp"
    "${CUDF_SOURCE_DIR}/src/datetime/datetime_ops.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/add_keys.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/decode.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/detail/concatenate.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/detail/merge.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/dictionary_column_view.cpp"
    "${CUDF_SOURCE_DIR}/src/dictionary/dictionary_factories.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/encode.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/remove_keys.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/replace.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/search.cu"
    "${CUDF_SOURCE_DIR}/src/dictionary/set_keys.cu"
    "${CUDF_SOURCE_DIR}/src/filling/fill.cu"
    "${CUDF_SOURCE_DIR}/src/filling/repeat.cu"
    "${CUDF_SOURCE_DIR}/src/filling/sequence.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/groupby.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/hash/groupby.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_argmax.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_argmin.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/groupby.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_collect.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_count.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_max.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_min.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_nth_element.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_nunique.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_quantiles.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_std.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/group_sum.cu"
    "${CUDF_SOURCE_DIR}/src/groupby/sort/sort_helper.cu"
    "${CUDF_SOURCE_DIR}/src/hash/hashing.cu"
    "${CUDF_SOURCE_DIR}/src/interop/dlpack.cpp"
    "${CUDF_SOURCE_DIR}/src/interop/from_arrow.cpp"
    "${CUDF_SOURCE_DIR}/src/interop/to_arrow.cpp"
    "${CUDF_SOURCE_DIR}/src/io/avro/avro.cpp"
    "${CUDF_SOURCE_DIR}/src/io/avro/avro_gpu.cu"
    "${CUDF_SOURCE_DIR}/src/io/avro/reader_impl.cu"
    "${CUDF_SOURCE_DIR}/src/io/comp/brotli_dict.cpp"
    "${CUDF_SOURCE_DIR}/src/io/comp/cpu_unbz2.cpp"
    "${CUDF_SOURCE_DIR}/src/io/comp/debrotli.cu"
    "${CUDF_SOURCE_DIR}/src/io/comp/gpuinflate.cu"
    "${CUDF_SOURCE_DIR}/src/io/comp/snap.cu"
    "${CUDF_SOURCE_DIR}/src/io/comp/uncomp.cpp"
    "${CUDF_SOURCE_DIR}/src/io/comp/unsnap.cu"
    "${CUDF_SOURCE_DIR}/src/io/csv/csv_gpu.cu"
    "${CUDF_SOURCE_DIR}/src/io/csv/durations.cu"
    "${CUDF_SOURCE_DIR}/src/io/csv/reader_impl.cu"
    "${CUDF_SOURCE_DIR}/src/io/csv/writer_impl.cu"
    "${CUDF_SOURCE_DIR}/src/io/functions.cpp"
    "${CUDF_SOURCE_DIR}/src/io/json/json_gpu.cu"
    "${CUDF_SOURCE_DIR}/src/io/json/reader_impl.cu"
    "${CUDF_SOURCE_DIR}/src/io/orc/dict_enc.cu"
    "${CUDF_SOURCE_DIR}/src/io/orc/orc.cpp"
    "${CUDF_SOURCE_DIR}/src/io/orc/reader_impl.cu"
    "${CUDF_SOURCE_DIR}/src/io/orc/stats_enc.cu"
    "${CUDF_SOURCE_DIR}/src/io/orc/stripe_data.cu"
    "${CUDF_SOURCE_DIR}/src/io/orc/stripe_enc.cu"
    "${CUDF_SOURCE_DIR}/src/io/orc/stripe_init.cu"
    "${CUDF_SOURCE_DIR}/src/io/orc/timezone.cpp"
    "${CUDF_SOURCE_DIR}/src/io/orc/writer_impl.cu"
    "${CUDF_SOURCE_DIR}/src/io/parquet/compact_protocol_writer.cpp"
    "${CUDF_SOURCE_DIR}/src/io/parquet/page_data.cu"
    "${CUDF_SOURCE_DIR}/src/io/parquet/page_dict.cu"
    "${CUDF_SOURCE_DIR}/src/io/parquet/page_enc.cu"
    "${CUDF_SOURCE_DIR}/src/io/parquet/page_hdr.cu"
    "${CUDF_SOURCE_DIR}/src/io/parquet/parquet.cpp"
    "${CUDF_SOURCE_DIR}/src/io/parquet/reader_impl.cu"
    "${CUDF_SOURCE_DIR}/src/io/parquet/writer_impl.cu"
    "${CUDF_SOURCE_DIR}/src/io/statistics/column_stats.cu"
    "${CUDF_SOURCE_DIR}/src/io/utilities/data_sink.cpp"
    "${CUDF_SOURCE_DIR}/src/io/utilities/datasource.cpp"
    "${CUDF_SOURCE_DIR}/src/io/utilities/parsing_utils.cu"
    "${CUDF_SOURCE_DIR}/src/io/utilities/type_conversion.cpp"
    "${CUDF_SOURCE_DIR}/src/jit/cache.cpp"
    "${CUDF_SOURCE_DIR}/src/jit/launcher.cpp"
    "${CUDF_SOURCE_DIR}/src/jit/parser.cpp"
    "${CUDF_SOURCE_DIR}/src/jit/type.cpp"
    "${CUDF_SOURCE_DIR}/src/join/cross_join.cu"
    "${CUDF_SOURCE_DIR}/src/join/hash_join.cu"
    "${CUDF_SOURCE_DIR}/src/join/join.cu"
    "${CUDF_SOURCE_DIR}/src/join/semi_join.cu"
    "${CUDF_SOURCE_DIR}/src/lists/contains.cu"
    "${CUDF_SOURCE_DIR}/src/lists/copying/concatenate.cu"
    "${CUDF_SOURCE_DIR}/src/lists/copying/copying.cu"
    "${CUDF_SOURCE_DIR}/src/lists/copying/gather.cu"
    "${CUDF_SOURCE_DIR}/src/lists/copying/segmented_gather.cu"
    "${CUDF_SOURCE_DIR}/src/lists/count_elements.cu"
    "${CUDF_SOURCE_DIR}/src/lists/extract.cu"
    "${CUDF_SOURCE_DIR}/src/lists/lists_column_factories.cu"
    "${CUDF_SOURCE_DIR}/src/lists/lists_column_view.cu"
    "${CUDF_SOURCE_DIR}/src/merge/merge.cu"
    "${CUDF_SOURCE_DIR}/src/partitioning/partitioning.cu"
    "${CUDF_SOURCE_DIR}/src/partitioning/round_robin.cu"
    "${CUDF_SOURCE_DIR}/src/quantiles/quantile.cu"
    "${CUDF_SOURCE_DIR}/src/quantiles/quantiles.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/all.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/any.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/max.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/mean.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/min.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/minmax.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/nth_element.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/product.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/reductions.cpp"
    "${CUDF_SOURCE_DIR}/src/reductions/scan.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/std.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/sum.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/sum_of_squares.cu"
    "${CUDF_SOURCE_DIR}/src/reductions/var.cu"
    "${CUDF_SOURCE_DIR}/src/replace/clamp.cu"
    "${CUDF_SOURCE_DIR}/src/replace/nans.cu"
    "${CUDF_SOURCE_DIR}/src/replace/nulls.cu"
    "${CUDF_SOURCE_DIR}/src/replace/replace.cu"
    "${CUDF_SOURCE_DIR}/src/reshape/byte_cast.cu"
    "${CUDF_SOURCE_DIR}/src/reshape/explode.cu"
    "${CUDF_SOURCE_DIR}/src/reshape/interleave_columns.cu"
    "${CUDF_SOURCE_DIR}/src/reshape/tile.cu"
    "${CUDF_SOURCE_DIR}/src/rolling/grouped_rolling.cu"
    "${CUDF_SOURCE_DIR}/src/rolling/jit/code/kernel.cpp"
    "${CUDF_SOURCE_DIR}/src/rolling/jit/code/operation.cpp"
    "${CUDF_SOURCE_DIR}/src/rolling/rolling.cu"
    "${CUDF_SOURCE_DIR}/src/round/round.cu"
    "${CUDF_SOURCE_DIR}/src/scalar/scalar.cpp"
    "${CUDF_SOURCE_DIR}/src/scalar/scalar_factories.cpp"
    "${CUDF_SOURCE_DIR}/src/search/search.cu"
    "${CUDF_SOURCE_DIR}/src/sort/is_sorted.cu"
    "${CUDF_SOURCE_DIR}/src/sort/rank.cu"
    "${CUDF_SOURCE_DIR}/src/sort/sort_column.cu"
    "${CUDF_SOURCE_DIR}/src/sort/sort.cu"
    "${CUDF_SOURCE_DIR}/src/sort/stable_sort_column.cu"
    "${CUDF_SOURCE_DIR}/src/sort/stable_sort.cu"
    "${CUDF_SOURCE_DIR}/src/stream_compaction/apply_boolean_mask.cu"
    "${CUDF_SOURCE_DIR}/src/stream_compaction/distinct_count.cu"
    "${CUDF_SOURCE_DIR}/src/stream_compaction/drop_duplicates.cu"
    "${CUDF_SOURCE_DIR}/src/stream_compaction/drop_nans.cu"
    "${CUDF_SOURCE_DIR}/src/stream_compaction/drop_nulls.cu"
    "${CUDF_SOURCE_DIR}/src/strings/attributes.cu"
    "${CUDF_SOURCE_DIR}/src/strings/capitalize.cu"
    "${CUDF_SOURCE_DIR}/src/strings/case.cu"
    "${CUDF_SOURCE_DIR}/src/strings/char_types/char_cases.cu"
    "${CUDF_SOURCE_DIR}/src/strings/char_types/char_types.cu"
    "${CUDF_SOURCE_DIR}/src/strings/combine.cu"
    "${CUDF_SOURCE_DIR}/src/strings/contains.cu"
    "${CUDF_SOURCE_DIR}/src/strings/convert/convert_booleans.cu"
    "${CUDF_SOURCE_DIR}/src/strings/convert/convert_datetime.cu"
    "${CUDF_SOURCE_DIR}/src/strings/convert/convert_durations.cu"
    "${CUDF_SOURCE_DIR}/src/strings/convert/convert_floats.cu"
    "${CUDF_SOURCE_DIR}/src/strings/convert/convert_hex.cu"
    "${CUDF_SOURCE_DIR}/src/strings/convert/convert_integers.cu"
    "${CUDF_SOURCE_DIR}/src/strings/convert/convert_ipv4.cu"
    "${CUDF_SOURCE_DIR}/src/strings/convert/convert_urls.cu"
    "${CUDF_SOURCE_DIR}/src/strings/copying/concatenate.cu"
    "${CUDF_SOURCE_DIR}/src/strings/copying/copying.cu"
    "${CUDF_SOURCE_DIR}/src/strings/extract.cu"
    "${CUDF_SOURCE_DIR}/src/strings/filling/fill.cu"
    "${CUDF_SOURCE_DIR}/src/strings/filter_chars.cu"
    "${CUDF_SOURCE_DIR}/src/strings/findall.cu"
    "${CUDF_SOURCE_DIR}/src/strings/find.cu"
    "${CUDF_SOURCE_DIR}/src/strings/find_multiple.cu"
    "${CUDF_SOURCE_DIR}/src/strings/padding.cu"
    "${CUDF_SOURCE_DIR}/src/strings/regex/regcomp.cpp"
    "${CUDF_SOURCE_DIR}/src/strings/regex/regexec.cu"
    "${CUDF_SOURCE_DIR}/src/strings/replace/backref_re.cu"
    "${CUDF_SOURCE_DIR}/src/strings/replace/backref_re_large.cu"
    "${CUDF_SOURCE_DIR}/src/strings/replace/backref_re_medium.cu"
    "${CUDF_SOURCE_DIR}/src/strings/replace/multi_re.cu"
    "${CUDF_SOURCE_DIR}/src/strings/replace/replace.cu"
    "${CUDF_SOURCE_DIR}/src/strings/replace/replace_re.cu"
    "${CUDF_SOURCE_DIR}/src/strings/split/partition.cu"
    "${CUDF_SOURCE_DIR}/src/strings/split/split.cu"
    "${CUDF_SOURCE_DIR}/src/strings/split/split_record.cu"
    "${CUDF_SOURCE_DIR}/src/strings/strings_column_factories.cu"
    "${CUDF_SOURCE_DIR}/src/strings/strings_column_view.cu"
    "${CUDF_SOURCE_DIR}/src/strings/strings_scalar_factories.cpp"
    "${CUDF_SOURCE_DIR}/src/strings/strip.cu"
    "${CUDF_SOURCE_DIR}/src/strings/substring.cu"
    "${CUDF_SOURCE_DIR}/src/strings/translate.cu"
    "${CUDF_SOURCE_DIR}/src/strings/utilities.cu"
    "${CUDF_SOURCE_DIR}/src/strings/wrap.cu"
    "${CUDF_SOURCE_DIR}/src/structs/copying/concatenate.cu"
    "${CUDF_SOURCE_DIR}/src/structs/structs_column_factories.cu"
    "${CUDF_SOURCE_DIR}/src/structs/structs_column_view.cpp"
    "${CUDF_SOURCE_DIR}/src/structs/utilities.cu"
    "${CUDF_SOURCE_DIR}/src/table/table.cpp"
    "${CUDF_SOURCE_DIR}/src/table/table_device_view.cu"
    "${CUDF_SOURCE_DIR}/src/table/table_view.cpp"
    "${CUDF_SOURCE_DIR}/src/text/detokenize.cu"
    "${CUDF_SOURCE_DIR}/src/text/edit_distance.cu"
    "${CUDF_SOURCE_DIR}/src/text/generate_ngrams.cu"
    "${CUDF_SOURCE_DIR}/src/text/ngrams_tokenize.cu"
    "${CUDF_SOURCE_DIR}/src/text/normalize.cu"
    "${CUDF_SOURCE_DIR}/src/text/replace.cu"
    "${CUDF_SOURCE_DIR}/src/text/stemmer.cu"
    "${CUDF_SOURCE_DIR}/src/text/subword/data_normalizer.cu"
    "${CUDF_SOURCE_DIR}/src/text/subword/load_hash_file.cu"
    "${CUDF_SOURCE_DIR}/src/text/subword/subword_tokenize.cu"
    "${CUDF_SOURCE_DIR}/src/text/subword/wordpiece_tokenizer.cu"
    "${CUDF_SOURCE_DIR}/src/text/tokenize.cu"
    "${CUDF_SOURCE_DIR}/src/transform/bools_to_mask.cu"
    "${CUDF_SOURCE_DIR}/src/transform/encode.cu"
    "${CUDF_SOURCE_DIR}/src/transform/jit/code/kernel.cpp"
    "${CUDF_SOURCE_DIR}/src/transform/mask_to_bools.cu"
    "${CUDF_SOURCE_DIR}/src/transform/nans_to_nulls.cu"
    "${CUDF_SOURCE_DIR}/src/transform/transform.cpp"
    "${CUDF_SOURCE_DIR}/src/transpose/transpose.cu"
    "${CUDF_SOURCE_DIR}/src/unary/cast_ops.cu"
    "${CUDF_SOURCE_DIR}/src/unary/math_ops.cu"
    "${CUDF_SOURCE_DIR}/src/unary/nan_ops.cu"
    "${CUDF_SOURCE_DIR}/src/unary/null_ops.cu"
    "${CUDF_SOURCE_DIR}/src/utilities/default_stream.cpp"
)

set_target_properties(cudf
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        14
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       14
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cudf
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_compile_definitions(cudf
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                   "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>"
)

# Disable Jitify log printing. See https://github.com/NVIDIA/jitify/issues/79
target_compile_definitions(cudf PRIVATE "JITIFY_PRINT_LOG=0")

# Specify include paths for the current target and dependents
target_include_directories(cudf
           PUBLIC      "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
                       "$<BUILD_INTERFACE:${JITIFY_INCLUDE_DIR}>"
                       "$<BUILD_INTERFACE:${LIBCUDACXX_INCLUDE_DIR}>"
                       "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
                       "$<BUILD_INTERFACE:${CUDF_GENERATED_INCLUDE_DIR}/include>"
           PRIVATE     "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
           INTERFACE   "$<INSTALL_INTERFACE:include>")

# Add Conda library paths if specified
if(CONDA_LINK_DIRS)
    target_link_directories(cudf PUBLIC "$<BUILD_INTERFACE:${CONDA_LINK_DIRS}>")
endif()

# Add Conda include paths if specified
if(CONDA_INCLUDE_DIRS)
    target_include_directories(cudf PUBLIC "$<BUILD_INTERFACE:${CONDA_INCLUDE_DIRS}>")
endif()

# Instruct jitify to use the kernel JIT cache
if(JITIFY_USE_CACHE)
    target_compile_definitions(cudf PUBLIC JITIFY_USE_CACHE "CUDF_VERSION=${CMAKE_PROJECT_VERSION}")
endif()

# Per-thread default stream
if(PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(cudf PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()

# Disable NVTX if necessary
if(NOT USE_NVTX)
    target_compile_definitions(cudf PUBLIC NVTX_DISABLE)
endif()

# Define spdlog level
target_compile_definitions(cudf PUBLIC "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

# Compile stringified JIT sources first
add_dependencies(cudf stringify_run)

# Specify the target module library dependencies
target_link_libraries(cudf
           PUBLIC ZLIB::ZLIB
                  Boost::filesystem
                  ${ARROW_LIBRARIES}
                  cudf::Thrust
                  rmm::rmm)

if(CUDA_STATIC_RUNTIME)
    target_link_libraries(cudf PUBLIC CUDA::nvrtc CUDA::cudart_static CUDA::cuda_driver)
else()
    target_link_libraries(cudf PUBLIC CUDA::nvrtc CUDA::cudart CUDA::cuda_driver)
endif()

file(WRITE "${CUDF_BINARY_DIR}/fatbin.ld" "
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
")

target_link_options(cudf PRIVATE "${CUDF_BINARY_DIR}/fatbin.ld")

add_library(cudf::cudf ALIAS cudf)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

include(GNUInstallDirs)

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/cudf)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cudf)

# install target for cudf_base
install(TARGETS cudf
        DESTINATION lib
        EXPORT cudf-targets)

# install target for the proxy libcudf.so
install(TARGETS ${CUDF_PROXY_NAME}
        DESTINATION lib
        EXPORT cudf-targets)

install(DIRECTORY ${CUDF_SOURCE_DIR}/include/cudf
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CUDF_SOURCE_DIR}/include/cudf_test
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY ${CUDF_GENERATED_INCLUDE_DIR}/include/libcxx
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libcudf)

install(DIRECTORY ${CUDF_GENERATED_INCLUDE_DIR}/include/libcudacxx
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libcudf)

include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/cudf-config.cmake.in "${CUDF_BINARY_DIR}/cudf-config.cmake"
                              INSTALL_DESTINATION "${INSTALL_CONFIGDIR}")

write_basic_package_version_file("${CUDF_BINARY_DIR}/cudf-config-version.cmake"
                                 COMPATIBILITY SameMinorVersion)

install(FILES   "${CUDF_BINARY_DIR}/cudf-config.cmake"
                "${CUDF_BINARY_DIR}/cudf-config-version.cmake"
                "${CUDF_SOURCE_DIR}/cmake/install/FindThrust.cmake"
    DESTINATION "${INSTALL_CONFIGDIR}")

install(EXPORT  cudf-targets
    FILE        cudf-targets.cmake
    NAMESPACE   cudf::
    DESTINATION "${INSTALL_CONFIGDIR}")

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

# doc targets for cuDF
add_custom_command(OUTPUT CUDF_DOXYGEN
                   WORKING_DIRECTORY ${CUDF_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(docs_cudf DEPENDS CUDF_DOXYGEN)
