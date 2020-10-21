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
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CUDA_DATAFRAME VERSION 0.16.0 LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif(NOT CMAKE_CUDA_COMPILER)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated-declarations")

    # Suppress parentheses warning which causes gmock to fail
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-parentheses")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(CMAKE_CUDA_COMPILER_VERSION)
  # Compute the version. from  CMAKE_CUDA_COMPILER_VERSION
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${CMAKE_CUDA_COMPILER_VERSION})
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
  mark_as_advanced(CUDA_VERSION)
endif()

message(STATUS "CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
message(STATUS "CUDA_VERSION_MINOR: ${CUDA_VERSION_MINOR}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")

# Always set this convenience variable
set(CUDA_VERSION_STRING "${CUDA_VERSION}")

# Auto-detect available GPU compute architectures
set(GPU_ARCHS "ALL" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for. Pass 'ALL' if you want to compile for all supported GPU architectures. Empty string means to auto-detect the GPUs on the current system")

if("${GPU_ARCHS}" STREQUAL "")
  include(cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

if("${GPU_ARCHS}" STREQUAL "ALL")
  
  # Check for embedded vs workstation architectures
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # This is being built for Linux4Tegra or SBSA ARM64
    set(GPU_ARCHS "62")
    if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
      set(GPU_ARCHS "${GPU_ARCHS};72")
    endif()
    if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
      set(GPU_ARCHS "${GPU_ARCHS};75;80")
    endif()

  else()
    # This is being built for an x86 or x86_64 architecture
    set(GPU_ARCHS "60")
    if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
      set(GPU_ARCHS "${GPU_ARCHS};70")
    endif()
    if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
      set(GPU_ARCHS "${GPU_ARCHS};75")
    endif()
    if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
      set(GPU_ARCHS "${GPU_ARCHS};80")
    endif()

  endif()
  
endif()
message("GPU_ARCHS = ${GPU_ARCHS}")

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${arch},code=sm_${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_${ptx},code=compute_${ptx}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# set warnings as errors
# TODO: remove `no-maybe-unitialized` used to suppress warnings in rmm::exec_policy
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror=cross-execution-space-call -Xcompiler -Wall,-Werror,-Wno-error=deprecated-declarations")

option(DISABLE_DEPRECATION_WARNING "Disable warnings generated from deprecated declarations." OFF)
if(DISABLE_DEPRECATION_WARNING)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif(DISABLE_DEPRECATION_WARNING)

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
if(CMAKE_CUDA_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler -rdynamic")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

# To apply RUNPATH to transitive dependencies (this is a temporary solution)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)

###################################################################################################
# - cudart options --------------------------------------------------------------------------------
# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking

option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)

if(CUDA_STATIC_RUNTIME)
    message(STATUS "Enabling static linking of cudart")
    set(CUDART_LIBRARY "cudart_static")
else()
    set(CUDART_LIBRARY "cudart")
endif(CUDA_STATIC_RUNTIME)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - third party dependencies-----------------------------------------------------------------------
# add third party dependencies using CMake FetchContent

add_subdirectory(thirdparty)

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

###################################################################################################
# - find arrow ------------------------------------------------------------------------------------

option(ARROW_STATIC_LIB "Build and statically link with Arrow libraries" OFF)

if(ARROW_STATIC_LIB)
  message(STATUS "BUILDING ARROW")
  include(ConfigureArrow)

  if(ARROW_FOUND)
    message(STATUS "Apache Arrow found in ${ARROW_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Apache Arrow not found, please check your settings.")
  endif(ARROW_FOUND)

  add_library(arrow STATIC IMPORTED ${ARROW_LIB})
  add_library(arrow_cuda STATIC IMPORTED ${ARROW_CUDA_LIB})
else()
  find_path(ARROW_INCLUDE_DIR "arrow"
      HINTS "$ENV{ARROW_ROOT}/include")

  find_library(ARROW_LIB "arrow"
      NAMES libarrow
      HINTS "$ENV{ARROW_ROOT}/lib" "$ENV{ARROW_ROOT}/build")

  find_library(ARROW_CUDA_LIB "arrow_cuda"
      NAMES libarrow_cuda
      HINTS "$ENV{ARROW_ROOT}/lib" "$ENV{ARROW_ROOT}/build")

  message(STATUS "ARROW: ARROW_INCLUDE_DIR set to ${ARROW_INCLUDE_DIR}")
  message(STATUS "ARROW: ARROW_LIB set to ${ARROW_LIB}")
  message(STATUS "ARROW: ARROW_CUDA_LIB set to ${ARROW_CUDA_LIB}")

  add_library(arrow SHARED IMPORTED ${ARROW_LIB})
  add_library(arrow_cuda SHARED IMPORTED ${ARROW_CUDA_LIB})
endif(ARROW_STATIC_LIB)

if(ARROW_INCLUDE_DIR AND ARROW_LIB AND ARROW_CUDA_LIB)
  set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIB})
  set_target_properties(arrow_cuda PROPERTIES IMPORTED_LOCATION ${ARROW_CUDA_LIB})
endif(ARROW_INCLUDE_DIR AND ARROW_LIB AND ARROW_CUDA_LIB)

###################################################################################################
# - copy libcu++ ------------------------------------------------------------------------------------

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/libcudacxx")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/libcxx/include")

file(INSTALL ${LIBCUDACXX_INCLUDE_DIR} DESTINATION ${CMAKE_BINARY_DIR}/include/libcudacxx/)
file(INSTALL ${LIBCXX_INCLUDE_DIR}     DESTINATION ${CMAKE_BINARY_DIR}/include/libcxx/include/)

###################################################################################################
# - find zlib -------------------------------------------------------------------------------------

find_package(ZLIB REQUIRED)

message(STATUS "ZLIB: ZLIB_LIBRARIES set to ${ZLIB_LIBRARIES}")
message(STATUS "ZLIB: ZLIB_INCLUDE_DIRS set to ${ZLIB_INCLUDE_DIRS}")

if(ZLIB_FOUND)
    message(STATUS "ZLib found in ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLib not found, please check your settings.")
endif(ZLIB_FOUND)

###################################################################################################
# - find boost ------------------------------------------------------------------------------------

# Don't look for a CMake configuration file
set(Boost_NO_BOOST_CMAKE ON)

find_package(
    Boost REQUIRED MODULE
    COMPONENTS filesystem
)

message(STATUS "BOOST: Boost_LIBRARIES set to ${Boost_LIBRARIES}")
message(STATUS "BOOST: Boost_INCLUDE_DIRS set to ${Boost_INCLUDE_DIRS}")

if(Boost_FOUND)
    message(STATUS "Boost found in ${Boost_INCLUDE_DIRS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
else()
    message(FATAL_ERROR "Boost not found, please check your settings.")
endif(Boost_FOUND)

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{RMM_ROOT}/include")

message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

###################################################################################################
# - DLPACK -------------------------------------------------------------------------------------------

find_path(
    DLPACK_INCLUDE "dlpack"
    HINTS "$ENV{DLPACK_ROOT}/include"
)

message(STATUS "DLPACK: DLPACK_INCLUDE set to ${DLPACK_INCLUDE}")

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
if(JITIFY_USE_CACHE)
    message(STATUS "Using file cache for JIT compiled kernels")
    add_definitions("-DJITIFY_USE_CACHE -DCUDF_VERSION=${CMAKE_PROJECT_VERSION}")
endif(JITIFY_USE_CACHE)

###################################################################################################
# - per-thread default stream option --------------------------------------------------------------
# This needs to be defined first so tests and benchmarks can inherit it.

option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
if(PER_THREAD_DEFAULT_STREAM)
    message(STATUS "Using per-thread default stream")
    add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif(PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories(${GTEST_INCLUDE_DIR})
        add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

message(STATUS "CUDF_TEST_LIST set to: ${CUDF_TEST_LIST}")

###################################################################################################
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)

  include(ConfigureGoogleBenchmark)

  if(GBENCH_FOUND)
    message(STATUS "Google C++ Benchmarking Framework (Google Benchmark) found in ${GBENCH_ROOT}")
    include_directories(${GBENCH_INCLUDE_DIR})
    add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks)
  else()
    message(AUTHOR_WARNING "Google C++ Benchmarking Framework (Google Benchmark) not found: automated tests are disabled.")
  endif(GBENCH_FOUND)

endif(BUILD_BENCHMARKS)

message(STATUS "BENCHMARK_LIST set to: ${BENCHMARK_LIST}")

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${THRUST_INCLUDE_DIR}"
                    "${JITIFY_INCLUDE_DIR}"
                    "${LIBCUDACXX_INCLUDE_DIR}")

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_BINARY_DIR}/include/jit"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${ARROW_INCLUDE_DIR}"
                    "${ZLIB_INCLUDE_DIRS}"
                    "${Boost_INCLUDE_DIRS}"
                    "${RMM_INCLUDE}"
                    "${DLPACK_INCLUDE}")

if(CONDA_INCLUDE_DIRS)
    include_directories("${CONDA_INCLUDE_DIRS}")
endif(CONDA_INCLUDE_DIRS)

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${GTEST_LIBRARY_DIR}")

if(CONDA_LINK_DIRS)
    link_directories("${CONDA_LINK_DIRS}")
endif(CONDA_LINK_DIRS)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf
            src/comms/ipc/ipc.cpp
            src/merge/merge.cu
            src/partitioning/round_robin.cu
            src/join/join.cu
            src/join/hash_join.cu
            src/join/cross_join.cu
            src/join/semi_join.cu
            src/sort/is_sorted.cu
            src/binaryop/binaryop.cpp
            src/binaryop/compiled/binary_ops.cu
            src/binaryop/jit/code/kernel.cpp
            src/binaryop/jit/code/operation.cpp
            src/binaryop/jit/code/traits.cpp
            src/interop/from_arrow.cpp
            src/interop/to_arrow.cpp
            src/interop/dlpack.cpp
            src/jit/type.cpp
            src/jit/parser.cpp
            src/jit/cache.cpp
            src/jit/launcher.cpp
            src/transform/jit/code/kernel.cpp
            src/transform/transform.cpp
            src/transform/nans_to_nulls.cu
            src/transform/bools_to_mask.cu
            src/transform/mask_to_bools.cu
            src/transform/encode.cu
            src/stream_compaction/apply_boolean_mask.cu
            src/stream_compaction/drop_nulls.cu
            src/stream_compaction/drop_nans.cu
            src/stream_compaction/drop_duplicates.cu
            src/datetime/datetime_ops.cu
            src/hash/hashing.cu
            src/partitioning/partitioning.cu
            src/quantiles/quantile.cu
            src/quantiles/quantiles.cu
            src/reductions/reductions.cpp
            src/reductions/nth_element.cu
            src/reductions/min.cu
            src/reductions/max.cu
            src/reductions/minmax.cu
            src/reductions/any.cu
            src/reductions/all.cu
            src/reductions/sum.cu
            src/reductions/product.cu
            src/reductions/sum_of_squares.cu
            src/reductions/mean.cu
            src/reductions/var.cu
            src/reductions/std.cu
            src/reductions/scan.cu
            src/replace/replace.cu
            src/replace/clamp.cu
            src/replace/nans.cu
            src/replace/nulls.cu
            src/reshape/interleave_columns.cu
            src/transpose/transpose.cu
            src/unary/cast_ops.cu
            src/unary/null_ops.cu
            src/unary/nan_ops.cu
            src/unary/math_ops.cu
            src/unary/unary_ops.cuh
            src/io/avro/avro_gpu.cu
            src/io/avro/avro.cpp
            src/io/avro/reader_impl.cu
            src/io/csv/csv_gpu.cu
            src/io/csv/reader_impl.cu
            src/io/csv/writer_impl.cu
            src/io/csv/durations.cu
            src/io/json/reader_impl.cu
            src/io/json/json_gpu.cu
            src/io/orc/orc.cpp
            src/io/orc/timezone.cpp
            src/io/orc/stripe_data.cu
            src/io/orc/stripe_init.cu
            src/io/orc/stripe_enc.cu
            src/io/orc/dict_enc.cu
            src/io/orc/stats_enc.cu
            src/io/orc/reader_impl.cu
            src/io/orc/writer_impl.cu
            src/io/parquet/page_data.cu
            src/io/parquet/compact_protocol_writer.cpp
            src/io/parquet/page_hdr.cu
            src/io/parquet/page_enc.cu
            src/io/parquet/page_dict.cu
            src/io/parquet/parquet.cpp
            src/io/parquet/reader_impl.cu
            src/io/parquet/writer_impl.cu
            src/io/comp/cpu_unbz2.cpp
            src/io/comp/uncomp.cpp
            src/io/comp/brotli_dict.cpp
            src/io/comp/debrotli.cu
            src/io/comp/snap.cu
            src/io/comp/unsnap.cu
            src/io/comp/gpuinflate.cu
            src/io/functions.cpp
            src/io/statistics/column_stats.cu
            src/io/utilities/datasource.cpp
            src/io/utilities/parsing_utils.cu
            src/io/utilities/type_conversion.cu
            src/io/utilities/data_sink.cpp
            src/copying/gather.cu
            src/copying/copy.cpp
            src/copying/sample.cu
            src/copying/scatter.cu
            src/copying/shift.cu
            src/copying/copy.cu
            src/copying/concatenate.cu
            src/copying/slice.cpp
            src/copying/split.cpp
            src/copying/contiguous_split.cu
            src/copying/copy_range.cu
            src/copying/get_element.cu
            src/filling/fill.cu
            src/filling/repeat.cu
            src/filling/sequence.cu
            src/reshape/byte_cast.cu
            src/reshape/tile.cu
            src/search/search.cu
            src/column/column.cu
            src/column/column_view.cpp
            src/column/column_device_view.cu
            src/column/column_factories.cpp
            src/table/table_view.cpp
            src/table/table_device_view.cu
            src/table/table.cpp
            src/bitmask/null_mask.cu
            src/rolling/rolling.cu
            src/rolling/jit/code/kernel.cpp
            src/rolling/jit/code/operation.cpp
            src/sort/sort.cu
            src/sort/stable_sort.cu
            src/sort/rank.cu
            src/strings/attributes.cu
            src/strings/case.cu
            src/strings/wrap.cu
            src/strings/capitalize.cu
            src/strings/char_types/char_types.cu
            src/strings/char_types/char_cases.cu
            src/strings/combine.cu
            src/strings/contains.cu
            src/strings/convert/convert_booleans.cu
            src/strings/convert/convert_datetime.cu
            src/strings/convert/convert_durations.cu
            src/strings/convert/convert_floats.cu
            src/strings/convert/convert_hex.cu
            src/strings/convert/convert_integers.cu
            src/strings/convert/convert_ipv4.cu
            src/strings/convert/convert_urls.cu
            src/strings/copying/concatenate.cu
            src/strings/copying/copying.cu
            src/strings/extract.cu
            src/strings/filter_chars.cu
            src/strings/find.cu
            src/strings/findall.cu
            src/strings/find_multiple.cu
            src/strings/filling/fill.cu
            src/strings/padding.cu
            src/strings/regex/regcomp.cpp
            src/strings/regex/regexec.cu
            src/strings/replace/replace_re.cu
            src/strings/replace/backref_re.cu
            src/strings/replace/backref_re_medium.cu
            src/strings/replace/backref_re_large.cu
            src/strings/replace/multi_re.cu
            src/strings/replace/replace.cu
            src/strings/sorting/sorting.cu
            src/strings/split/partition.cu
            src/strings/split/split.cu
            src/strings/split/split_record.cu
            src/strings/strings_column_factories.cu
            src/strings/strings_column_view.cu
            src/strings/strings_scalar_factories.cpp
            src/strings/strip.cu
            src/strings/substring.cu
            src/strings/translate.cu
            src/strings/utilities.cu
            src/lists/extract.cu
            src/lists/lists_column_factories.cu
            src/lists/lists_column_view.cu
            src/lists/copying/concatenate.cu
            src/lists/copying/gather.cu
            src/lists/copying/copying.cu
            src/structs/structs_column_view.cu
            src/structs/structs_column_factories.cu
            src/text/detokenize.cu
            src/text/edit_distance.cu
            src/text/generate_ngrams.cu
            src/text/normalize.cu
            src/text/stemmer.cu
            src/text/tokenize.cu
            src/text/ngrams_tokenize.cu
            src/text/replace.cu
            src/text/subword/load_hash_file.cu
            src/text/subword/data_normalizer.cu
            src/text/subword/wordpiece_tokenizer.cu
            src/text/subword/subword_tokenize.cu
            src/scalar/scalar.cpp
            src/scalar/scalar_factories.cpp
            src/dictionary/add_keys.cu
            src/dictionary/detail/concatenate.cu
            src/dictionary/dictionary_column_view.cpp
            src/dictionary/dictionary_factories.cu
            src/dictionary/decode.cu
            src/dictionary/encode.cu
            src/dictionary/remove_keys.cu
            src/dictionary/replace.cu
            src/dictionary/search.cu
            src/dictionary/set_keys.cu
            src/groupby/groupby.cu
            src/groupby/hash/groupby.cu
            src/groupby/sort/groupby.cu
            src/groupby/sort/sort_helper.cu
            src/groupby/sort/group_sum.cu
            src/groupby/sort/group_min.cu
            src/groupby/sort/group_max.cu
            src/groupby/sort/group_argmax.cu
            src/groupby/sort/group_argmin.cu
            src/groupby/sort/group_count.cu
            src/groupby/sort/group_nunique.cu
            src/groupby/sort/group_nth_element.cu
            src/groupby/sort/group_std.cu
            src/groupby/sort/group_quantiles.cu
            src/groupby/sort/group_collect.cu
            src/aggregation/aggregation.cpp
            src/aggregation/aggregation.cu
            src/aggregation/result_cache.cpp
            src/ast/transform.cu
            src/ast/linearizer.cpp
            src/utilities/default_stream.cpp
)

# Override RPATH for cudf
set_target_properties(cudf PROPERTIES BUILD_RPATH "\$ORIGIN")

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

# Creates executable stringify and uses it to convert types.h to c-str for use in JIT code
add_executable(stringify "${JITIFY_INCLUDE_DIR}/stringify.cpp")
execute_process(WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/include/jit
        ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/details
        ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt
        ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include)

add_custom_command(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
                   COMMENT "Stringify headers for use in JIT compiled code"
                   DEPENDS stringify
                   OUTPUT ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                          ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/bit.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/details/__config.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/limits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cfloat.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/chrono.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ctime.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ratio.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/type_traits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/version.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cmath.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cassert.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__config.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__undef_macros.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cfloat.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/chrono.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ctime.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/limits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ratio.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/type_traits.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cmath.jit
                          ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cassert.jit
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.h
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/utilities/bit.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/wrappers/timestamps.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/fixed_point/fixed_point.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/wrappers/durations.hpp
                                   ${LIBCUDACXX_INCLUDE_DIR}/details/__config
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/limits
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/cfloat
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/chrono
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/ctime
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/ratio
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/type_traits
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/version
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/cmath
                                   ${LIBCUDACXX_INCLUDE_DIR}/simt/cassert
                                   ${LIBCXX_INCLUDE_DIR}/__config
                                   ${LIBCXX_INCLUDE_DIR}/__undef_macros
                                   ${LIBCXX_INCLUDE_DIR}/cfloat
                                   ${LIBCXX_INCLUDE_DIR}/chrono
                                   ${LIBCXX_INCLUDE_DIR}/ctime
                                   ${LIBCXX_INCLUDE_DIR}/limits
                                   ${LIBCXX_INCLUDE_DIR}/ratio
                                   ${LIBCXX_INCLUDE_DIR}/type_traits
                                   ${LIBCXX_INCLUDE_DIR}/cmath
                                   ${LIBCXX_INCLUDE_DIR}/cassert

                   # stringified headers are placed underneath the bin include jit directory and end in ".jit"
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.h > ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.hpp > ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/utilities/bit.hpp > ${CMAKE_BINARY_DIR}/include/bit.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ../src/rolling/rolling_jit_detail.hpp > ${CMAKE_BINARY_DIR}/include/rolling_jit_detail.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/wrappers/timestamps.hpp > ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/fixed_point/fixed_point.hpp > ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/wrappers/durations.hpp > ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/details/__config libcudacxx_details_config > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/details/__config.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/limits libcudacxx_simt_limits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/limits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/cfloat libcudacxx_simt_cfloat > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cfloat.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/chrono libcudacxx_simt_chrono > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/chrono.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/ctime libcudacxx_simt_ctime > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ctime.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/ratio libcudacxx_simt_ratio > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ratio.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/type_traits libcudacxx_simt_type_traits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/type_traits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/cmath libcudacxx_simt_cmath > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cmath.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/simt/cassert libcudacxx_simt_cassert > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cassert.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCUDACXX_INCLUDE_DIR}/version libcudacxx_simt_version > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/version.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/__config libcxx_config > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__config.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/__undef_macros libcxx_undef_macros > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__undef_macros.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/cfloat libcxx_cfloat > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cfloat.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/chrono libcxx_chrono > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/chrono.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/ctime libcxx_ctime > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ctime.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/limits libcxx_limits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/limits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/ratio libcxx_ratio > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ratio.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/type_traits libcxx_type_traits > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/type_traits.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/cmath libcxx_cmath > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cmath.jit
                   COMMAND ${CMAKE_BINARY_DIR}/stringify ${LIBCXX_INCLUDE_DIR}/cassert libcxx_cassert > ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cassert.jit
                   )

add_custom_target(stringify_run DEPENDS
                  ${CMAKE_BINARY_DIR}/include/jit/types.h.jit
                  ${CMAKE_BINARY_DIR}/include/jit/types.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/bit.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/timestamps.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/fixed_point.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/durations.hpp.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/details/__config.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/limits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cfloat.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/chrono.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ctime.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/ratio.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/type_traits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cmath.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/cassert.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/simt/version.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__config.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/__undef_macros.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cfloat.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/chrono.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ctime.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/limits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/ratio.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/type_traits.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cmath.jit
                  ${CMAKE_BINARY_DIR}/include/jit/libcudacxx/libcxx/include/cassert.jit
                  )

add_dependencies(cudf stringify_run)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
else()
    # Makes NVTX APIs no-ops
    add_compile_definitions(NVTX_DISABLE)
endif(USE_NVTX)

option(HT_DEFAULT_ALLOCATOR "Use the default allocator for hash tables" ON)
if(HT_DEFAULT_ALLOCATOR)
    message(STATUS "Using default allocator for hash tables")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DHT_DEFAULT_ALLOCATOR")
endif(HT_DEFAULT_ALLOCATOR)

###################################################################################################
# - rmm logging level -----------------------------------------------------------------------------

set(RMM_LOGGING_LEVEL "INFO" CACHE STRING "Choose the logging level.")
# Set the possible values of build type for cmake-gui
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS
        "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

target_compile_definitions(cudf PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL})

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

# link targets for cuDF
target_link_libraries(cudf arrow arrow_cuda nvrtc ${CUDART_LIBRARY} cuda ${ZLIB_LIBRARIES} ${Boost_LIBRARIES})

###################################################################################################
# - install targets -------------------------------------------------------------------------------

# install targets for cuDF
install(TARGETS cudf
        DESTINATION lib
        COMPONENT cudf)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf
        DESTINATION include
        COMPONENT cudf)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf_test
        DESTINATION include
        COMPONENT cudf)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/libcxx
        DESTINATION include/libcudf
        COMPONENT cudf)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/libcudacxx
        DESTINATION include/libcudf
        COMPONENT cudf)

add_custom_target(install_cudf
                  COMMAND "${CMAKE_COMMAND}" -DCOMPONENT=cudf -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
                  DEPENDS cudf)

if(BUILD_TESTS)
    add_dependencies(install_cudf cudftestutil)
endif(BUILD_TESTS)

add_custom_target(build_tests_cudf
                  DEPENDS ${CUDF_TEST_LIST})

add_custom_target(test_cudf
                  COMMAND ctest
                  DEPENDS build_tests_cudf)

add_custom_target(build_benchmarks_cudf
                  DEPENDS ${BENCHMARK_LIST})

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

# doc targets for cuDF
add_custom_command(OUTPUT CUDF_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(docs_cudf DEPENDS CUDF_DOXYGEN)
