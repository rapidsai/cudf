#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

find_package(Threads REQUIRED)

add_library(cudf_datagen STATIC common/generate_benchmark_input.cpp)
target_compile_features(cudf_datagen PUBLIC cxx_std_17 cuda_std_17)

target_compile_options(cudf_datagen
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                   "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")

target_link_libraries(cudf_datagen
               PUBLIC GTest::gmock
                      GTest::gtest
                      GTest::gmock_main
                      GTest::gtest_main
                      benchmark::benchmark
                      Threads::Threads
                      cudf)

target_include_directories(cudf_datagen
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
           "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
           "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>")

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

# Use an OBJECT library so we only compile these helper source files only once
add_library(cudf_benchmark_common OBJECT
    "${CUDF_SOURCE_DIR}/tests/utilities/base_fixture.cpp"
    synchronization/synchronization.cpp
    io/cuio_benchmark_common.cpp)
target_link_libraries(cudf_benchmark_common PRIVATE cudf_datagen)

function(ConfigureBench CMAKE_BENCH_NAME)
    add_executable(${CMAKE_BENCH_NAME} ${ARGN})
    set_target_properties(${CMAKE_BENCH_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/gbenchmarks>")
    target_link_libraries(${CMAKE_BENCH_NAME}
        PRIVATE cudf_benchmark_common cudf_datagen benchmark::benchmark_main)
endfunction()

###################################################################################################
# - column benchmarks -----------------------------------------------------------------------------
ConfigureBench(COLUMN_CONCAT_BENCH column/concatenate_benchmark.cpp)

###################################################################################################
# - gather benchmark ------------------------------------------------------------------------------
ConfigureBench(GATHER_BENCH copying/gather_benchmark.cu)

###################################################################################################
# - scatter benchmark -----------------------------------------------------------------------------
ConfigureBench(SCATTER_BENCH copying/scatter_benchmark.cu)

###################################################################################################
# - lists scatter benchmark -----------------------------------------------------------------------
ConfigureBench(SCATTER_LISTS_BENCH lists/copying/scatter_lists_benchmark.cu)

###################################################################################################
# - contiguous_split benchmark  -------------------------------------------------------------------
ConfigureBench(CONTIGUOUS_SPLIT_BENCH copying/contiguous_split_benchmark.cu)

###################################################################################################
# - shift benchmark -------------------------------------------------------------------------------
ConfigureBench(SHIFT_BENCH copying/shift_benchmark.cu)

###################################################################################################
# - transpose benchmark ---------------------------------------------------------------------------
ConfigureBench(TRANSPOSE_BENCH transpose/transpose_benchmark.cu)

###################################################################################################
# - apply_boolean_mask benchmark ------------------------------------------------------------------
ConfigureBench(APPLY_BOOLEAN_MASK_BENCH stream_compaction/apply_boolean_mask_benchmark.cpp)

###################################################################################################
# - stream_compaction benchmark -------------------------------------------------------------------
ConfigureBench(STREAM_COMPACTION_BENCH stream_compaction/drop_duplicates_benchmark.cpp)

###################################################################################################
# - join benchmark --------------------------------------------------------------------------------
ConfigureBench(JOIN_BENCH join/join_benchmark.cu join/conditional_join_benchmark.cu)

###################################################################################################
# - iterator benchmark ----------------------------------------------------------------------------
ConfigureBench(ITERATOR_BENCH iterator/iterator_benchmark.cu)

###################################################################################################
# - search benchmark ------------------------------------------------------------------------------
ConfigureBench(SEARCH_BENCH search/search_benchmark.cpp)

###################################################################################################
# - sort benchmark --------------------------------------------------------------------------------
ConfigureBench(SORT_BENCH
  sort/rank_benchmark.cpp
  sort/sort_benchmark.cpp
  sort/sort_strings_benchmark.cpp)

###################################################################################################
# - quantiles benchmark --------------------------------------------------------------------------------
ConfigureBench(QUANTILES_BENCH
  quantiles/quantiles_benchmark.cpp)

###################################################################################################
# - type_dispatcher benchmark ---------------------------------------------------------------------
ConfigureBench(TYPE_DISPATCHER_BENCH type_dispatcher/type_dispatcher_benchmark.cu)

###################################################################################################
# - reduction benchmark ---------------------------------------------------------------------------
ConfigureBench(REDUCTION_BENCH
  reduction/anyall_benchmark.cpp
  reduction/dictionary_benchmark.cpp
  reduction/reduce_benchmark.cpp
  reduction/scan_benchmark.cpp
  reduction/minmax_benchmark.cpp)

###################################################################################################
# - reduction benchmark ---------------------------------------------------------------------------
ConfigureBench(REPLACE_BENCH
  replace/clamp_benchmark.cpp)

###################################################################################################
# - filling benchmark -----------------------------------------------------------------------------
ConfigureBench(FILL_BENCH
  filling/repeat_benchmark.cpp)

###################################################################################################
# - groupby benchmark -----------------------------------------------------------------------------
ConfigureBench(GROUPBY_BENCH
  groupby/group_sum_benchmark.cu
  groupby/group_nth_benchmark.cu
  groupby/group_shift_benchmark.cu)

###################################################################################################
# - hashing benchmark -----------------------------------------------------------------------------
ConfigureBench(HASHING_BENCH
  hashing/hash_benchmark.cpp
  hashing/partition_benchmark.cpp)

###################################################################################################
# - merge benchmark -------------------------------------------------------------------------------
ConfigureBench(MERGE_BENCH merge/merge_benchmark.cpp)

###################################################################################################
# - null_mask benchmark ---------------------------------------------------------------------------
ConfigureBench(NULLMASK_BENCH null_mask/set_null_mask_benchmark.cpp)

###################################################################################################
# - parquet writer chunks benchmark ---------------------------------------------------------------
ConfigureBench(PARQUET_WRITER_CHUNKS_BENCH io/parquet/parquet_writer_chunks_benchmark.cpp)

###################################################################################################
# - parquet reader benchmark ----------------------------------------------------------------------
ConfigureBench(PARQUET_READER_BENCH io/parquet/parquet_reader_benchmark.cpp)

###################################################################################################
# - orc reader benchmark --------------------------------------------------------------------------
ConfigureBench(ORC_READER_BENCH io/orc/orc_reader_benchmark.cpp)

###################################################################################################
# - csv reader benchmark --------------------------------------------------------------------------
ConfigureBench(CSV_READER_BENCH io/csv/csv_reader_benchmark.cpp)

###################################################################################################
# - parquet writer benchmark ----------------------------------------------------------------------
ConfigureBench(PARQUET_WRITER_BENCH io/parquet/parquet_writer_benchmark.cpp)

###################################################################################################
# - orc writer benchmark --------------------------------------------------------------------------
ConfigureBench(ORC_WRITER_BENCH io/orc/orc_writer_benchmark.cpp)

###################################################################################################
# - csv writer benchmark --------------------------------------------------------------------------
ConfigureBench(CSV_WRITER_BENCH io/csv/csv_writer_benchmark.cpp)

###################################################################################################
# - ast benchmark ---------------------------------------------------------------------------------
ConfigureBench(AST_BENCH ast/transform_benchmark.cpp)

###################################################################################################
# - binaryop benchmark ----------------------------------------------------------------------------
ConfigureBench(BINARYOP_BENCH
  binaryop/binaryop_benchmark.cpp
  binaryop/jit_binaryop_benchmark.cpp)

###################################################################################################
# - nvtext benchmark -------------------------------------------------------------------
ConfigureBench(TEXT_BENCH
  text/ngrams_benchmark.cpp
  text/normalize_benchmark.cpp
  text/normalize_spaces_benchmark.cpp
  text/replace_benchmark.cpp
  text/subword_benchmark.cpp
  text/tokenize_benchmark.cpp)

###################################################################################################
# - strings benchmark -------------------------------------------------------------------
ConfigureBench(STRINGS_BENCH
  string/case_benchmark.cpp
  string/combine_benchmark.cpp
  string/contains_benchmark.cpp
  string/convert_datetime_benchmark.cpp
  string/convert_durations_benchmark.cpp
  string/convert_fixed_point_benchmark.cpp
  string/convert_numerics_benchmark.cpp
  string/copy_benchmark.cpp
  string/extract_benchmark.cpp
  string/factory_benchmark.cu
  string/filter_benchmark.cpp
  string/find_benchmark.cpp
  string/replace_benchmark.cpp
  string/replace_re_benchmark.cpp
  string/split_benchmark.cpp
  string/substring_benchmark.cpp
  string/translate_benchmark.cpp
  string/url_decode_benchmark.cpp)

###################################################################################################
# - json benchmark -------------------------------------------------------------------
ConfigureBench(JSON_BENCH
  string/json_benchmark.cpp)
