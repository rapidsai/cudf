# =============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

find_package(Threads REQUIRED)

add_library(cudf_datagen STATIC common/generate_input.cu)
target_compile_features(cudf_datagen PUBLIC cxx_std_17 cuda_std_17)

target_compile_options(
  cudf_datagen PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_link_libraries(
  cudf_datagen
  PUBLIC GTest::gmock GTest::gtest benchmark::benchmark nvbench::nvbench Threads::Threads cudf
         cudftestutil
  PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>
)

target_include_directories(
  cudf_datagen
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>" "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
         "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
)

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# Use an OBJECT library so we only compile these helper source files only once
add_library(
  cudf_benchmark_common OBJECT "${CUDF_SOURCE_DIR}/tests/utilities/base_fixture.cpp"
                               synchronization/synchronization.cpp io/cuio_common.cpp
)
target_link_libraries(cudf_benchmark_common PRIVATE cudf_datagen $<TARGET_NAME_IF_EXISTS:conda_env>)
add_custom_command(
  OUTPUT CUDF_BENCHMARKS
  COMMAND echo Running benchmarks
  COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running cudf benchmarks."
  USES_TERMINAL
)

# This function takes in a benchmark name and benchmark source and handles setting all of the
# associated properties and linking to build the benchmark
function(ConfigureBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/benchmarks>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using cuda::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    ${CMAKE_BENCH_NAME} PRIVATE cudf_benchmark_common cudf_datagen benchmark::benchmark_main
                                $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  add_custom_command(
    OUTPUT CUDF_BENCHMARKS
    COMMAND ${CMAKE_BENCH_NAME} --benchmark_out_format=json
            --benchmark_out=results/${CMAKE_BENCH_NAME}.json
    APPEND
    COMMENT "Adding ${CMAKE_BENCH_NAME}"
  )

  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/benchmarks/libcudf
    EXCLUDE_FROM_ALL
  )
endfunction()

# This function takes in a benchmark name and benchmark source for nvbench benchmarks and handles
# setting all of the associated properties and linking to build the benchmark
function(ConfigureNVBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN} fixture/nvbench_main.cpp)
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/benchmarks>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )
  target_link_libraries(
    ${CMAKE_BENCH_NAME} PRIVATE cudf_benchmark_common cudf_datagen nvbench::nvbench
                                $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/benchmarks/libcudf
    EXCLUDE_FROM_ALL
  )
endfunction()

# ##################################################################################################
# * column benchmarks -----------------------------------------------------------------------------
ConfigureBench(COLUMN_CONCAT_BENCH column/concatenate.cpp)

# ##################################################################################################
# * gather benchmark ------------------------------------------------------------------------------
ConfigureBench(GATHER_BENCH copying/gather.cu)

# ##################################################################################################
# * scatter benchmark -----------------------------------------------------------------------------
ConfigureBench(SCATTER_BENCH copying/scatter.cu)

# ##################################################################################################
# * lists scatter benchmark -----------------------------------------------------------------------
ConfigureBench(SCATTER_LISTS_BENCH lists/copying/scatter_lists.cu)

# ##################################################################################################
# * Other list-related operartions benchmark ------------------------------------------------------
ConfigureNVBench(SET_OPS_NVBENCH lists/set_operations.cpp)

# ##################################################################################################
# * contiguous_split benchmark  -------------------------------------------------------------------
ConfigureBench(CONTIGUOUS_SPLIT_BENCH copying/contiguous_split.cu)

# ##################################################################################################
# * shift benchmark -------------------------------------------------------------------------------
ConfigureBench(SHIFT_BENCH copying/shift.cu)

# ##################################################################################################
# * copy-if-else benchmark
# -----------------------------------------------------------------------------
ConfigureBench(COPY_IF_ELSE_BENCH copying/copy_if_else.cpp)

# ##################################################################################################
# * transpose benchmark ---------------------------------------------------------------------------
ConfigureBench(TRANSPOSE_BENCH transpose/transpose.cpp)

# ##################################################################################################
# * apply_boolean_mask benchmark ------------------------------------------------------------------
ConfigureBench(APPLY_BOOLEAN_MASK_BENCH stream_compaction/apply_boolean_mask.cpp)

# ##################################################################################################
# * stream_compaction benchmark -------------------------------------------------------------------
ConfigureNVBench(
  STREAM_COMPACTION_NVBENCH
  stream_compaction/distinct.cpp
  stream_compaction/distinct_count.cpp
  stream_compaction/stable_distinct.cpp
  stream_compaction/unique.cpp
  stream_compaction/unique_count.cpp
)

# ##################################################################################################
# * join benchmark --------------------------------------------------------------------------------
ConfigureBench(JOIN_BENCH join/left_join.cu join/conditional_join.cu)
ConfigureNVBench(JOIN_NVBENCH join/join.cu join/mixed_join.cu)

# ##################################################################################################
# * iterator benchmark ----------------------------------------------------------------------------
ConfigureBench(ITERATOR_BENCH iterator/iterator.cu)

# ##################################################################################################
# * search benchmark ------------------------------------------------------------------------------
ConfigureBench(SEARCH_BENCH search/search.cpp)
ConfigureNVBench(SEARCH_NVBENCH search/contains.cpp)

# ##################################################################################################
# * sort benchmark --------------------------------------------------------------------------------
ConfigureBench(SORT_BENCH sort/rank.cpp sort/sort.cpp sort/sort_strings.cpp)
ConfigureNVBench(
  SORT_NVBENCH sort/rank_lists.cpp sort/rank_structs.cpp sort/segmented_sort.cpp
  sort/sort_lists.cpp sort/sort_structs.cpp
)

# ##################################################################################################
# * quantiles benchmark
# --------------------------------------------------------------------------------
ConfigureBench(QUANTILES_BENCH quantiles/quantiles.cpp)

# ##################################################################################################
# * type_dispatcher benchmark ---------------------------------------------------------------------
ConfigureBench(TYPE_DISPATCHER_BENCH type_dispatcher/type_dispatcher.cu)

# ##################################################################################################
# * reduction benchmark ---------------------------------------------------------------------------
ConfigureBench(
  REDUCTION_BENCH reduction/anyall.cpp reduction/dictionary.cpp reduction/minmax.cpp
  reduction/reduce.cpp reduction/scan.cpp
)
ConfigureNVBench(
  REDUCTION_NVBENCH reduction/rank.cpp reduction/scan_structs.cpp reduction/segmented_reduce.cpp
)

# ##################################################################################################
# * reduction benchmark ---------------------------------------------------------------------------
ConfigureBench(REPLACE_BENCH replace/clamp.cpp replace/nans.cpp)

# ##################################################################################################
# * filling benchmark -----------------------------------------------------------------------------
ConfigureBench(FILL_BENCH filling/repeat.cpp)

# ##################################################################################################
# * groupby benchmark -----------------------------------------------------------------------------
ConfigureBench(
  GROUPBY_BENCH groupby/group_sum.cpp groupby/group_nth.cpp groupby/group_shift.cpp
  groupby/group_struct_values.cpp groupby/group_no_requests.cpp groupby/group_scan.cpp
)

ConfigureNVBench(
  GROUPBY_NVBENCH groupby/group_max.cpp groupby/group_nunique.cpp groupby/group_rank.cpp
  groupby/group_struct_keys.cpp
)

# ##################################################################################################
# * hashing benchmark -----------------------------------------------------------------------------
ConfigureBench(HASHING_BENCH hashing/hash.cpp hashing/partition.cpp)

# ##################################################################################################
# * merge benchmark -------------------------------------------------------------------------------
ConfigureBench(MERGE_BENCH merge/merge.cpp)

# ##################################################################################################
# * null_mask benchmark ---------------------------------------------------------------------------
ConfigureBench(NULLMASK_BENCH null_mask/set_null_mask.cpp)

# ##################################################################################################
# * parquet writer benchmark ----------------------------------------------------------------------
ConfigureNVBench(
  PARQUET_WRITER_NVBENCH io/parquet/parquet_writer.cpp io/parquet/parquet_writer_chunks.cpp
)

# ##################################################################################################
# * parquet reader benchmark ----------------------------------------------------------------------
ConfigureNVBench(
  PARQUET_READER_NVBENCH io/parquet/parquet_reader_input.cpp io/parquet/parquet_reader_options.cpp
)

# ##################################################################################################
# * orc reader benchmark --------------------------------------------------------------------------
ConfigureNVBench(ORC_READER_NVBENCH io/orc/orc_reader_input.cpp io/orc/orc_reader_options.cpp)

# ##################################################################################################
# * csv reader benchmark --------------------------------------------------------------------------
ConfigureNVBench(CSV_READER_NVBENCH io/csv/csv_reader_input.cpp io/csv/csv_reader_options.cpp)

# ##################################################################################################
# * orc writer benchmark --------------------------------------------------------------------------
ConfigureNVBench(ORC_WRITER_NVBENCH io/orc/orc_writer.cpp io/orc/orc_writer_chunks.cpp)

# ##################################################################################################
# * csv writer benchmark --------------------------------------------------------------------------
ConfigureNVBench(CSV_WRITER_NVBENCH io/csv/csv_writer.cpp)

# ##################################################################################################
# * ast benchmark ---------------------------------------------------------------------------------
ConfigureBench(AST_BENCH ast/transform.cpp)

# ##################################################################################################
# * binaryop benchmark ----------------------------------------------------------------------------
ConfigureBench(BINARYOP_BENCH binaryop/binaryop.cpp binaryop/compiled_binaryop.cpp)

# ##################################################################################################
# * nvtext benchmark -------------------------------------------------------------------
ConfigureBench(
  TEXT_BENCH text/ngrams.cpp text/normalize.cpp text/normalize_spaces.cpp text/replace.cpp
  text/subword.cpp text/tokenize.cpp
)

ConfigureNVBench(TEXT_NVBENCH text/minhash.cpp)

# ##################################################################################################
# * strings benchmark -------------------------------------------------------------------
ConfigureBench(
  STRINGS_BENCH
  string/combine.cpp
  string/contains.cpp
  string/convert_datetime.cpp
  string/convert_durations.cpp
  string/convert_fixed_point.cpp
  string/convert_numerics.cpp
  string/copy.cu
  string/extract.cpp
  string/factory.cu
  string/filter.cpp
  string/find.cpp
  string/repeat_strings.cpp
  string/replace.cpp
  string/replace_re.cpp
  string/slice.cpp
  string/split.cpp
  string/translate.cpp
  string/url_decode.cu
)

ConfigureNVBench(
  STRINGS_NVBENCH string/case.cpp string/char_types.cpp string/join_strings.cpp string/lengths.cpp
  string/like.cpp string/reverse.cpp
)

# ##################################################################################################
# * json benchmark -------------------------------------------------------------------
ConfigureBench(JSON_BENCH string/json.cu)
ConfigureNVBench(FST_NVBENCH io/fst.cu)
ConfigureNVBench(JSON_READER_NVBENCH io/json/nested_json.cpp io/json/json_reader_input.cpp)
ConfigureNVBench(JSON_WRITER_NVBENCH io/json/json_writer.cpp)

# ##################################################################################################
# * io benchmark ---------------------------------------------------------------------
ConfigureNVBench(MULTIBYTE_SPLIT_NVBENCH io/text/multibyte_split.cpp)
target_link_libraries(MULTIBYTE_SPLIT_NVBENCH PRIVATE ZLIB::ZLIB)

add_custom_target(
  run_benchmarks
  DEPENDS CUDF_BENCHMARKS
  COMMENT "Custom command for running cudf benchmarks."
)
