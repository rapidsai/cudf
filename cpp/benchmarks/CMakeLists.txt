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

# Find or install GoogleTest
CPMFindPackage(NAME GTest
    VERSION         1.10.0
    GIT_REPOSITORY  https://github.com/google/googletest.git
    GIT_TAG         release-1.10.0
    GIT_SHALLOW     TRUE
    OPTIONS         "INSTALL_GTEST OFF"
    # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
    FIND_PACKAGE_ARGUMENTS "CONFIG")

if(GTest_ADDED)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gmock ALIAS gmock)
    add_library(GTest::gtest_main ALIAS gtest_main)
    add_library(GTest::gmock_main ALIAS gmock_main)
endif()

# Find or install GoogleBench
CPMFindPackage(NAME benchmark
    VERSION         1.5.2
    GIT_REPOSITORY  https://github.com/google/benchmark.git
    GIT_TAG         v1.5.2
    GIT_SHALLOW     TRUE
    OPTIONS         "BENCHMARK_ENABLE_TESTING OFF"
                    "BENCHMARK_ENABLE_INSTALL OFF")

add_library(cudf_datagen STATIC "${CUDF_SOURCE_DIR}/benchmarks/common/generate_benchmark_input.cpp")

target_compile_options(cudf_datagen
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_link_libraries(cudf_datagen
               PUBLIC GTest::gmock
                      GTest::gtest
                      GTest::gmock_main
                      GTest::gtest_main
                      benchmark::benchmark
                      benchmark::benchmark_main
                      pthread
                      cudf)

target_include_directories(cudf_datagen
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
           "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
           "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>")

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC}
                   "${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp"
                   "${CUDF_SOURCE_DIR}/tests/utilities/base_fixture.cpp"
                   "${CUDF_SOURCE_DIR}/benchmarks/io/cuio_benchmark_common.cpp")
    target_compile_options(${CMAKE_BENCH_NAME}
                PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
    )
    set_target_properties(${CMAKE_BENCH_NAME}
        PROPERTIES CXX_STANDARD              14
                   CXX_STANDARD_REQUIRED     ON
                   CUDA_STANDARD             14
                   CUDA_STANDARD_REQUIRED    ON
                   RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/gbenchmarks>")
    target_link_libraries(${CMAKE_BENCH_NAME} cudf_datagen)
endfunction(ConfigureBench)

###################################################################################################
# - column benchmarks -----------------------------------------------------------------------------

set(COLUMN_CONCAT_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/column/concatenate_benchmark.cpp")

ConfigureBench(COLUMN_CONCAT_BENCH "${COLUMN_CONCAT_BENCH_SRC}")

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
# - shift benchmark -------------------------------------------------------------------------------

set(SHIFT_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/copying/shift_benchmark.cu")

ConfigureBench(SHIFT_BENCH "${SHIFT_BENCH_SRC}")

###################################################################################################
# - transpose benchmark ---------------------------------------------------------------------------

set(TRANSPOSE_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/transpose/transpose_benchmark.cu")

ConfigureBench(TRANSPOSE_BENCH "${TRANSPOSE_BENCH_SRC}")

###################################################################################################
# - apply_boolean_mask benchmark ------------------------------------------------------------------

set(APPLY_BOOLEAN_MASK_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/apply_boolean_mask_benchmark.cpp")

ConfigureBench(APPLY_BOOLEAN_MASK_BENCH "${APPLY_BOOLEAN_MASK_BENCH_SRC}")

###################################################################################################
# - stream_compaction benchmark -------------------------------------------------------------------

set(STREAM_COMPACTION_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/stream_compaction/drop_duplicates_benchmark.cpp")

ConfigureBench(STREAM_COMPACTION_BENCH "${STREAM_COMPACTION_BENCH_SRC}")

###################################################################################################
# - join benchmark --------------------------------------------------------------------------------

set(JOIN_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/join/join_benchmark.cu")

ConfigureBench(JOIN_BENCH "${JOIN_BENCH_SRC}")

###################################################################################################
# - iterator benchmark ----------------------------------------------------------------------------

set(ITERATOR_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/iterator/iterator_benchmark.cu")

ConfigureBench(ITERATOR_BENCH "${ITERATOR_BENCH_SRC}")

###################################################################################################
# - search benchmark ------------------------------------------------------------------------------

set(SEARCH_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/search/search_benchmark.cu")

ConfigureBench(SEARCH_BENCH "${SEARCH_BENCH_SRC}")

###################################################################################################
# - sort benchmark --------------------------------------------------------------------------------

set(SORT_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/sort/sort_benchmark.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/sort/sort_strings_benchmark.cpp")

ConfigureBench(SORT_BENCH "${SORT_BENCH_SRC}")

###################################################################################################
# - type_dispatcher benchmark ---------------------------------------------------------------------

set(TD_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/type_dispatcher/type_dispatcher_benchmark.cu")

ConfigureBench(TYPE_DISPATCHER_BENCH "${TD_BENCH_SRC}")

###################################################################################################
# - reduction benchmark ---------------------------------------------------------------------------

set(REDUCTION_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/reduction/anyall_benchmark.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/reduction/reduce_benchmark.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/reduction/minmax_benchmark.cpp")

ConfigureBench(REDUCTION_BENCH "${REDUCTION_BENCH_SRC}")

###################################################################################################
# - groupby benchmark -----------------------------------------------------------------------------

set(GROUPBY_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_sum_benchmark.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/groupby/group_nth_benchmark.cu")

ConfigureBench(GROUPBY_BENCH "${GROUPBY_BENCH_SRC}")

###################################################################################################
# - hashing benchmark -----------------------------------------------------------------------------

set(HASHING_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/hashing/hashing_benchmark.cpp")

ConfigureBench(HASHING_BENCH "${HASHING_BENCH_SRC}")

###################################################################################################
# - merge benchmark -------------------------------------------------------------------------------

set(MERGE_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/merge/merge_benchmark.cpp")

ConfigureBench(MERGE_BENCH "${MERGE_BENCH_SRC}")

###################################################################################################
# - null_mask benchmark ---------------------------------------------------------------------------

set(NULLMASK_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/null_mask/set_null_mask_benchmark.cpp")

ConfigureBench(NULLMASK_BENCH "${NULLMASK_BENCH_SRC}")

###################################################################################################
# - parquet writer chunks benchmark ---------------------------------------------------------------

set(PARQUET_WRITER_CHUNKS_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/io/parquet/parquet_writer_chunks_benchmark.cpp")

ConfigureBench(PARQUET_WRITER_CHUNKS_BENCH "${PARQUET_WRITER_CHUNKS_BENCH_SRC}")

###################################################################################################
# - parquet reader benchmark ----------------------------------------------------------------------

set(PARQUET_READER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/io/parquet/parquet_reader_benchmark.cpp")

ConfigureBench(PARQUET_READER_BENCH "${PARQUET_READER_BENCH_SRC}")

###################################################################################################
# - orc reader benchmark --------------------------------------------------------------------------

set(ORC_READER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/io/orc/orc_reader_benchmark.cpp")

ConfigureBench(ORC_READER_BENCH "${ORC_READER_BENCH_SRC}")

###################################################################################################
# - csv reader benchmark --------------------------------------------------------------------------

set(CSV_READER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/io/csv/csv_reader_benchmark.cpp")

ConfigureBench(CSV_READER_BENCH "${CSV_READER_BENCH_SRC}")

###################################################################################################
# - parquet writer benchmark ----------------------------------------------------------------------

set(PARQUET_WRITER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/io/parquet/parquet_writer_benchmark.cpp")

ConfigureBench(PARQUET_WRITER_BENCH "${PARQUET_WRITER_BENCH_SRC}")

###################################################################################################
# - orc writer benchmark --------------------------------------------------------------------------

set(ORC_WRITER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/io/orc/orc_writer_benchmark.cpp")

ConfigureBench(ORC_WRITER_BENCH "${ORC_WRITER_BENCH_SRC}")

###################################################################################################
# - csv writer benchmark --------------------------------------------------------------------------

set(CSV_WRITER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/io/csv/csv_writer_benchmark.cpp")

ConfigureBench(CSV_WRITER_BENCH "${CSV_WRITER_BENCH_SRC}")

###################################################################################################
# - ast benchmark ---------------------------------------------------------------------------------

set(AST_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/ast/transform_benchmark.cpp")

ConfigureBench(AST_BENCH "${AST_BENCH_SRC}")

###################################################################################################
# - binaryop benchmark ----------------------------------------------------------------------------

set(BINARYOP_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/binaryop/binaryop_benchmark.cu")

ConfigureBench(BINARYOP_BENCH "${BINARYOP_BENCH_SRC}")

###################################################################################################
# - subword tokenizer benchmark -------------------------------------------------------------------

set(SUBWORD_TOKENIZER_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/text/subword_benchmark.cpp")

ConfigureBench(SUBWORD_TOKENIZER_BENCH "${SUBWORD_TOKENIZER_BENCH_SRC}")

###################################################################################################
# - convert to string benchmark -------------------------------------------------------------------

set(DURATION_TO_STRING_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/string/convert_durations_benchmark.cpp")

ConfigureBench(DURATION_TO_STRING_BENCH "${DURATION_TO_STRING_BENCH_SRC}")
