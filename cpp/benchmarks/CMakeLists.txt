cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(CUDF_BENCHS LANGUAGES C CXX CUDA)

add_library(cudf_datagen STATIC
                   "${CMAKE_SOURCE_DIR}/benchmarks/common/generate_benchmark_input.cpp"
)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

set(BENCHMARK_LIST CACHE INTERNAL "BENCHMARK_LIST")

function(ConfigureBench CMAKE_BENCH_NAME CMAKE_BENCH_SRC)
    add_executable(${CMAKE_BENCH_NAME}
                   ${CMAKE_BENCH_SRC}
                   "${CMAKE_CURRENT_SOURCE_DIR}/synchronization/synchronization.cpp"
                   "${CMAKE_SOURCE_DIR}/tests/utilities/base_fixture.cpp")
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_BENCH_NAME} benchmark benchmark_main pthread cudf_datagen cudf)
    set_target_properties(${CMAKE_BENCH_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gbenchmarks")
    set(BENCHMARK_LIST ${BENCHMARK_LIST} ${CMAKE_BENCH_NAME} CACHE INTERNAL "BENCHMARK_LIST")
endfunction(ConfigureBench)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${THRUST_INCLUDE_DIR}"
                    "${JITIFY_INCLUDE_DIR}"
                    "${LIBCUDACXX_INCLUDE_DIR}")

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    "${GTEST_INCLUDE_DIR}"
                    "${GBENCH_INCLUDE_DIR}"
                    "${ARROW_INCLUDE_DIR}"
                    "${RMM_INCLUDE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${GBENCH_LIBRARY_DIR}"
                 "${RMM_LIBRARY}")

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
  "${CMAKE_CURRENT_SOURCE_DIR}/sort/sort_benchmark.cu")

ConfigureBench(SORT_BENCH "${SORT_BENCH_SRC}")

###################################################################################################
# - type_dispatcher benchmark ---------------------------------------------------------------------

set(TD_BENCH_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/type_dispatcher/type_dispatcher_benchmark.cu")

ConfigureBench(TYPE_DISPATCHER_BENCH "${TD_BENCH_SRC}")

###################################################################################################
# - reduction benchmark ---------------------------------------------------------------------------

set(REDUCTION_BENCH_SRC
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
