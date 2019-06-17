cmake_minimum_required(VERSION 3.12)

include(ExternalProject)

ExternalProject_Add(GoogleBenchmark
                    GIT_REPOSITORY    https://github.com/google/benchmark.git
                    GIT_TAG           master 
                    SOURCE_DIR        "${GBENCH_ROOT}/googlebenchmark"
                    BINARY_DIR        "${GBENCH_ROOT}/build"
                    INSTALL_DIR		    "${GBENCH_ROOT}/install"
                    CMAKE_ARGS        ${GBENCH_CMAKE_ARGS} -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=${GBENCH_ROOT}/install)
                    # The flag BENCHMARK_ENABLE_TESTING=OFF prevents Google Benchmark from asking for Google Test.







