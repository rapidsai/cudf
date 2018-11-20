cmake_minimum_required(VERSION 3.12)

include(ExternalProject)

set(ARROW_CMAKE_ARGS
    #Arrow dependencies
    -DARROW_WITH_LZ4=OFF
    -DARROW_WITH_ZSTD=OFF
    -DARROW_WITH_BROTLI=OFF
    -DARROW_WITH_SNAPPY=OFF
    -DARROW_WITH_ZLIB=OFF

    #Build settings
    -DARROW_BUILD_STATIC=ON
    -DARROW_BUILD_SHARED=OFF
    -DARROW_BOOST_USE_SHARED=ON
    -DARROW_BUILD_TESTS=OFF
    -DARROW_TEST_MEMCHECK=OFF
    -DARROW_BUILD_BENCHMARKS=OFF

    #Arrow modules
    -DARROW_IPC=ON
    -DARROW_COMPUTE=OFF
    -DARROW_GPU=OFF
    -DARROW_JEMALLOC=OFF
    -DARROW_BOOST_VENDORED=OFF
    -DARROW_PYTHON=OFF
)

ExternalProject_Add(Arrow
                    GIT_REPOSITORY    https://github.com/apache/arrow.git
                    GIT_TAG           apache-arrow-0.10.0
                    SOURCE_DIR        "${ARROW_ROOT}/arrow"
                    SOURCE_SUBDIR     "cpp"
                    BINARY_DIR        "${ARROW_ROOT}/build"
                    INSTALL_DIR       "${ARROW_ROOT}/install"
                    CMAKE_ARGS        ${ARROW_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${ARROW_ROOT}/install)







