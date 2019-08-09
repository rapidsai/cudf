cmake_minimum_required(VERSION 3.12)

include(ExternalProject)

ExternalProject_Add(
        librdkafka-project
        GIT_REPOSITORY "https://github.com/edenhill/librdkafka.git"
        GIT_TAG "v1.0.1"
        SOURCE_DIR        "${LIBRDKAFKA_ROOT}/librdkafka"
        BINARY_DIR        "${LIBRDKAFKA_ROOT}/build"
        INSTALL_DIR       "${LIBRDKAFKA_ROOT}/install"
        CMAKE_ARGS ${PASSTHROUGH_CMAKE_ARGS}
        "-DWITH_SASL=OFF"
        "-DOPENSSL_VERSION=1.0.2"
        "-DRDKAFKA_BUILD_STATIC=ON"
        "-DRDKAFKA_BUILD_EXAMPLES=OFF"
        "-DRDKAFKA_BUILD_TESTS=OFF"
        "-DENABLE_LZ4_EXT=OFF"
        "-DWITH_ZSTD=OFF"
        "-DCMAKE_INSTALL_PREFIX=${LIBRDKAFKA_ROOT}/install"
        "-DCMAKE_C_FLAGS=${CURL_C_FLAGS}"
        "-DCMAKE_INSTALL_LIBDIR=lib"
        "-DCMAKE_CXX_FLAGS=${CURL_CXX_FLAGS}"
)