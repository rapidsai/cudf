cmake_minimum_required(VERSION 3.14)
project(cudf-Arrow)

include(ExternalProject)

ExternalProject_Add(Arrow
    GIT_REPOSITORY    https://github.com/apache/arrow.git
    GIT_TAG           apache-arrow-1.0.1
    GIT_SHALLOW       true
    SOURCE_DIR        "${ARROW_ROOT}/arrow"
    SOURCE_SUBDIR     "cpp"
    BINARY_DIR        "${ARROW_ROOT}/build"
    INSTALL_DIR       "${ARROW_ROOT}/install"
    CMAKE_ARGS        ${ARROW_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${ARROW_ROOT}/install)
