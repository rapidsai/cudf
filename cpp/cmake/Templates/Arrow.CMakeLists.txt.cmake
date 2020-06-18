cmake_minimum_required(VERSION 3.14)

include(ExternalProject)

ExternalProject_Add(Arrow
    GIT_REPOSITORY    https://github.com/apache/arrow.git
    GIT_TAG           apache-arrow-0.17.1
    SOURCE_DIR        "${ARROW_ROOT}/arrow"
    SOURCE_SUBDIR     "cpp"
    BINARY_DIR        "${ARROW_ROOT}/build"
    INSTALL_DIR       "${ARROW_ROOT}/install"
    CMAKE_ARGS        ${ARROW_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${ARROW_ROOT}/install)
