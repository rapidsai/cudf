###################################################################################################
# - find arrow ------------------------------------------------------------------------------------

if(ARROW_STATIC_LIB)
  message(STATUS "BUILDING ARROW")
  CPMFindPackage(
    NAME Arrow
    GITHUB_REPOSITORY google/googletest
    GIT_TAG apache-arrow-1.0.1
    VERSION 1.0.1
    GIT_SHALLOW TRUE
    OPTIONS
      "ARROW_WITH_LZ4 OFF"
      "ARROW_WITH_ZSTD OFF"
      "ARROW_WITH_BROTLI OFF"
      "ARROW_WITH_SNAPPY OFF"
      "ARROW_WITH_ZLIB OFF"
      "ARROW_BUILD_STATIC ON"
      "ARROW_BUILD_SHARED OFF"
      "ARROW_BOOST_USE_SHARED OFF"
      "ARROW_BUILD_TESTS OFF"
      "ARROW_TEST_MEMCHECK OFF"
      "ARROW_BUILD_BENCHMARKS OFF"
      "ARROW_IPC ON"
      "ARROW_FLIGHT OFF"
      "ARROW_COMPUTE OFF"
      "ARROW_CUDA ON"
      "ARROW_JEMALLOC OFF"
      "ARROW_BOOST_VENDORED OFF"
      "ARROW_PYTHON OFF"
      "ARROW_USE_GLOG OFF"
      "ARROW_DATASET ON"
      "ARROW_BUILD_UTILITIES OFF"
      "ARROW_HDFS OFF"
      "CMAKE_VERBOSE_MAKEFILE ON"
  )
  
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
# - find zlib -------------------------------------------------------------------------------------

CPMFindPackage(
  NAME ZLIB
  GITHUB_REPOSITORY madler/zlib
  VERSION 1.2.11
)

message(STATUS "ZLIB: ZLIB_LIBRARY set to ${ZLIB_LIBRARY}")
message(STATUS "ZLIB: ZLIB_INCLUDE_DIR set to ${ZLIB_INCLUDE_DIR}")

if(ZLIB_INCLUDE_DIR)
    message(STATUS "ZLib found in ${ZLIB_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "ZLib not found, please check your settings.")
endif(ZLIB_INCLUDE_DIR)

###################################################################################################
# - find boost ------------------------------------------------------------------------------------

CPMFindPackage(
  NAME Boost
  GITHUB_REPOSITORY Orphis/boost-cmake
  GIT_TAG 60fb5fa
  VERSION 1.70.0
  FIND_PACKAGE_ARGUMENTS "COMPONENTS filesystem"
)
if(Boost_INCLUDE_DIR)
    message(STATUS "Boost found in ${Boost_INCLUDE_DIR}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DBOOST_NO_CXX14_CONSTEXPR")
else()
    message(FATAL_ERROR "Boost not found, please check your settings.")
endif(Boost_INCLUDE_DIR)

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

# FindCUDA doesn't quite set the variables it should. RMM needs this to be set to build from CPM
set(CUDAToolkit_INCLUDE_DIR "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

CPMFindPackage(
  NAME RMM
  GITHUB_REPOSITORY rapidsai/rmm
  VERSION ${CMAKE_PROJECT_VERSION}
  GIT_TAG "branch-${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}"
)

###################################################################################################
# - DLPACK -------------------------------------------------------------------------------------------

CPMFindPackage(
  NAME DLPACK
  GITHUB_REPOSITORY dmlc/dlpack
  VERSION 0.3
)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
  CPMFindPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG release-1.10.0
    VERSION 1.10.0
    GIT_SHALLOW TRUE
    OPTIONS
      "INSTALL_GTEST OFF"
    # googletest >= 1.10.0 provides a cmake config file -- use it if it exists
    FIND_PACKAGE_ARGUMENTS "CONFIG"
    )
  include_directories("${gtest_SOURCE_DIR}/include"
                      "${gmock_SOURCE_DIR}/include")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
  message(STATUS "CUDF_TEST_LIST set to: ${CUDF_TEST_LIST}")
endif(BUILD_TESTS)

###################################################################################################
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)
  CPMFindPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    VERSION 1.5.1
    OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
    # The REGEX feature test fails when gbench's cmake is run under CPM because it doesn't assume C++11
    # Additionally, attempting to set the CMAKE_CXX_VERSION here doesn't propogate to the feature test build
    # Therefore, we just disable the feature test and assume platforms we care about have a regex impl available
    "RUN_HAVE_GNU_POSIX_REGEX 0" #
  )
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
  message(STATUS "BENCHMARK_LIST set to: ${BENCHMARK_LIST}")
endif(BUILD_BENCHMARKS)
