cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDA_DATAFRAME LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQURIED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# suppress SHFL warnings caused by Modern GPU
# TODO: remove this when Modern GPU is removed or fixed to use shfl_sync
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")

# set warnings as errors
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf SHARED
            src/comm/ipc/ipc.cu
            src/core/dataframe/column.cpp
            src/core/dataframe/context.cpp
            src/core/error/error_handles.cpp
            src/core/filter/filter_ops.cu
            src/core/join/joining.cu
            src/core/sort/segmented_sorting.cu
            src/core/sort/sorting.cu
            src/core/sqls/sqls_ops.cu
            src/functions/binary/binary_ops.cu
            src/functions/bitmask/src/bitmask_ops.cu
            src/functions/bitmask/src/valid_ops.cu
            src/functions/compaction/stream_compaction_ops.cu
            src/functions/datetime/datetime_ops.cu
            src/functions/hash/hashing.cu
            src/functions/hash/hash_ops.cu
            src/functions/quantiles/quantiles.cu
            src/functions/reductions/reductions.cu
            src/functions/unary/unary_ops.cu
            # src/functions/windowed/windowed_ops.cu ... this is broken
            src/io/convert/csr/cudf_to_csr.cu
            src/io/csv/csv_reader.cu
            src/utilities/cuda_utils.cu
            src/utilities/scan.cu
            src/utilities/nvtx/nvtx_utils.cpp)

add_library(rmm SHARED
            src/rmm/src/memory.cpp
            src/rmm/src/memory_manager.cpp
            thirdparty/cnmem/src/cnmem.cpp
            )

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ENV LD_LIBRARY_PATH)
    target_link_libraries(cudf ${NVTX_LIBRARY})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --define-macro USE_NVTX")
endif(USE_NVTX)

option(HT_LEGACY_ALLOCATOR "Use the legacy allocator for hash tables" ON)
if(HT_LEGACY_ALLOCATOR)
    message(STATUS "Using legacy allocator for hash tables")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --define-macro HT_LEGACY_ALLOCATOR")
endif(HT_LEGACY_ALLOCATOR)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

include(ConfigureGoogleTest)
include(ConfigureArrow)

###################################################################################################
# - arrow -----------------------------------------------------------------------------------------

# Locate the Apache Arrow package (Requires that you use ConfigureArrow module)
find_package(Arrow REQUIRED)

message(STATUS "ARROW_ROOT: " ${ARROW_ROOT})

set_package_properties(
    Arrow PROPERTIES TYPE REQUIRED
    PURPOSE "Apache Arrow is a cross-language development platform for in-memory data."
    URL "https://github.com/apache/arrow")

###################################################################################################
# - gtest -----------------------------------------------------------------------------------------

find_package(GTest)
set_package_properties(GTest PROPERTIES TYPE OPTIONAL
    PURPOSE "Google C++ Testing Framework (Google Test)."
    URL "https://github.com/google/googletest")
if(GTEST_FOUND)
    message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/tests)
else()
    message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
endif(GTEST_FOUND)


###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${ARROW_INCLUDEDIR}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/src"
                    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/moderngpu/src"
                    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cnmem/include")

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(rmm cuda)
target_link_libraries(cudf arrow rmm)
