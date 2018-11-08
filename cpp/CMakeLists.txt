cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDA_DATAFRAME VERSION 0.3.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQURIED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# suppress SHFL warnings caused by Modern GPU
# TODO: remove this when Modern GPU is removed or fixed to use shfl_sync
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")

# set warnings as errors
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror")

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
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/modules/tests)
else()
    message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
endif(GTEST_FOUND)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${ARROW_INCLUDEDIR}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "${CMAKE_CURRENT_SOURCE_DIR}/modules"
                    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/moderngpu/src"
                    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cnmem/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf SHARED
            modules/comm/ipc/ipc.cu
            modules/dataframe/column.cpp
            modules/dataframe/context.cpp
            modules/error/error_handles.cpp
            modules/filter/filter_ops.cu
            modules/join/joining.cu
            modules/sort/segmented_sorting.cu
            modules/sort/sorting.cu
            modules/sqls/sqls_ops.cu
            modules/binary/binary_ops.cu
            modules/bitmask/bitmask_ops.cu
            modules/bitmask/valid_ops.cu
            modules/compaction/stream_compaction_ops.cu
            modules/datetime/datetime_ops.cu
            modules/hash/hashing.cu
            modules/hash/hash_ops.cu
            modules/quantiles/quantiles.cu
            modules/reductions/reductions.cu
            modules/unary/unary_ops.cu
            # modules/windowed/windowed_ops.cu ... this is broken
            modules/io/convert/csr/cudf_to_csr.cu
            modules/io/csv/csv_reader.cu
            modules/utilities/cuda_utils.cu
            modules/utilities/scan.cu
            modules/utilities/nvtx/nvtx_utils.cpp)

add_library(rmm SHARED
            modules/rmm/memory.cpp
            modules/rmm/memory_manager.cpp
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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(rmm cudart cuda)
target_link_libraries(cudf arrow rmm)

###################################################################################################
# - python cffi bindings --------------------------------------------------------------------------

add_custom_command(OUTPUT PYTHON_CFFI
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/../python ${CMAKE_BINARY_DIR}/python
                   COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/python python setup.py build_ext --inplace
                   VERBATIM)

add_custom_target(python_cffi DEPENDS PYTHON_CFFI cudf)



