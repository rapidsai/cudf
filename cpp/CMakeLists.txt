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
# TODO: remove this when Modern GPU is removed for fixed to use shfl_sync
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")

# set warnings as errors
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf SHARED
            source/comm/ipc/ipc.cu
            source/core/dataframe/column.cpp
            source/core/dataframe/context.cpp
            source/core/error/error_handles.cpp
            source/core/filter/filterops.cu
            source/core/join/joining.cu
            source/core/sort/segmented_sorting.cu
            source/core/sort/sorting.cu
            source/core/sqls/sqls_ops.cu
            source/functions/binary/binary_ops.cu
            source/functions/bitmask/src/bitmask_ops.cu
            source/functions/bitmask/src/valid_ops.cu
            source/functions/compaction/stream_compaction_ops.cu
            source/functions/datetime/datetime_ops.cu
            source/functions/hash/src/hashing.cu
            source/functions/hash/src/hash_ops.cu
            source/functions/reductions/reductions.cu
            source/functions/unary/unary_ops.cu
            source/functions/windowed/windowed_ops.cu
            source/functions/windowed/quantiles/quantiles.cu
            source/io/convert/csr/cudf_to_csr.cu
            source/io/csv/src/csv_reader.cu)

add_library(rmm SHARED
            source/rmm/src/memory.cpp
            source/rmm/src/memory_manager.cpp
            thirdparty/cnmem/src/cnmem.cpp
            )

target_link_libraries(rmm cuda)

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

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif(CMAKE_BUILD_TYPE MATCHES DEBUG)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

# include(FeatureSummary)
# include(CheckIncludeFiles)
# include(CheckLibraryExists)
# include(CTest)
# 
# include(ConfigureGoogleTest)
# include(ConfigureArrow)

message(STATUS " " ${CMAKE_CUDA_FLAGS})

