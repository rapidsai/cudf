#=============================================================================
# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDA_DATAFRAME VERSION 0.4.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# suppress SHFL warnings caused by Modern GPU
# TODO: remove this when Modern GPU is removed or fixed to use shfl_sync
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")

# set warnings as errors
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")

# set default build type
set(CMAKE_BUILD_TYPE "Release")

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

include(ConfigureArrow)

###################################################################################################
# - find arrow ------------------------------------------------------------------------------------

if (ARROW_FOUND)
    message(STATUS "Apache Arrow found in ${ARROW_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Apache Arrow not found, please check your settings.")
endif()

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories(${GTEST_INCLUDE_DIR})
        add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/moderngpu/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cnmem/include")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}")

###################################################################################################
# - nvstrings -------------------------------------------------------------------------------------

find_file(NVSTRINGS_INCLUDE "NVStrings.h"
          HINTS "$ENV{NVSTRINGS_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include")
find_library(NVSTRINGS_LIBRARY "NVStrings"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

if(NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)
    execute_process(WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
                    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} ln -snf ${NVSTRINGS_INCLUDE} ${CMAKE_BINARY_DIR}/include/NVStrings.h
                    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR} ln -snf ${NVSTRINGS_LIBRARY} ${CMAKE_BINARY_DIR}/lib/libNVStrings.so)
endif(NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf SHARED
            src/comms/ipc/ipc.cu
            src/dataframe/column.cpp
            src/dataframe/context.cpp
            src/filter/filter_ops.cu
            src/join/joining.cu
            src/orderby/orderby.cu
            src/sort/segmented_sorting.cu
            src/sort/sorting.cu
            src/sqls/sqls_ops.cu
            src/groupby/groupby.cu
            src/groupby/new_groupby.cu
            src/binary/binary_ops.cu
            src/bitmask/bitmask_ops.cu
            src/bitmask/valid_ops.cu
            src/compaction/stream_compaction_ops.cu
            src/datetime/datetime_ops.cu
            src/hash/hashing.cu
            src/hash/hash_ops.cu
            src/quantiles/quantiles.cu
            src/reductions/reductions.cu
            src/reductions/scan.cu
            src/unary/unary_ops.cu
            # src/windowed/windowed_ops.cu ... this is broken
            src/io/convert/csr/cudf_to_csr.cu
            src/io/csv/csv_reader.cu
            src/utilities/cuda_utils.cu
            src/utilities/error_utils.cpp
            src/utilities/nvtx/nvtx_utils.cpp)

add_library(rmm SHARED
            src/rmm/memory.cpp
            src/rmm/memory_manager.cpp
            thirdparty/cnmem/src/cnmem.cpp)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
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

target_link_libraries(rmm cudart cuda NVStrings)
target_link_libraries(cudf rmm "${ARROW_LIB}")

###################################################################################################
# - python cffi bindings --------------------------------------------------------------------------

add_custom_command(OUTPUT PYTHON_CFFI
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_BINARY_DIR}/../python ${CMAKE_BINARY_DIR}/python
                   COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/python python setup.py build_ext --inplace
                   VERBATIM)

add_custom_target(python_cffi DEPENDS cudf PYTHON_CFFI)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS cudf rmm
        DESTINATION lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf.h
        DESTINATION include)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf
        DESTINATION include)

add_custom_command(OUTPUT INSTALL_PYTHON_CFFI
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/python python setup.py install --single-version-externally-managed --record=record.txt
                   VERBATIM)

add_custom_target(install_python DEPENDS cudf PYTHON_CFFI INSTALL_PYTHON_CFFI)






