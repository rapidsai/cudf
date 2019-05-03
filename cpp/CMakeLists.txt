#=============================================================================
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

project(CUDA_DATAFRAME VERSION 0.7.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

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

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "CUDF: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "CUDF: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# set warnings as errors
# TODO: remove `no-maybe-unitialized` used to suppress warnings in rmm::exec_policy
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror")

# Option to enable line info in CUDA device compilation to allow introspection when profiling / memchecking
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
if (CMAKE_CUDA_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler -rdynamic")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - find arrow ------------------------------------------------------------------------------------

include(ConfigureArrow)

if (ARROW_FOUND)
    message(STATUS "Apache Arrow found in ${ARROW_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Apache Arrow not found, please check your settings.")
endif(ARROW_FOUND)

###################################################################################################
# - find zlib -------------------------------------------------------------------------------------

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "ZLib found in ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLib not found, please check your settings.")
endif(ZLIB_FOUND)

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
          HINTS "$ENV{RMM_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/rmm"
                "$ENV{CONDA_PREFIX}/include")

find_library(RMM_LIBRARY "rmm"
             HINTS "$ENV{RMM_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if (RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif (RMM_INCLUDE AND RMM_LIBRARY)

###################################################################################################
# - NVStrings -------------------------------------------------------------------------------------

find_path(NVSTRINGS_INCLUDE "nvstrings"
          HINTS "$ENV{NVSTRINGS_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/nvstrings"
                "$ENV{CONDA_PREFIX}/include")

find_library(NVSTRINGS_LIBRARY "NVStrings"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

find_library(NVCATEGORY_LIBRARY "NVCategory"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

find_library(NVTEXT_LIBRARY "NVText"
             HINTS "$ENV{NVSTRINGS_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

message(STATUS "NVSTRINGS: NVSTRINGS_INCLUDE set to ${NVSTRINGS_INCLUDE}")
message(STATUS "NVSTRINGS: NVSTRINGS_LIBRARY set to ${NVSTRINGS_LIBRARY}")
message(STATUS "NVSTRINGS: NVCATEGORY_LIBRARY set to ${NVCATEGORY_LIBRARY}")
message(STATUS "NVSTRINGS: NVTEXT_LIBRARY set to ${NVTEXT_LIBRARY}")

add_library(NVStrings SHARED IMPORTED ${NVSTRINGS_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)
    set_target_properties(NVStrings PROPERTIES IMPORTED_LOCATION ${NVSTRINGS_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)

add_library(NVCategory SHARED IMPORTED ${NVCATEGORY_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVCATEGORY_LIBRARY)
    set_target_properties(NVCategory PROPERTIES IMPORTED_LOCATION ${NVCATEGORY_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVCATEGORY_LIBRARY)

add_library(NVText SHARED IMPORTED ${NVTEXT_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVTEXT_LIBRARY)
    set_target_properties(NVText PROPERTIES IMPORTED_LOCATION ${NVTEXT_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVTEXT_LIBRARY)

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

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/jitify"
                    "${CMAKE_SOURCE_DIR}/thirdparty/dlpack/include"
                    "${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${ZLIB_INCLUDE_DIRS}"
                    "${RMM_INCLUDE}"
                    "${NVSTRINGS_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${FLATBUFFERS_LIBRARY_DIR}"
                 "${GTEST_LIBRARY_DIR}"
                 "${RMM_LIBRARY}"
                 "${NVSTRINGS_LIBRARY}"
                 "${NVCATEGORY_LIBRARY}"
                 "${NVTEXT_LIBRARY}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cudf SHARED
            src/comms/ipc/ipc.cu
            src/column/column.cpp
            src/column/context.cpp
            src/string/nvcategory_util.cpp
            src/join/joining.cu
            src/orderby/orderby.cu
            src/sort/digitize.cu
            src/sort/segmented_sorting.cu
            src/sort/sorting.cu
            src/sqls/sqls_ops.cu
            src/groupby/groupby.cu
            src/groupby/new_groupby.cu
            src/binary/binary_ops.cu
            src/binary/jit/code/kernel.cpp
            src/binary/jit/code/operation.cpp
            src/binary/jit/code/traits.cpp
            src/binary/jit/core/binop.cpp
            src/binary/jit/core/launcher.cpp
            src/binary/jit/util/operator.cpp
            src/binary/jit/util/type.cpp
            src/bitmask/bitmask_ops.cu
            src/compaction/stream_compaction_ops.cu
            src/datetime/datetime_ops.cu
            src/hash/hashing.cu
            src/hash/hash_ops.cu
            src/quantiles/quantiles.cu
            src/reductions/reductions.cu
            src/replace/replace.cu
            src/reductions/scan.cu
            src/transpose/transpose.cu
            src/unary/math_ops.cu
            src/unary/cast_ops.cu
            # src/windowed/windowed_ops.cu ... this is broken
            src/io/convert/csr/cudf_to_csr.cu
            src/io/utilities/parsing_utils.cu
            src/io/convert/dlpack/cudf_dlpack.cpp
            src/io/csv/csv_reader.cu
            src/io/csv/csv_writer.cu
            src/io/orc/orc_reader.cu
            src/io/orc/orc.cpp
            src/io/orc/timezone.cpp
            src/io/orc/stripe_data.cu
            src/io/orc/stripe_init.cu
            src/io/parquet/page_data.cu
            src/io/parquet/page_hdr.cu
            src/io/parquet/parquet_reader.cu
            src/io/parquet/parquet.cpp
            src/io/comp/cpu_unbz2.cpp
            src/io/comp/uncomp.cpp
            src/io/comp/brotli_dict.cpp
            src/io/comp/debrotli.cu
            src/io/comp/unsnap.cu
            src/io/comp/gpuinflate.cu
            src/utilities/cuda_utils.cu
            src/utilities/error_utils.cpp
            src/utilities/nvtx/nvtx_utils.cpp
            src/copying/gather.cu
            src/copying/scatter.cu
            src/bitmask/legacy_bitmask.cpp)

#Override RPATH for cudf
SET_TARGET_PROPERTIES(cudf PROPERTIES BUILD_RPATH "\$ORIGIN")

###################################################################################################
# - jitify ----------------------------------------------------------------------------------------

# Creates executable stringify and uses it to convert types.h to c-str for use in JIT code
add_executable(stringify "${CMAKE_SOURCE_DIR}/thirdparty/jitify/stringify.cpp")
execute_process(WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/include/types.h.jit
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
                   COMMAND ${CMAKE_BINARY_DIR}/stringify cudf/types.h > ${CMAKE_BINARY_DIR}/include/types.h.jit
                   COMMENT "Run stringify on header types.h to convert it to c-str for use in JIT compiled code"
                   DEPENDS stringify
                   MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf/types.h)

add_custom_target(stringify_run DEPENDS ${CMAKE_BINARY_DIR}/include/types.h.jit)

add_dependencies(cudf stringify_run)

option(JITIFY_PROCESS_CACHE "Use a process level (instead of thread level) cache for JIT compiled kernels" ON)
if(JITIFY_PROCESS_CACHE)
    message(STATUS "Using process level cache for JIT compiled kernels")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --define-macro JITIFY_THREAD_SAFE")
endif(JITIFY_PROCESS_CACHE)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(cudf ${NVTX_LIBRARY})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX")
endif(USE_NVTX)

option(HT_LEGACY_ALLOCATOR "Use the legacy allocator for hash tables" ON)
if(HT_LEGACY_ALLOCATOR)
    message(STATUS "Using legacy allocator for hash tables")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --define-macro HT_LEGACY_ALLOCATOR")
endif(HT_LEGACY_ALLOCATOR)


###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(cudf rmm cudart cuda "${ARROW_LIB}" ${ZLIB_LIBRARIES} NVCategory NVStrings nvrtc)

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

install(TARGETS cudf
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

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

add_custom_command(OUTPUT CUDF_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(doc DEPENDS CUDF_DOXYGEN)
