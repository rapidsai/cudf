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

project(CUDA_DATAFRAME VERSION 0.9.0 LANGUAGES C CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()

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

# To apply RUNPATH to transitive dependencies (this is a temporary solution)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

option(BUILD_BENCHMARKS "Configure CMake to build (google) benchmarks" OFF)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if ("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif (DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif ()

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

message(STATUS "ZLIB: ZLIB_LIBRARIES set to ${ZLIB_LIBRARIES}")
message(STATUS "ZLIB: ZLIB_INCLUDE_DIRS set to ${ZLIB_INCLUDE_DIRS}")

if(ZLIB_FOUND)
    message(STATUS "ZLib found in ${ZLIB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "ZLib not found, please check your settings.")
endif(ZLIB_FOUND)

###################################################################################################
# - find boost ------------------------------------------------------------------------------------

# Don't look for a CMake configuration file
set(Boost_NO_BOOST_CMAKE ON)

find_package(
    Boost REQUIRED MODULE
    COMPONENTS filesystem
)

message(STATUS "BOOST: Boost_LIBRARIES set to ${Boost_LIBRARIES}")
message(STATUS "BOOST: Boost_INCLUDE_DIRS set to ${Boost_INCLUDE_DIRS}")

if(Boost_FOUND)
    message(STATUS "Boost found in ${Boost_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Boost not found, please check your settings.")
endif(Boost_FOUND)

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(
    RMM_INCLUDE "rmm"
    HINTS "$ENV{RMM_ROOT}/include"
)

find_library(
    RMM_LIBRARY "rmm"
    HINTS "$ENV{RMM_ROOT}/lib"
)

message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if (RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif (RMM_INCLUDE AND RMM_LIBRARY)

###################################################################################################
# - DLPACK -------------------------------------------------------------------------------------------

find_path(DLPACK_INCLUDE "dlpack"
          HINTS "$ENV{DLPACK_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include/dlpack"
                "$ENV{CONDA_PREFIX}/include")

message(STATUS "DLPACK: DLPACK_INCLUDE set to ${DLPACK_INCLUDE}")

###################################################################################################
# - NVStrings -------------------------------------------------------------------------------------

find_path(
    NVSTRINGS_INCLUDE "nvstrings"
    HINTS "$ENV{NVSTRINGS_ROOT}/include"
)

find_library(
    NVSTRINGS_LIBRARY "NVStrings"
    HINTS "$ENV{NVSTRINGS_ROOT}/lib"
)

find_library(
    NVCATEGORY_LIBRARY "NVCategory"
    HINTS "$ENV{NVSTRINGS_ROOT}/lib"
)

find_library(
    NVTEXT_LIBRARY "NVText"
    HINTS "$ENV{NVSTRINGS_ROOT}/lib"
)

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
# - jitify ----------------------------------------------------------------------------------------

option(JITIFY_USE_CACHE "Use a file cache for JIT compiled kernels" ON)
if(JITIFY_USE_CACHE)
    message(STATUS "Using file cache for JIT compiled kernels")
    add_definitions("-DJITIFY_USE_CACHE -DCUDF_VERSION=${CMAKE_PROJECT_VERSION}")
endif(JITIFY_USE_CACHE)

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
# - add google benchmark --------------------------------------------------------------------------

if(BUILD_BENCHMARKS)
  
  include(ConfigureGoogleBenchmark)
  
  if(GBENCH_FOUND)
    message(STATUS "Google C++ Benchmarking Framework (Google Benchmark) found in ${GBENCH_ROOT}")
    include_directories(${GBENCH_INCLUDE_DIR})
    add_subdirectory(${CMAKE_SOURCE_DIR}/benchmarks)
  else()
    message(AUTHOR_WARNING "Google C++ Benchmarking Framework (Google Benchmark) not found: automated tests are disabled.")
  endif(GBENCH_FOUND)

endif(BUILD_BENCHMARKS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
	include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

include_directories("${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/jitify"
                    "${ARROW_INCLUDE_DIR}"
                    "${FLATBUFFERS_INCLUDE_DIR}"
                    "${ZLIB_INCLUDE_DIRS}"
                    "${Boost_INCLUDE_DIRS}"
                    "${RMM_INCLUDE}"
                    "${DLPACK_INCLUDE}"
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
option(BUILD_SHARED_LIBS "Build a shared library" ON)
add_library(cudf 
            src/comms/ipc/ipc.cpp
            src/column/legacy/column.cpp
            src/column/legacy/context.cpp
            src/table/legacy/table.cpp
            src/string/nvcategory_util.cpp
            src/join/joining.cu
            src/orderby/orderby.cu
            src/predicates/is_sorted.cu
            src/sort/digitize.cu
            src/sort/segmented_sorting.cu
            src/groupby/hash/groupby.cu
            src/groupby/groupby_without_aggregation.cu
            src/groupby/hash/aggregation_requests.cpp
            src/rolling/rolling.cu
            src/rolling/jit/code/kernel.cpp
            src/rolling/jit/code/operation.cpp
            src/rolling/jit/util/type.cpp
            src/binaryop/binaryop.cpp
            src/binaryop/compiled/binary_ops.cu
            src/binaryop/jit/code/kernel.cpp
            src/binaryop/jit/code/operation.cpp
            src/binaryop/jit/code/traits.cpp
            src/binaryop/jit/util/operator.cpp
            src/binaryop/jit/util/type.cpp
            src/jit/type.cpp
            src/jit/types_h_jit.cpp
            src/jit/parser.cpp
            src/jit/cache.cpp
            src/jit/launcher.cpp
            src/transform/transform.cpp
            src/transform/jit/code/kernel.cpp
            src/transform/nans_to_nulls.cu
            src/bitmask/legacy/bitmask_ops.cu
            src/stream_compaction/apply_boolean_mask.cu
            src/stream_compaction/drop_nulls.cu
            src/stream_compaction/drop_duplicates.cu
            src/datetime/datetime_ops.cu
            src/datetime/datetime_util.cpp
            src/hash/hashing.cu
            src/quantiles/quantiles.cu
            src/reductions/reductions.cu
            src/reductions/min.cu
            src/reductions/max.cu
            src/reductions/sum.cu
            src/reductions/product.cu
            src/reductions/sum_of_squares.cu
            src/reductions/mean.cu
            src/reductions/var.cu
            src/reductions/std.cu
            src/replace/replace.cu
            src/reductions/scan.cu
            src/transpose/transpose.cu
            src/merge/merge.cu
            src/unary/math_ops.cu
            src/unary/cast_ops.cu
            src/io/cuio_common.cpp
            src/io/io_functions.cpp
            src/io/convert/csr/cudf_to_csr.cu
            src/io/convert/dlpack/cudf_dlpack.cpp
            src/io/avro/avro_reader_impl.cu
            src/io/avro/avro_gpu.cu
            src/io/avro/avro.cpp
            src/io/csv/csv_reader_impl.cu
            src/io/csv/csv_writer.cu
            src/io/json/json_reader_impl.cu
            src/io/orc/orc_reader_impl.cu
            src/io/orc/orc.cpp
            src/io/orc/timezone.cpp
            src/io/orc/stripe_data.cu
            src/io/orc/stripe_init.cu
            src/io/parquet/page_data.cu
            src/io/parquet/page_hdr.cu
            src/io/parquet/parquet_reader_impl.cu
            src/io/parquet/parquet.cpp
            src/io/comp/cpu_unbz2.cpp
            src/io/comp/uncomp.cpp
            src/io/comp/brotli_dict.cpp
            src/io/comp/debrotli.cu
            src/io/comp/unsnap.cu
            src/io/comp/gpuinflate.cu
            src/io/utilities/datasource.cpp
            src/io/utilities/parsing_utils.cu
            src/utilities/cuda_utils.cu
            src/utilities/column_utils.cpp
            src/utilities/error_utils.cpp
            src/utilities/nvtx/nvtx_utils.cpp
            src/copying/copy.cpp
            src/copying/gather.cu
            src/copying/scatter.cu
            src/copying/slice.cu
            src/copying/split.cu
            src/bitmask/legacy/legacy_bitmask.cpp
            src/gis/point_in_polygon.cu
            src/copying/copy_range.cu
            src/filling/fill.cu
            src/search/search.cu)

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

SET(ARROW_CUDA_LIB_LINK -Wl,--whole-archive ${ARROW_CUDA_LIB} -Wl,--no-whole-archive)

target_link_libraries(cudf rmm ${ARROW_CUDA_LIB_LINK} ${ARROW_LIB} cudart cuda ${ZLIB_LIBRARIES} ${Boost_LIBRARIES} NVCategory NVStrings nvrtc)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS cudf
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf
        DESTINATION include)

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

add_custom_command(OUTPUT CUDF_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(doc DEPENDS CUDF_DOXYGEN)
