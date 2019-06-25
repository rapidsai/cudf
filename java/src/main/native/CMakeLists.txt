#=============================================================================
# Copyright (c) 2019, NVIDIA CORPORATION.
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

project(CUDF_JNI VERSION 0.7.0 LANGUAGES C CXX CUDA)

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
# - CUDF ------------------------------------------------------------------------------------------

set(CUDF_INCLUDE "${PROJECT_SOURCE_DIR}/../../../../cpp/include"
                 "${PROJECT_SOURCE_DIR}/../../../../cpp/src/"
                 "${PROJECT_SOURCE_DIR}/../../../../cpp/thirdparty/cub")

find_library(CUDF_LIBRARY "cudf"
    HINTS "$ENV{CUDF_ROOT}/lib"
          "$ENV{CONDA_PREFIX}/lib"
          "../../../../cpp/build")

message(STATUS "CUDF: CUDF_LIBRARY set to ${CUDF_LIBRARY}")
message(STATUS "CUDF: CUDF_INCLUDE set to ${CUDF_INCLUDE}")

add_library(cudf SHARED IMPORTED ${CUDF_LIBRARY})
if (CUDF_INCLUDE AND CUDF_LIBRARY)
    set_target_properties(cudf PROPERTIES IMPORTED_LOCATION ${CUDF_LIBRARY})
endif (CUDF_INCLUDE AND CUDF_LIBRARY)


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

message(STATUS "NVSTRINGS: NVSTRINGS_INCLUDE set to ${NVSTRINGS_INCLUDE}")
message(STATUS "NVSTRINGS: NVSTRINGS_LIBRARY set to ${NVSTRINGS_LIBRARY}")
message(STATUS "NVSTRINGS: NVCATEGORY_LIBRARY set to ${NVCATEGORY_LIBRARY}")

add_library(NVStrings SHARED IMPORTED ${NVSTRINGS_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)
    set_target_properties(NVStrings PROPERTIES IMPORTED_LOCATION ${NVSTRINGS_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY)

add_library(NVCategory SHARED IMPORTED ${NVCATEGORY_LIBRARY})
if (NVSTRINGS_INCLUDE AND NVCATEGORY_LIBRARY)
    set_target_properties(NVCategory PROPERTIES IMPORTED_LOCATION ${NVCATEGORY_LIBRARY})
endif (NVSTRINGS_INCLUDE AND NVCATEGORY_LIBRARY)


###################################################################################################
# - find JNI -------------------------------------------------------------------------------------
find_package(JNI REQUIRED)
if(JNI_FOUND)
    message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif(JNI_FOUND)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${JNI_INCLUDE_DIRS}"
                    "${CUDF_INCLUDE}"
                    "${RMM_INCLUDE}"
                    "${NVSTRINGS_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${RMM_LIBRARY}"
                 "${CUDF_LIBRARY}"
                 "${NVSTRINGS_LIBRARY}"
                 "${NVCATEGORY_LIBRARY}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

set(SOURCE_FILES
    "src/CudfJni.cpp"
    "src/CudaJni.cpp"
    "src/ColumnVectorJni.cpp"
    "src/RmmJni.cpp"
    "src/TableJni.cpp")
add_library(cudfjni SHARED ${SOURCE_FILES})

#Override RPATH for cudfjni
SET_TARGET_PROPERTIES(cudfjni PROPERTIES BUILD_RPATH "\$ORIGIN")

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(cudfjni ${NVTX_LIBRARY})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NVTX")
endif(USE_NVTX)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(cudfjni cudf rmm cudart cuda NVCategory NVStrings nvrtc)



