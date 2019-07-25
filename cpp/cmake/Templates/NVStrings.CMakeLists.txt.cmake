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
cmake_minimum_required (VERSION 3.12)

project(NVStrings VERSION 0.9.0 LANGUAGES C CXX CUDA)

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
        message(STATUS "NVSTRINGS: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "NVSTRINGS: Disabling the GLIBCXX11 ABI")
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
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/src"
                    "${RMM_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
                 "${CMAKE_BINARY_DIR}/lib"
                 "${RMM_LIBRARY}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(NVStrings SHARED 
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/NVStrings.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/NVStringsImpl.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/array.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/attrs.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/case.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/combine.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/convert.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/count.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/datetime.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/extract.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/extract_record.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/find.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/findall.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/findall_record.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/modify.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/pad.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/replace.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/replace_backref.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/replace_multi.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/split.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/strip.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/strings/substr.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/util.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/regex/regexec.cpp
            ${CMAKE_SOURCE_DIR}/src/custrings/regex/regcomp.cpp)

add_library(NVCategory SHARED
            ${CMAKE_SOURCE_DIR}/src/custrings/category/NVCategory.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/category/numeric_category.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/category/numeric_category_int.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/category/numeric_category_long.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/category/numeric_category_float.cu
            ${CMAKE_SOURCE_DIR}/src/custrings/category/numeric_category_double.cu)

add_library(NVText SHARED
            ${CMAKE_SOURCE_DIR}/src/custrings/text/NVText.cu)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(NVStrings rmm cudart cuda)
target_link_libraries(NVCategory NVStrings rmm cudart cuda)
target_link_libraries(NVText NVStrings rmm cudart cuda)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS NVStrings
        DESTINATION lib)

install(TARGETS NVCategory
        DESTINATION lib)

install(TARGETS NVText LIBRARY
        DESTINATION lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/nvstrings
        DESTINATION include)

###################################################################################################
# - make documentation ----------------------------------------------------------------------------

add_custom_command(OUTPUT DOCS_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(doc DEPENDS DOCS_DOXYGEN)
