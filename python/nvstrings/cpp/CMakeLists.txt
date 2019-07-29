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

project(pyniNVStrings VERSION 0.9.0 LANGUAGES C CXX CUDA)

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

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "NVSTRINGS PYTHON: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "NVSTRINGS PYTHON: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

# To apply RUNPATH to transitive dependencies (this is a temporary solution)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

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

if(EXISTS "${CMAKE_SOURCE_DIR}/../../../cpp/build/nvstrings/libNVStrings.so" AND
   EXISTS "${CMAKE_SOURCE_DIR}/../../../cpp/build/nvstrings/libNVCategory.so" AND
   EXISTS "${CMAKE_SOURCE_DIR}/../../../cpp/build/nvstrings/libNVText.so")
  set(NVSTRINGS_LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp/build/nvstrings")
  message(STATUS "NVStrings found successfully")
  message(STATUS "NVStrings library path: " ${NVSTRINGS_LIBRARY_DIR})
  
  add_library(NVStrings SHARED IMPORTED)
  set_target_properties(NVStrings PROPERTIES IMPORTED_LOCATION ${NVSTRINGS_LIBRARY_DIR}/libNVStrings.so)

  add_library(NVCategory SHARED IMPORTED)
  set_target_properties(NVCategory PROPERTIES IMPORTED_LOCATION ${NVSTRINGS_LIBRARY_DIR}/libNVCategory.so)

  add_library(NVText SHARED IMPORTED)
  set_target_properties(NVText PROPERTIES IMPORTED_LOCATION ${NVSTRINGS_LIBRARY_DIR}/libNVText.so)
else()
  message(FATAL_ERROR "Could not find NVStrings under `cudf/cpp/build`")
endif(EXISTS "${CMAKE_SOURCE_DIR}/../../../cpp/build/nvstrings/libNVStrings.so" AND
      EXISTS "${CMAKE_SOURCE_DIR}/../../../cpp/build/nvstrings/libNVCategory.so" AND
      EXISTS "${CMAKE_SOURCE_DIR}/../../../cpp/build/nvstrings/libNVText.so")

###################################################################################################
# - python ----------------------------------------------------------------------------------------

find_package(PythonLibs)

message(STATUS "PYTHON: PYTHON_LIBRARIES set to ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON: PYTHON_INCLUDE_DIRS set to ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON: PYTHONLIBS_VERSION_STRING set to ${PYTHONLIBS_VERSION_STRING}")

add_library(python3 SHARED IMPORTED ${PYTHON_LIBRARIES})
if (PYTHON_INCLUDE_DIRS AND PYTHON_LIBRARIES)
    set_target_properties(python3 PROPERTIES IMPORTED_LOCATION ${PYTHON_LIBRARIES})
endif (PYTHON_INCLUDE_DIRS AND PYTHON_LIBRARIES)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${PYTHON_INCLUDE_DIRS}"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                    "${CMAKE_SOURCE_DIR}/../../../cpp/include"
                    "${RMM_INCLUDE}")

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${PYTHON_LIBRARIES}"
                 "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
                 "${NVSTRINGS_LIBRARY_DIR}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(pyniNVStrings SHARED pystrings.cpp)
add_library(pyniNVCategory SHARED pycategory.cpp numeric_category.cpp)
add_library(pyniNVText SHARED pytext.cpp)

set_target_properties(pyniNVStrings PROPERTIES SUFFIX ".so")
set_target_properties(pyniNVStrings PROPERTIES PREFIX "")
set_target_properties(pyniNVCategory PROPERTIES SUFFIX ".so")
set_target_properties(pyniNVCategory PROPERTIES PREFIX "")
set_target_properties(pyniNVText PROPERTIES SUFFIX ".so")
set_target_properties(pyniNVText PROPERTIES PREFIX "")

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(pyniNVStrings python3 NVStrings)
target_link_libraries(pyniNVCategory python3 NVStrings NVCategory)
target_link_libraries(pyniNVText python3 NVStrings NVText)
