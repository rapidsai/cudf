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

project(pyniNVStrings VERSION 0.10.0 LANGUAGES C CXX CUDA)

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
# - conda environment -----------------------------------------------------------------------------

if ("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    message(STATUS "Conda build detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
elseif (DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_SYSTEM_PREFIX_PATH}")
    message(STATUS "Conda environment detected, CMAKE_SYSTEM_PREFIX_PATH set to: ${CMAKE_SYSTEM_PREFIX_PATH}")
endif ()

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
# - NVStrings -------------------------------------------------------------------------------------

find_path(
    NVSTRINGS_INCLUDE "nvstrings"
    HINTS "$ENV{NVSTRINGS_ROOT}/include"
          "../../../cpp/include"
)

find_library(
    NVSTRINGS_LIBRARY "NVStrings"
    HINTS "$ENV{NVSTRINGS_ROOT}/lib"
          "$ENV{NVSTRINGS_ROOT}"
)

find_library(
    NVCATEGORY_LIBRARY "NVCategory"
    HINTS "$ENV{NVSTRINGS_ROOT}/lib"
          "$ENV{NVSTRINGS_ROOT}"
)

find_library(
    NVTEXT_LIBRARY "NVText"
    HINTS "$ENV{NVSTRINGS_ROOT}/lib"
          "$ENV{NVSTRINGS_ROOT}"
)

message(STATUS "NVSTRINGS: NVSTRINGS_INCLUDE set to ${NVSTRINGS_INCLUDE}")
message(STATUS "NVSTRINGS: NVSTRINGS_LIBRARY set to ${NVSTRINGS_LIBRARY}")
message(STATUS "NVSTRINGS: NVCATEGORY_LIBRARY set to ${NVCATEGORY_LIBRARY}")
message(STATUS "NVSTRINGS: NVTEXT_LIBRARY set to ${NVTEXT_LIBRARY}")

if (NOT (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY AND NVCATEGORY_LIBRARY AND NVTEXT_LIBRARY))
    message(FATAL_ERROR "Unable to find NVStrings libraries")
endif (NOT (NVSTRINGS_INCLUDE AND NVSTRINGS_LIBRARY AND NVCATEGORY_LIBRARY AND NVTEXT_LIBRARY))

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
