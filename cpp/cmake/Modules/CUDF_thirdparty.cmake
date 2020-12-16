#=============================================================================
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

###################################################################################################
# - find zlib -------------------------------------------------------------------------------------

find_package(ZLIB REQUIRED)

message(STATUS "ZLIB_LIBRARIES: ${ZLIB_LIBRARIES}")
message(STATUS "ZLIB_INCLUDE_DIRS: ${ZLIB_INCLUDE_DIRS}")

###################################################################################################
# - find boost ------------------------------------------------------------------------------------

# Don't look for a CMake configuration file
# set(Boost_NO_BOOST_CMAKE ON)

set(CUDF_MIN_VERSION_Boost 1.71.0)

find_package(Boost ${CUDF_MIN_VERSION_Boost} QUIET MODULE COMPONENTS filesystem)

if (NOT Boost_FOUND)
    CPMAddPackage(NAME         Boost
        GIT_REPOSITORY         https://github.com/Orphis/boost-cmake.git
        VERSION                ${CUDF_MIN_VERSION_Boost}
        FIND_PACKAGE_ARGUMENTS "COMPONENTS filesystem"
    )
endif()

message(STATUS "Boost_FOUND: ${Boost_FOUND}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")

###################################################################################################
# - find jitify -----------------------------------------------------------------------------------

CPMAddPackage(NAME  jitify
    VERSION         1.0.0
    GIT_REPOSITORY  https://github.com/rapidsai/jitify.git
    GIT_TAG         cudf_0.16
    GIT_SHALLOW     TRUE
    DONWLOAD_ONLY   TRUE)

set(JITIFY_INCLUDE_DIR "${jitify_SOURCE_DIR}")

message(STATUS "JITIFY_INCLUDE_DIR: ${JITIFY_INCLUDE_DIR}")

###################################################################################################
# - find libcu++ ----------------------------------------------------------------------------------

set(CUDF_MIN_VERSION_libcudacxx 1.4.0)

CPMAddPackage(NAME  libcudacxx
    VERSION         ${CUDF_MIN_VERSION_libcudacxx}
    GIT_REPOSITORY  https://github.com/NVIDIA/libcudacxx.git
    GIT_TAG         ${CUDF_MIN_VERSION_libcudacxx}
    GIT_SHALLOW     TRUE
    DONWLOAD_ONLY   TRUE
    OPTIONS         "LIBCXX_CONFIGURE_IDE OFF"
                    "LIBCXX_ENABLE_STATIC OFF"
                    "LIBCXX_ENABLE_SHARED OFF"
                    "LIBCXX_INCLUDE_TESTS OFF"
                    "LIBCXX_INSTALL_LIBRARY OFF"
                    "LIBCXX_INSTALL_HEADERS OFF"
                    "LIBCXX_STANDALONE_BUILD OFF"
                    "LIBCXX_DISABLE_ARCH_BY_DEFAULT ON"
                    "LIBCXX_INSTALL_SUPPORT_HEADERS OFF"
                    "LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY OFF"
                    # Set this path to a non-existent LLVM to defeat libcu++'s CMakeLists.txt.
                    # Caused by a CPM bug? https://github.com/TheLartians/CPM.cmake/issues/173
                    "LLVM_PATH /tmp"
)

set(LIBCUDACXX_DIR "${libcudacxx_SOURCE_DIR}")
set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include")

message(STATUS "LIBCUDACXX_DIR: ${LIBCUDACXX_DIR}")
message(STATUS "LIBCUDACXX_INCLUDE_DIR: ${LIBCUDACXX_INCLUDE_DIR}")

set(LIBCXX_DIR "${libcudacxx_SOURCE_DIR}/libcxx")
set(LIBCXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/libcxx/include")

message(STATUS "LIBCXX_DIR: ${LIBCXX_DIR}")
message(STATUS "LIBCXX_INCLUDE_DIR: ${LIBCXX_INCLUDE_DIR}")

###################################################################################################
# - find spdlog -----------------------------------------------------------------------------------

set(CUDF_MIN_VERSION_spdlog 1.7.0)

if (NOT SPDLOG_INCLUDE)
    CPMAddPackage(NAME  spdlog
        VERSION         ${CUDF_MIN_VERSION_spdlog}
        GIT_REPOSITORY  https://github.com/gabime/spdlog.git
        GIT_TAG         "v${CUDF_MIN_VERSION_spdlog}"
        GIT_SHALLOW     TRUE
        # If there is no pre-installed spdlog we can use, we'll install our fetched copy together with cuDF
        OPTIONS         "SPDLOG_INSTALL TRUE")
    set(SPDLOG_INCLUDE "${spdlog_SOURCE_DIR}/include")
endif()

set(SPDLOG_INCLUDE "${SPDLOG_INCLUDE}")

message(STATUS "SPDLOG_INCLUDE: ${SPDLOG_INCLUDE}")

###################################################################################################
# - find thrust/cub -------------------------------------------------------------------------------

set(CUDF_MIN_VERSION_Thrust 1.10.0)

CPMAddPackage(NAME  Thrust
  GIT_REPOSITORY    https://github.com/NVIDIA/thrust.git
  GIT_TAG           ${CUDF_MIN_VERSION_Thrust}
  VERSION           ${CUDF_MIN_VERSION_Thrust}
  GIT_SHALLOW       TRUE
  # If there is no pre-installed thrust we can use, we'll install our fetched copy together with cuDF
  OPTIONS           "THRUST_INSTALL TRUE"
  PATCH_COMMAND     patch -p1 -N < ${CUDA_DATAFRAME_SOURCE_DIR}/cmake/thrust.patch || true)

thrust_create_target(cudf::Thrust FROM_OPTIONS)

set(THRUST_INCLUDE_DIR "${Thrust_SOURCE_DIR}")

message(STATUS "THRUST_INCLUDE_DIR: ${THRUST_INCLUDE_DIR}")

###################################################################################################
# - find rmm --------------------------------------------------------------------------------------

set(CUDF_MIN_VERSION_rmm "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}")

if(RMM_INCLUDE)
    add_library(rmm INTERFACE)
    target_include_directories(rmm INTERFACE "$<BUILD_INTERFACE:${RMM_INCLUDE}>"
                                             "$<BUILD_INTERFACE:${SPDLOG_INCLUDE}>"
                                             "$<BUILD_INTERFACE:${THRUST_INCLUDE_DIR}>")
    add_library(rmm::rmm ALIAS rmm)
else()
    find_package(rmm ${CUDF_MIN_VERSION_rmm} QUIET)
    if (NOT rmm_FOUND)
        CPMAddPackage(NAME  rmm
            VERSION         ${CUDF_MIN_VERSION_rmm}
            GIT_REPOSITORY  https://github.com/rapidsai/rmm.git
            GIT_TAG         branch-${CUDF_MIN_VERSION_rmm}
            GIT_SHALLOW     TRUE
            OPTIONS         "BUILD_TESTS OFF"
                            "BUILD_BENCHMARKS OFF"
                            "CUDA_STATIC_RUNTIME ${CUDA_STATIC_RUNTIME}"
                            "DISABLE_DEPRECATION_WARNING ${DISABLE_DEPRECATION_WARNING}"
        )
    endif()
    set(RMM_INCLUDE "${rmm_SOURCE_DIR}/include")
endif()

set(RMM_INCLUDE "${RMM_INCLUDE}")

message(STATUS "RMM_INCLUDE: ${RMM_INCLUDE}")

###################################################################################################
# - find dlpack -----------------------------------------------------------------------------------

set(CUDF_MIN_VERSION_dlpack 0.3)

if (DLPACK_INCLUDE)
    set(dlpack_FOUND TRUE)
else()
    find_package(dlpack ${CUDF_MIN_VERSION_dlpack} QUIET)
endif()

if(NOT dlpack_FOUND)
    CPMAddPackage(NAME  dlpack
        VERSION         ${CUDF_MIN_VERSION_dlpack}
        GIT_REPOSITORY  https://github.com/dmlc/dlpack.git
        GIT_TAG         "v${CUDF_MIN_VERSION_dlpack}"
        GIT_SHALLOW     TRUE
        # If there is no pre-installed dlpack we can use, we'll install our fetched copy together with cuDF
        OPTIONS         "BUILD_MOCK OFF")
    set(DLPACK_INCLUDE "${dlpack_SOURCE_DIR}/include")
endif()

set(DLPACK_INCLUDE "${DLPACK_INCLUDE}")

message(STATUS "DLPACK_INCLUDE: ${DLPACK_INCLUDE}")

###################################################################################################
# - find arrow ------------------------------------------------------------------------------------

set(CUDF_MIN_VERSION_arrow 1.0.1)

if(ARROW_STATIC_LIB)
  message(STATUS "BUILDING ARROW")
  include(ConfigureArrow)

  if(ARROW_FOUND)
    message(STATUS "Apache Arrow found in ${ARROW_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "Apache Arrow not found, please check your settings.")
  endif(ARROW_FOUND)

  add_library(arrow STATIC IMPORTED ${ARROW_LIB})
  add_library(arrow_cuda STATIC IMPORTED ${ARROW_CUDA_LIB})
else()
  find_path(ARROW_INCLUDE_DIR "arrow"
      HINTS "$ENV{ARROW_ROOT}/include")

  find_library(ARROW_LIB "arrow"
      NAMES libarrow
      HINTS "$ENV{ARROW_ROOT}/lib" "$ENV{ARROW_ROOT}/build")

  find_library(ARROW_CUDA_LIB "arrow_cuda"
      NAMES libarrow_cuda
      HINTS "$ENV{ARROW_ROOT}/lib" "$ENV{ARROW_ROOT}/build")

  message(STATUS "ARROW: ARROW_INCLUDE_DIR set to ${ARROW_INCLUDE_DIR}")
  message(STATUS "ARROW: ARROW_LIB set to ${ARROW_LIB}")
  message(STATUS "ARROW: ARROW_CUDA_LIB set to ${ARROW_CUDA_LIB}")

  add_library(arrow SHARED IMPORTED ${ARROW_LIB})
  add_library(arrow_cuda SHARED IMPORTED ${ARROW_CUDA_LIB})
endif(ARROW_STATIC_LIB)

if(ARROW_INCLUDE_DIR AND ARROW_LIB AND ARROW_CUDA_LIB)
  set_target_properties(arrow PROPERTIES IMPORTED_LOCATION ${ARROW_LIB})
  set_target_properties(arrow_cuda PROPERTIES IMPORTED_LOCATION ${ARROW_CUDA_LIB})
endif(ARROW_INCLUDE_DIR AND ARROW_LIB AND ARROW_CUDA_LIB)

set(ARROW_LIB ${ARROW_LIB})
set(ARROW_CUDA_LIB ${ARROW_CUDA_LIB})
set(ARROW_INCLUDE_DIR ${ARROW_INCLUDE_DIR})

message(STATUS "ARROW_LIB: ${ARROW_LIB}")
message(STATUS "ARROW_CUDA_LIB: ${ARROW_CUDA_LIB}")
message(STATUS "ARROW_INCLUDE_DIR: ${ARROW_INCLUDE_DIR}")
