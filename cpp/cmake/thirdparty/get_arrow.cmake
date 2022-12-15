# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Finding arrow is far more complex than it should be, and as a result we violate multiple linting
# rules aiming to limit complexity. Since all our other CMake scripts conform to expectations
# without undue difficulty, disabling those rules for just this function is our best approach for
# now. The spacing between this comment, the cmake-lint directives, and the function docstring is
# necessary to prevent cmake-format from trying to combine the lines.

# cmake-lint: disable=R0912,R0913,R0915

include_guard(GLOBAL)

# Generate a FindArrow module for the case where we need to search for arrow within a pip install
# pyarrow.
function(find_libarrow_in_python_wheel PYARROW_VERSION)
  string(REPLACE "." "" PYARROW_SO_VER "${PYARROW_VERSION}")
  set(PYARROW_LIB libarrow.so.${PYARROW_SO_VER})

  find_package(Python REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_library_dirs()[0])"
    OUTPUT_VARIABLE CUDF_PYARROW_WHEEL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  list(APPEND CMAKE_PREFIX_PATH "${CUDF_PYARROW_WHEEL_DIR}")
  rapids_find_generate_module(
    Arrow NO_CONFIG
    VERSION "${PYARROW_VERSION}"
    LIBRARY_NAMES "${PYARROW_LIB}"
    BUILD_EXPORT_SET cudf-exports
    INSTALL_EXPORT_SET cudf-exports
    HEADER_NAMES arrow/python/arrow_to_pandas.h
  )

  find_package(Arrow ${PYARROW_VERSION} MODULE REQUIRED GLOBAL)
  add_library(arrow_shared ALIAS Arrow::Arrow)

  # When using the libarrow inside a wheel we must build libcudf with the old ABI because pyarrow's
  # `libarrow.so` is compiled for manylinux2014 (centos7 toolchain) which uses the old ABI. Note
  # that these flags will often be redundant because we build wheels in manylinux containers that
  # actually have the old libc++ anyway, but setting them explicitly ensures correct and consistent
  # behavior in all other cases such as aarch builds on newer manylinux or testing builds in newer
  # containers. Note that tests will not build successfully without also propagating these options
  # to builds of GTest. Similarly, benchmarks will not work without updating GBench (and possibly
  # NVBench) builds. We are currently ignoring these limitations since we don't anticipate using
  # this feature except for building wheels.
  target_compile_options(
    Arrow::Arrow INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-D_GLIBCXX_USE_CXX11_ABI=0>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=0>"
  )

  rapids_export_package(BUILD Arrow cudf-exports)
  rapids_export_package(INSTALL Arrow cudf-exports)

  list(POP_BACK CMAKE_PREFIX_PATH)
endfunction()

# This function finds arrow and sets any additional necessary environment variables.
function(find_and_configure_arrow VERSION BUILD_STATIC ENABLE_S3 ENABLE_ORC ENABLE_PYTHON
         ENABLE_PARQUET
)

  if(USE_LIBARROW_FROM_PYARROW)
    # Generate a FindArrow.cmake to find pyarrow's libarrow.so
    find_libarrow_in_python_wheel(${VERSION})
    set(ARROW_FOUND
        TRUE
        PARENT_SCOPE
    )
    set(ARROW_LIBRARIES
        arrow_shared
        PARENT_SCOPE
    )
    return()
  endif()

  if(BUILD_STATIC)
    if(TARGET arrow_static)
      set(ARROW_FOUND
          TRUE
          PARENT_SCOPE
      )
      set(ARROW_LIBRARIES
          arrow_static
          PARENT_SCOPE
      )
      return()
    endif()
  else()
    if(TARGET arrow_shared)
      set(ARROW_FOUND
          TRUE
          PARENT_SCOPE
      )
      set(ARROW_LIBRARIES
          arrow_shared
          PARENT_SCOPE
      )
      return()
    endif()
  endif()

  if(NOT ARROW_ARMV8_ARCH)
    set(ARROW_ARMV8_ARCH "armv8-a")
  endif()

  if(NOT ARROW_SIMD_LEVEL)
    set(ARROW_SIMD_LEVEL "NONE")
  endif()

  if(BUILD_STATIC)
    set(ARROW_BUILD_STATIC ON)
    set(ARROW_BUILD_SHARED OFF)
    # Turn off CPM using `find_package` so we always download and make sure we get proper static
    # library.
    set(CPM_DOWNLOAD_Arrow TRUE)
    # By default ARROW will try to search for a static version of OpenSSL which is a bad idea given
    # that shared linking is advised for critical components like SSL. If a static build is
    # requested, we honor ARROW's default of static linking, but users may consider setting
    # ARROW_OPENSSL_USE_SHARED even in static builds.
  else()
    set(ARROW_BUILD_SHARED ON)
    set(ARROW_BUILD_STATIC OFF)
    # By default ARROW will try to search for a static version of OpenSSL which is a bad idea given
    # that shared linking is advised for critical components like SSL
    set(ARROW_OPENSSL_USE_SHARED ON)
  endif()

  set(ARROW_PYTHON_OPTIONS "")
  if(ENABLE_PYTHON)
    list(APPEND ARROW_PYTHON_OPTIONS "ARROW_PYTHON ON")
    # Arrow's logic to build Boost from source is busted, so we have to get it from the system.
    list(APPEND ARROW_PYTHON_OPTIONS "BOOST_SOURCE SYSTEM")
    list(APPEND ARROW_PYTHON_OPTIONS "ARROW_DEPENDENCY_SOURCE AUTO")
  endif()

  set(ARROW_PARQUET_OPTIONS "")
  if(ENABLE_PARQUET)
    # Arrow's logic to build Boost from source is busted, so we have to get it from the system.
    list(APPEND ARROW_PARQUET_OPTIONS "BOOST_SOURCE SYSTEM")
    list(APPEND ARROW_PARQUET_OPTIONS "Thrift_SOURCE BUNDLED")
    list(APPEND ARROW_PARQUET_OPTIONS "ARROW_DEPENDENCY_SOURCE AUTO")
  endif()

  rapids_cpm_find(
    Arrow ${VERSION}
    GLOBAL_TARGETS arrow_shared parquet_shared arrow_dataset_shared arrow_static parquet_static
                   arrow_dataset_static
    CPM_ARGS
    GIT_REPOSITORY https://github.com/apache/arrow.git
    GIT_TAG apache-arrow-${VERSION}
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "CMAKE_VERBOSE_MAKEFILE ON"
            "ARROW_IPC ON"
            "ARROW_DATASET ON"
            "ARROW_WITH_BACKTRACE ON"
            "ARROW_CXXFLAGS -w"
            "ARROW_JEMALLOC OFF"
            "ARROW_S3 ${ENABLE_S3}"
            "ARROW_ORC ${ENABLE_ORC}"
            # e.g. needed by blazingsql-io
            ${ARROW_PARQUET_OPTIONS}
            "ARROW_PARQUET ${ENABLE_PARQUET}"
            ${ARROW_PYTHON_OPTIONS}
            # Arrow modifies CMake's GLOBAL RULE_LAUNCH_COMPILE unless this is off
            "ARROW_USE_CCACHE OFF"
            "ARROW_ARMV8_ARCH ${ARROW_ARMV8_ARCH}"
            "ARROW_SIMD_LEVEL ${ARROW_SIMD_LEVEL}"
            "ARROW_BUILD_STATIC ${ARROW_BUILD_STATIC}"
            "ARROW_BUILD_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_POSITION_INDEPENDENT_CODE ON"
            "ARROW_DEPENDENCY_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_BOOST_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_BROTLI_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_GFLAGS_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_GRPC_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_PROTOBUF_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_ZSTD_USE_SHARED ${ARROW_BUILD_SHARED}"
            "xsimd_SOURCE AUTO"
  )

  set(ARROW_FOUND
      TRUE
      PARENT_SCOPE
  )

  if(BUILD_STATIC)
    set(ARROW_LIBRARIES arrow_static)
  else()
    set(ARROW_LIBRARIES arrow_shared)
  endif()

  # Arrow_DIR:   set if CPM found Arrow on the system/conda/etc.
  if(Arrow_DIR)
    # This extra find_package is necessary because rapids_cpm_find does not propagate all the
    # variables from find_package that we might need. This is especially problematic when
    # rapids_cpm_find builds from source.
    find_package(Arrow REQUIRED QUIET)
    if(ENABLE_PARQUET)
      # Setting Parquet_DIR is conditional because parquet may be installed independently of arrow.
      if(NOT Parquet_DIR)
        # Set this to enable `find_package(Parquet)`
        set(Parquet_DIR "${Arrow_DIR}")
      endif()
      # Set this to enable `find_package(ArrowDataset)`
      set(ArrowDataset_DIR "${Arrow_DIR}")
      find_package(ArrowDataset REQUIRED QUIET)
    endif()
    # Arrow_ADDED: set if CPM downloaded Arrow from Github
  elseif(Arrow_ADDED)
    # Copy these files so we can avoid adding paths in Arrow_BINARY_DIR to
    # target_include_directories. That defeats ccache.
    file(INSTALL "${Arrow_BINARY_DIR}/src/arrow/util/config.h"
         DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/arrow/util"
    )
    if(ENABLE_PARQUET)
      file(INSTALL "${Arrow_BINARY_DIR}/src/parquet/parquet_version.h"
           DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/parquet"
      )
    endif()
    # Arrow populates INTERFACE_INCLUDE_DIRECTORIES for the `arrow_static` and `arrow_shared`
    # targets in FindArrow, so for static source-builds, we have to do it after-the-fact.
    #
    # This only works because we know exactly which components we're using. Don't forget to update
    # this list if we add more!
    #
    foreach(ARROW_LIBRARY ${ARROW_LIBRARIES})
      target_include_directories(
        ${ARROW_LIBRARY}
        INTERFACE "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src/generated>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/hadoop/include>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/flatbuffers/include>"
      )
    endforeach()
  else()
    set(ARROW_FOUND
        FALSE
        PARENT_SCOPE
    )
    message(FATAL_ERROR "CUDF: Arrow library not found or downloaded.")
  endif()

  if(Arrow_ADDED)

    set(arrow_code_string
        [=[
          if (TARGET cudf::arrow_shared AND (NOT TARGET arrow_shared))
              add_library(arrow_shared ALIAS cudf::arrow_shared)
          endif()
          if (TARGET cudf::arrow_static AND (NOT TARGET arrow_static))
              add_library(arrow_static ALIAS cudf::arrow_static)
          endif()
          if (NOT TARGET arrow::flatbuffers)
            add_library(arrow::flatbuffers INTERFACE IMPORTED)
          endif()
          if (NOT TARGET arrow::hadoop)
            add_library(arrow::hadoop INTERFACE IMPORTED)
          endif()
        ]=]
    )
    if(ENABLE_PARQUET)
      string(
        APPEND
        arrow_code_string
        "
          find_package(Boost)
          if (NOT TARGET Boost::headers)
            add_library(Boost::headers INTERFACE IMPORTED)
          endif()
        "
      )
    endif()
    if(NOT TARGET xsimd)
      string(
        APPEND
        arrow_code_string
        "
          if(NOT TARGET xsimd)
            add_library(xsimd INTERFACE IMPORTED)
            target_include_directories(xsimd INTERFACE \"${Arrow_BINARY_DIR}/xsimd_ep/src/xsimd_ep-install/include\")
          endif()
        "
      )
    endif()

    rapids_export(
      BUILD Arrow
      VERSION ${VERSION}
      EXPORT_SET arrow_targets
      GLOBAL_TARGETS arrow_shared arrow_static
      NAMESPACE cudf::
      FINAL_CODE_BLOCK arrow_code_string
    )

    if(ENABLE_PARQUET)

      set(arrow_dataset_code_string
          [=[
              if (TARGET cudf::arrow_dataset_shared AND (NOT TARGET arrow_dataset_shared))
                  add_library(arrow_dataset_shared ALIAS cudf::arrow_dataset_shared)
              endif()
              if (TARGET cudf::arrow_dataset_static AND (NOT TARGET arrow_dataset_static))
                  add_library(arrow_dataset_static ALIAS cudf::arrow_dataset_static)
              endif()
            ]=]
      )

      rapids_export(
        BUILD ArrowDataset
        VERSION ${VERSION}
        EXPORT_SET arrow_dataset_targets
        GLOBAL_TARGETS arrow_dataset_shared arrow_dataset_static
        NAMESPACE cudf::
        FINAL_CODE_BLOCK arrow_dataset_code_string
      )

      set(parquet_code_string
          [=[
              if (TARGET cudf::parquet_shared AND (NOT TARGET parquet_shared))
                  add_library(parquet_shared ALIAS cudf::parquet_shared)
              endif()
              if (TARGET cudf::parquet_static AND (NOT TARGET parquet_static))
                  add_library(parquet_static ALIAS cudf::parquet_static)
              endif()
            ]=]
      )

      rapids_export(
        BUILD Parquet
        VERSION ${VERSION}
        EXPORT_SET parquet_targets
        GLOBAL_TARGETS parquet_shared parquet_static
        NAMESPACE cudf::
        FINAL_CODE_BLOCK parquet_code_string
      )
    endif()
  endif()
  # We generate the arrow-configfiles when we built arrow locally, so always do `find_dependency`
  rapids_export_package(BUILD Arrow cudf-exports)
  rapids_export_package(INSTALL Arrow cudf-exports)

  if(ENABLE_PARQUET)
    rapids_export_package(BUILD Parquet cudf-exports)
    rapids_export_package(BUILD ArrowDataset cudf-exports)
  endif()

  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(BUILD Arrow [=[${CMAKE_CURRENT_LIST_DIR}]=] cudf-exports)
  if(ENABLE_PARQUET)
    rapids_export_find_package_root(BUILD Parquet [=[${CMAKE_CURRENT_LIST_DIR}]=] cudf-exports)
    rapids_export_find_package_root(BUILD ArrowDataset [=[${CMAKE_CURRENT_LIST_DIR}]=] cudf-exports)
  endif()

  set(ARROW_LIBRARIES
      "${ARROW_LIBRARIES}"
      PARENT_SCOPE
  )
endfunction()

if(NOT DEFINED CUDF_VERSION_Arrow)
  set(CUDF_VERSION_Arrow
      9.0.0
      CACHE STRING "The version of Arrow to find (or build)"
  )
endif()

find_and_configure_arrow(
  ${CUDF_VERSION_Arrow} ${CUDF_USE_ARROW_STATIC} ${CUDF_ENABLE_ARROW_S3} ${CUDF_ENABLE_ARROW_ORC}
  ${CUDF_ENABLE_ARROW_PYTHON} ${CUDF_ENABLE_ARROW_PARQUET}
)
