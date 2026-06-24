# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# ##################################################################################################
# enable testing -----------------------------------------------------------------------------------
# ##################################################################################################
enable_testing()

file(WRITE "${CUDF_STREAMING_BINARY_DIR}/CTestTestfile.cmake" "subdirs(\"tests\")\n")

# Create a separate test case for each of these n_ranks.
set(nranks_to_run 1 2 5)

# ##################################################################################################
# Test registration model --------------------------------------------------------------------------
# All tests are registered with plain add_test() using the absolute build-tree path of each test
# executable (${CUDF_STREAMING_BINARY_DIR}/gtests/<target>). The generated build-tree
# CTestTestfile.cmake is re-rooted for the install tree by the install section at the bottom: the
# executables are installed next to the CTestTestfile.cmake, so those absolute build paths are
# rewritten to "./<target>". Matching the full absolute build path keeps that rewrite unambiguous.
# ##################################################################################################

# Register a single-process test for TEST_TARGET.
function(libcudf_streaming_test_add)
  set(options) # no options
  set(one_value TEST_TARGET)
  set(multi_value) # no multi_value args
  cmake_parse_arguments(_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _TEST_TEST_TARGET)
    message(FATAL_ERROR "libcudf_streaming_test_add called without a test target")
  endif()

  add_test(NAME "${_TEST_TEST_TARGET}"
           COMMAND "${CUDF_STREAMING_BINARY_DIR}/gtests/${_TEST_TEST_TARGET}"
  )
endfunction()

# Register an mpirun-launched test for TEST_TARGET, creating one test case per parallelism in
# nranks_to_run identified by <TEST_TARGET>_<n_ranks>.
function(libcudf_streaming_mpirun_test_add)
  set(options) # no options
  set(one_value TEST_TARGET)
  set(multi_value) # no multi_value args
  cmake_parse_arguments(_MPIRUN_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _MPIRUN_TEST_TEST_TARGET)
    message(FATAL_ERROR "libcudf_streaming_mpirun_test_add called without a test prefix")
  endif()

  message(STATUS "Adding mpirun test: ${_MPIRUN_TEST_TEST_TARGET} nranks: ${nranks_to_run}")
  foreach(np IN ITEMS ${nranks_to_run})
    add_test(
      NAME "${_MPIRUN_TEST_TEST_TARGET}_${np}"
      COMMAND mpirun --map-by node --bind-to none -np ${np}
              "${CUDF_STREAMING_BINARY_DIR}/gtests/${_MPIRUN_TEST_TEST_TARGET}"
    )
  endforeach(np)
endfunction(libcudf_streaming_mpirun_test_add)

# Collect targets in this directory carrying the property KEY:VALUE into OUTPUT.
function(libcudf_streaming_targets_with_property)
  set(options)
  set(one_value KEY VALUE OUTPUT)
  set(multi_value)
  cmake_parse_arguments(_PROP "${options}" "${one_value}" "${multi_value}" ${ARGN})

  get_property(
    all_targets
    DIRECTORY
    PROPERTY BUILDSYSTEM_TARGETS
  )

  set(matched_targets "")
  foreach(target ${all_targets})
    get_target_property(prop_value ${target} ${_PROP_KEY})
    if(prop_value STREQUAL ${_PROP_VALUE})
      list(APPEND matched_targets ${target})
    endif()
  endforeach()

  set(${_PROP_OUTPUT}
      ${matched_targets}
      PARENT_SCOPE
  )
endfunction()

add_library(libcudf_streaming_test_sources OBJECT)
set_target_properties(
  libcudf_streaming_test_sources
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_include_directories(
  libcudf_streaming_test_sources PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                         "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_options(
  libcudf_streaming_test_sources
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS};--expt-extended-lambda>
)
target_link_libraries(
  libcudf_streaming_test_sources
  PRIVATE cudf_streaming rapidsmpf::rapidsmpf cudf::cudftestutil cudf::cudftestutil_impl
  PUBLIC GTest::gmock GTest::gtest
)
# cudf::cudftestutil_impl injects cudf test-utility .cu sources (via INTERFACE_SOURCES) that are
# compiled into this object library. Those sources use device lambdas, so they require the same CUDA
# flags cudf builds them with (notably --expt-extended-lambda from CUDF_CUDA_FLAGS).
target_compile_options(
  libcudf_streaming_test_sources PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)
target_sources(
  libcudf_streaming_test_sources
  PRIVATE streaming/test_table_chunk.cpp
          streaming/test_read_parquet.cpp
          streaming/test_channel_metadata.cpp
          streaming/test_partition.cpp
          streaming/test_cudf_utils.cpp
          streaming/test_shuffler.cpp
          streaming/test_leaf_actor.cpp
          test_partition.cpp
          test_shuffler.cpp
)

add_executable(libcudf_streaming_single_tests main/single.cpp)
set_target_properties(
  libcudf_streaming_single_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/gtests"
             INSTALL_RPATH "\$ORIGIN/../../../lib"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             # Custom marker property so this target can be discovered and installed below.
             COMPONENT testing
)
target_include_directories(
  libcudf_streaming_single_tests PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                         "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
  libcudf_streaming_single_tests
  PRIVATE cudf_streaming rapidsmpf::rapidsmpf GTest::gmock GTest::gtest
          $<TARGET_NAME_IF_EXISTS:conda_env> libcudf_streaming_test_sources
)
libcudf_streaming_test_add(TEST_TARGET libcudf_streaming_single_tests)

# ##################################################################################################
# MPI and UCXX based tests -------------------------------------------------------------------------
# These executables reuse libcudf_streaming_test_sources but provide a different test environment
# main (main/mpi.cpp / main/ucxx.cpp), and are launched via mpirun for multiple ranks.
if(CUDF_STREAMING_HAVE_MPI)
  add_executable(libcudf_streaming_mpi_tests main/mpi.cpp)
  set_target_properties(
    libcudf_streaming_mpi_tests
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/gtests"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               # Custom marker property so this target can be discovered and installed below.
               COMPONENT testing
  )
  target_include_directories(
    libcudf_streaming_mpi_tests PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                        "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(
    libcudf_streaming_mpi_tests
    PRIVATE cudf_streaming rapidsmpf::rapidsmpf cudf_streaming::optional_communication GTest::gmock
            GTest::gtest $<TARGET_NAME_IF_EXISTS:conda_env> libcudf_streaming_test_sources
  )
  libcudf_streaming_mpirun_test_add(TEST_TARGET libcudf_streaming_mpi_tests)

  if(CUDF_STREAMING_HAVE_UCXX)
    add_executable(libcudf_streaming_ucxx_tests main/ucxx.cpp)
    set_target_properties(
      libcudf_streaming_ucxx_tests
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/gtests"
                 INSTALL_RPATH "\$ORIGIN/../../../lib"
                 CXX_STANDARD 20
                 CXX_STANDARD_REQUIRED ON
                 CXX_EXTENSIONS ON
                 CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 # Custom marker property so this target can be discovered and installed below.
                 COMPONENT testing
    )
    target_include_directories(
      libcudf_streaming_ucxx_tests PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                           "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(
      libcudf_streaming_ucxx_tests
      PRIVATE cudf_streaming
              rapidsmpf::rapidsmpf
              cudf_streaming::optional_communication
              ucxx::ucxx
              GTest::gmock
              GTest::gtest
              $<TARGET_NAME_IF_EXISTS:conda_env>
              libcudf_streaming_test_sources
    )
    libcudf_streaming_mpirun_test_add(TEST_TARGET libcudf_streaming_ucxx_tests)
  else()
    cudf_streaming_skip_optional_target(libcudf_streaming_ucxx_tests "UCXX not found")
  endif()
else()
  cudf_streaming_skip_optional_target(libcudf_streaming_mpi_tests "MPI not found")
endif()

# ##################################################################################################
# Install the tests --------------------------------------------------------------------------------
# Discover every test executable via the custom COMPONENT marker property, install them, and write
# the installed CTestTestfile.cmake from the entries accumulated by the test-add helpers. The
# testing component is EXCLUDE_FROM_ALL so it is only installed on demand (e.g. `cmake --install .
# --component testing`).
# ##################################################################################################
libcudf_streaming_targets_with_property(
  KEY COMPONENT VALUE testing OUTPUT _cudf_streaming_test_targets
)

if(_cudf_streaming_test_targets)
  install(
    TARGETS ${_cudf_streaming_test_targets}
    DESTINATION bin/gtests/libcudf_streaming
    COMPONENT testing
    EXCLUDE_FROM_ALL
  )

  # Re-root the generated build-tree CTestTestfile.cmake for the install tree. Tests reference the
  # absolute build path of each executable (${CUDF_STREAMING_BINARY_DIR}/gtests/<target>); in the
  # install tree the executables sit next to the CTestTestfile.cmake, so that absolute prefix is
  # rewritten to "./. Matching the full absolute build path keeps the rewrite unambiguous. This runs
  # at install time because CTestTestfile.cmake is produced during CMake's generate step (it does
  # not exist yet while this directory is being configured).
  set(_cudf_streaming_relocate_script
      "${CMAKE_CURRENT_BINARY_DIR}/relocate_installed_ctestfile.cmake"
  )
  file(
    WRITE "${_cudf_streaming_relocate_script}"
    "set(_src_ctestfile \"${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake\")\n"
    "set(_build_gtests_prefix \"${CUDF_STREAMING_BINARY_DIR}/gtests/\")\n"
  )
  file(
    APPEND "${_cudf_streaming_relocate_script}"
    [=[
file(READ "${_src_ctestfile}" _ctest_contents)
string(REPLACE "\"${_build_gtests_prefix}" "\"./" _ctest_contents "${_ctest_contents}")
file(WRITE "${CMAKE_INSTALL_PREFIX}/bin/gtests/libcudf_streaming/CTestTestfile.cmake"
     "${_ctest_contents}"
)
]=]
  )
  install(
    SCRIPT "${_cudf_streaming_relocate_script}"
    COMPONENT testing
    EXCLUDE_FROM_ALL
  )
endif()
