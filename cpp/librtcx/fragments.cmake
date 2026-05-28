# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# cmake-lint: disable=C0112,C0113,E1120

#[=======================================================================[.rst:
get_jit_fragment_architectures
------------------------------

Returns the CUDA architectures used for JIT fragment compilation.

.. code-block:: cmake

  get_jit_fragment_architectures(<out_var>)

This function reads the ``rapids_cuda_architectures`` global property when it is available and falls
back to :cmake:variable:`CMAKE_CUDA_ARCHITECTURES` otherwise. Only real architectures and the
virtual architectures required for fragment compatibility are returned.

Result Variables
^^^^^^^^^^^^^^^^

``<out_var>``
  List of CUDA architectures for JIT fragment targets.

#]=======================================================================]
function(get_jit_fragment_architectures OUT_VAR)
  get_property(_RAPIDS_CUDA_ARCHITECTURES GLOBAL PROPERTY rapids_cuda_architectures)
  if(_RAPIDS_CUDA_ARCHITECTURES)
    set(_ARCHITECTURES ${_RAPIDS_CUDA_ARCHITECTURES})
  else()
    set(_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  endif()

  set(_FRAGMENT_ARCHITECTURES)
  foreach(_ARCH IN LISTS _ARCHITECTURES)
    if(_ARCH STREQUAL "RAPIDS" OR _ARCH STREQUAL "ALL")
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0)
        list(APPEND _FRAGMENT_ARCHITECTURES 75-real 80-real 86-real 90a-real 100f-real 120a-real
             120
        )
      elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9.0)
        list(APPEND _FRAGMENT_ARCHITECTURES 70-real 75-real 80-real 86-real 90a-real 100f-real
             120a-real 120
        )
      else()
        list(APPEND _FRAGMENT_ARCHITECTURES 70-real 75-real 80-real 86-real 90a-real 90-virtual)
      endif()
    elseif(_ARCH MATCHES "^[0-9]+[a-z]*(-real)?$")
      list(APPEND _FRAGMENT_ARCHITECTURES ${_ARCH})
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _FRAGMENT_ARCHITECTURES)
  set(${OUT_VAR}
      ${_FRAGMENT_ARCHITECTURES}
      PARENT_SCOPE
  )
endfunction()

#[=======================================================================[.rst:
add_fragment
------------

Compiles and embeds a CUDA source file as per-architecture JIT fragments.

.. code-block:: cmake

  add_fragment(<target>
               FRAGMENT <name>
               SOURCE <source>
               [ENTRY_NAME <symbol>]
               [KERNEL_ONLY]
               [DEFINITIONS <definition>...]
               [ARRAY_IDS <id>...]
               [ARRAY_VALUES <value>...]
               [LINK_LIBRARIES <target>...]
               [INCLUDE_DIRECTORIES <directory>...]
               [COMPILE_DEFINITIONS <definition>...])

The generated fragment files are embedded in ``<target>`` with names of the form
``fragments/<name>_sm<arch>.fatbin``. Fragment identifiers must use the ``_sm<arch>`` suffix so the
runtime can select fragments for the current device architecture.

``FRAGMENT``
  Fragment name used for the generated object targets and embedded blob identifiers.

``SOURCE``
  CUDA source file compiled for each fragment architecture.

``ENTRY_NAME``
  Kernel entry symbol. Defaults to ``kernel_entry``.

``KERNEL_ONLY``
  Passes ``--kernels-used=<symbol>`` to nvlink so each fragment fatbin only retains ``ENTRY_NAME``
  and device code reachable from that kernel.

``DEFINITIONS``
  Additional values passed to :cmake:command:`target_compile_definitions`.

``ARRAY_IDS`` and ``ARRAY_VALUES``
  Additional metadata arrays passed to :cmake:command:`embed_blob`.

``LINK_LIBRARIES``
  Targets linked by each fragment object target.

``INCLUDE_DIRECTORIES``
  Include directories used by each fragment object target.

``COMPILE_DEFINITIONS``
  Compile definitions used by each fragment object target.

#]=======================================================================]
macro(add_fragment)
  set(TARGET ${ARGV0})
  set(OPTIONS KERNEL_ONLY)
  set(ONE_VALUE_ARGS FRAGMENT SOURCE ENTRY_NAME)
  set(MULTI_VALUE_ARGS DEFINITIONS ARRAY_IDS ARRAY_VALUES LINK_LIBRARIES INCLUDE_DIRECTORIES
                       COMPILE_DEFINITIONS
  )
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT ARG_FRAGMENT)
    message(FATAL_ERROR "add_fragment requires FRAGMENT argument")
  endif()

  if(NOT ARG_SOURCE)
    message(FATAL_ERROR "add_fragment requires SOURCE argument")
  endif()

  if(NOT ARG_ENTRY_NAME)
    set(ARG_ENTRY_NAME "kernel_entry")
  endif()

  get_jit_fragment_architectures(FRAGMENT_ARCHITECTURES)
  foreach(FRAGMENT_ARCH IN LISTS FRAGMENT_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" FRAGMENT_ARCH_ID "${FRAGMENT_ARCH}")
    string(REGEX REPLACE "[^0-9]" "" FRAGMENT_ARCH_NUM "${FRAGMENT_ARCH_ID}")
    if(FRAGMENT_ARCH_NUM AND FRAGMENT_ARCH_NUM LESS 80)
      set(FRAGMENT_LTO OFF)
    else()
      set(FRAGMENT_LTO ON)
    endif()
    set(OBJECT_ID ${TARGET}_${ARG_FRAGMENT}_sm${FRAGMENT_ARCH_ID})
    add_library(${OBJECT_ID} OBJECT ${ARG_SOURCE})
    target_compile_options(
      ${OBJECT_ID} PRIVATE --compress-mode=size --expt-relaxed-constexpr --extended-lambda
                           -Xfatbin=--compress-all
    )

    if(ARG_KERNEL_ONLY)
      # Ensure that the FATBIN symbols only contain the specified kernel.
      target_compile_options(${OBJECT_ID} PRIVATE -Xnvlink=--kernels-used=${ARG_ENTRY_NAME})
    endif()

    target_compile_definitions(${OBJECT_ID} PRIVATE ${ARG_COMPILE_DEFINITIONS} ${ARG_DEFINITIONS})
    set_target_properties(
      ${OBJECT_ID}
      PROPERTIES CUDA_ARCHITECTURES ${FRAGMENT_ARCH}
                 CUDA_SEPARABLE_COMPILATION ON
                 CUDA_FATBIN_COMPILATION ON
                 POSITION_INDEPENDENT_CODE ON
                 INTERPROCEDURAL_OPTIMIZATION ${FRAGMENT_LTO}
                 CXX_STANDARD 20
                 CXX_STANDARD_REQUIRED ON
                 CXX_EXTENSIONS ON
                 CXX_VISIBILITY_PRESET hidden
                 CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_VISIBILITY_PRESET hidden
    )
    if(ARG_LINK_LIBRARIES)
      target_link_libraries(${OBJECT_ID} PRIVATE ${ARG_LINK_LIBRARIES})
    endif()
    if(ARG_INCLUDE_DIRECTORIES)
      target_include_directories(${OBJECT_ID} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
    endif()

    embed_blob(
      ${TARGET}
      FILE
      $<TARGET_OBJECTS:${OBJECT_ID}>
      DEST
      fragments/${ARG_FRAGMENT}_sm${FRAGMENT_ARCH_ID}.fatbin
      ID
      ${ARG_FRAGMENT}_sm${FRAGMENT_ARCH_ID}
      ARRAY_IDS
      ${ARG_ARRAY_IDS}
      fragment_arch
      ARRAY_VALUES
      ${ARG_ARRAY_VALUES}
      sm${FRAGMENT_ARCH_ID}
    )
  endforeach()
endmacro()

#[=======================================================================[.rst:
add_composite_fragment
----------------------

Pre-links LTO-IR fragments into a composite fatbin and embeds it in a target.

.. code-block:: cmake

  add_composite_fragment(<target>
                         <composite_name>
                         FRAGMENTS <fragment>...
                         [ENTRY_NAME <symbol>])

If a compatible ``nvJitLink`` executable is not available, the composite fragment is skipped.

.. note::
  This macro currently has no call sites. It is retained as supporting infrastructure for future
  pre-linked fragments.

``FRAGMENTS``
  Fragment object targets to pre-link into the composite fatbin.

``ENTRY_NAME``
  Kernel entry symbol. Defaults to ``kernel_entry``.

#]=======================================================================]
macro(add_composite_fragment)
  set(_ACF_TARGET ${ARGV0})
  set(_ACF_COMPOSITE_NAME ${ARGV1})
  set(_ACF_OPTIONS "")
  set(_ACF_ONE_VALUE_ARGS ENTRY_NAME)
  set(_ACF_MULTI_VALUE_ARGS FRAGMENTS)
  cmake_parse_arguments(
    _ACF "${_ACF_OPTIONS}" "${_ACF_ONE_VALUE_ARGS}" "${_ACF_MULTI_VALUE_ARGS}" ${ARGN}
  )

  if(NOT _ACF_COMPOSITE_NAME)
    message(FATAL_ERROR "add_composite_fragment requires COMPOSITE_NAME argument")
  endif()

  if(NOT _ACF_FRAGMENTS)
    message(FATAL_ERROR "add_composite_fragment requires FRAGMENTS argument")
  endif()

  if(NOT _ACF_ENTRY_NAME)
    set(_ACF_ENTRY_NAME "kernel_entry")
  endif()

  # nvJitLink is a library API, not a CLI tool. Look for a helper binary that wraps it. If not
  # found, the composite fragment is skipped gracefully.
  find_program(
    NVJITLINK_EXECUTABLE nvJitLink HINTS "${CUDAToolkit_BIN_DIR}"
                                         "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/bin" ENV PATH
  )

  if(NOT NVJITLINK_EXECUTABLE)
    message(STATUS "nvJitLink tool not found — composite fragment '${_ACF_COMPOSITE_NAME}' skipped")
  else()
    # Collect input fatbin files from OBJECT targets of registered fragments.
    set(_ACF_INPUT_FILES "")
    set(_ACF_INPUT_DEPS "")
    foreach(_ACF_FRAG_ID IN LISTS _ACF_FRAGMENTS)
      # Each fragment was registered as an OBJECT library named ${TARGET}_${FRAGMENT}.
      set(_ACF_OBJ_TARGET "${_ACF_TARGET}_${_ACF_FRAG_ID}")
      list(APPEND _ACF_INPUT_FILES "$<TARGET_OBJECTS:${_ACF_OBJ_TARGET}>")
      list(APPEND _ACF_INPUT_DEPS "${_ACF_OBJ_TARGET}")
    endforeach()

    # Output composite fatbin path.
    set(_ACF_COMPOSITE_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/fragments/${_ACF_COMPOSITE_NAME}.fatbin")

    # Custom command to run nvJitLink to pre-link fragments at build time.
    add_custom_command(
      OUTPUT "${_ACF_COMPOSITE_OUTPUT}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/fragments"
      COMMAND ${NVJITLINK_EXECUTABLE} -o "${_ACF_COMPOSITE_OUTPUT}" ${_ACF_INPUT_FILES}
      DEPENDS ${_ACF_INPUT_FILES}
      COMMENT "Pre-linking composite fragment: ${_ACF_COMPOSITE_NAME}"
      VERBATIM
    )

    # Custom target to ensure the composite is built before embedding.
    add_custom_target(
      "${_ACF_TARGET}_composite_${_ACF_COMPOSITE_NAME}" ALL DEPENDS "${_ACF_COMPOSITE_OUTPUT}"
    )

    # Embed the composite fatbin via the existing embed_blob infrastructure.
    embed_blob(
      ${_ACF_TARGET} FILE "${_ACF_COMPOSITE_OUTPUT}" DEST "fragments/${_ACF_COMPOSITE_NAME}.fatbin"
      ID "${_ACF_COMPOSITE_NAME}"
    )
  endif()
endmacro()
