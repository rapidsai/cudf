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
