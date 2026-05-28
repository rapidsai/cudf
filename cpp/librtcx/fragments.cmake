# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# cmake-lint: disable=C0112,C0113,E1120

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

  get_property(FRAGMENT_ARCHITECTURES GLOBAL PROPERTY rapids_cuda_architectures)
  if(NOT FRAGMENT_ARCHITECTURES)
    set(FRAGMENT_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  endif()
  foreach(FRAGMENT_ARCH IN LISTS FRAGMENT_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" FRAGMENT_ARCH_ID "${FRAGMENT_ARCH}")
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
                 INTERPROCEDURAL_OPTIMIZATION ON
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
