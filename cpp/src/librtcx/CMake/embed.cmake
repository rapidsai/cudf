# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# TODO: add glob restrictions:exclude .cpp .cc, .cmake and .cxx source files, only include .h .hpp
# .cuh .cu files

# This function registers a directory of include files to be embedded for JIT compilation. It
# gathers the specified files, their destinations, and include directories, and stores them in
# target-specific variables for later use when generating the embed.
function(jit_add_include_directory)
  set(TARGET ${ARGV0})
  set(OPTIONS "")
  set(ONE_VALUE_ARGS COPY_DIRECTORY # Source directory where files will be copied from
                     DEST_DIRECTORY # Destination directory where files will be copied to
  )
  set(MULTI_VALUE_ARGS
      FILES # Source files relative to COPY_DIRECTORY (optional, if not provided, all files under
            # COPY_DIRECTORY will be used)
      INCLUDE_DIRECTORIES # Include directories to be used when compiling with these files
  )
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET argument is required")
  endif()

  if(NOT ARG_COPY_DIRECTORY)
    message(FATAL_ERROR "COPY_DIRECTORY argument is required")
  endif()

  if(NOT ARG_DEST_DIRECTORY)
    message(FATAL_ERROR "DEST_DIRECTORY argument is required")
  endif()

  if(NOT ARG_INCLUDE_DIRECTORIES)
    message(FATAL_ERROR "INCLUDE_DIRECTORIES argument is required")
  endif()

  if(NOT ARG_FILES)
    # gather all include files under the specified directory
    file(GLOB_RECURSE INCLUDE_FILES "${ARG_COPY_DIRECTORY}/*")

    # get their paths relative to the base include directory
    set(INCLUDE_FILES_RELATIVE_PATHS "")
    foreach(INCLUDE_FILE IN LISTS INCLUDE_FILES)
      file(RELATIVE_PATH INCLUDE_FILE_REL_PATH "${ARG_COPY_DIRECTORY}" "${INCLUDE_FILE}")
      list(APPEND INCLUDE_FILES_RELATIVE_PATHS "${INCLUDE_FILE_REL_PATH}")
    endforeach()

    set(ARG_FILES ${INCLUDE_FILES_RELATIVE_PATHS})
  endif()

  # check that each source file exists
  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    if(NOT EXISTS "${ARG_COPY_DIRECTORY}/${SOURCE_FILE}")
      message(FATAL_ERROR "Source file '${ARG_COPY_DIRECTORY}/${SOURCE_FILE}' does not exist")
    endif()
  endforeach(SOURCE_FILE)

  # Set scope variables to accumulate results

  set(SOURCE_FILES ${jitembed_${TARGET}_incdir__source_files})
  set(SOURCE_FILE_DESTS ${jitembed_${TARGET}_incdir__source_file_dests})
  set(INCLUDE_DIRECTORIES ${jitembed_${TARGET}_incdir__include_directories})

  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    list(APPEND SOURCE_FILES "${ARG_COPY_DIRECTORY}/${SOURCE_FILE}")
    list(APPEND SOURCE_FILE_DESTS "${ARG_DEST_DIRECTORY}/${SOURCE_FILE}")
  endforeach()

  list(APPEND INCLUDE_DIRECTORIES ${ARG_INCLUDE_DIRECTORIES})

  set(jitembed_${TARGET}_incdir__source_files
      ${SOURCE_FILES}
      PARENT_SCOPE
  )
  set(jitembed_${TARGET}_incdir__source_file_dests
      ${SOURCE_FILE_DESTS}
      PARENT_SCOPE
  )
  set(jitembed_${TARGET}_incdir__include_directories
      ${INCLUDE_DIRECTORIES}
      PARENT_SCOPE
  )

endfunction()

# pass the encoded args to the embed.py script to generate the embed
function(jit_embed)
  set(TARGET ${ARGV0})
  set(OPTIONS "")
  set(ONE_VALUE_ARGS "COMPRESSION")
  set(MULTI_VALUE_ARGS "")
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET argument is required")
  endif()

  if(NOT DEFINED ARG_COMPRESSION)
    message(FATAL_ERROR "COMPRESSION argument is required")
  endif()

  if(NOT ARG_COMPRESSION STREQUAL "none" AND NOT ARG_COMPRESSION STREQUAL "zstd")
    message(FATAL_ERROR "COMPRESSION argument must be either none or zstd")
  endif()

  if(NOT DEFINED jitembed_${TARGET}_incdir__source_files)
    message(
      FATAL_ERROR
        "No source files registered for target '${TARGET}'. Call jit_add_include_directory() first"
    )
  endif()

  set(OUTPUT_DIR "${CUDF_GENERATED_INCLUDE_DIR}/rtcx_embed")
  set(CONFIGURED_EMBED_PY "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_embed.py")
  set(EMBED_PY_IN "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed.in.py")

  set(RTCX_EMBED_PY_ARG__ID "${TARGET}")
  set(RTCX_EMBED_PY_ARG__FILE_PATHS "${jitembed_${TARGET}_incdir__source_files}")
  set(RTCX_EMBED_PY_ARG__FILE_DESTS "${jitembed_${TARGET}_incdir__source_file_dests}")
  set(RTCX_EMBED_PY_ARG__INCLUDE_DIRS "${jitembed_${TARGET}_incdir__include_directories}")
  set(RTCX_EMBED_PY_ARG__COMPRESSION "${ARG_COMPRESSION}")
  set(RTCX_EMBED_PY_ARG__OUTPUT_DIR "${OUTPUT_DIR}")

  configure_file("${EMBED_PY_IN}" "${CONFIGURED_EMBED_PY}" @ONLY)

  add_custom_command(
    OUTPUT ${OUTPUT_DIR}/${TARGET}.hpp ${OUTPUT_DIR}/${TARGET}.s ${OUTPUT_DIR}/${TARGET}.bin
    BYPRODUCTS ${OUTPUT_DIR}/*
    COMMAND ${Python3_EXECUTABLE} "${CONFIGURED_EMBED_PY}"
    DEPENDS "${EMBED_PY_IN}" "${CONFIGURED_EMBED_PY}" ${jitembed_${TARGET}_incdir__source_files}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating JIT embed for ${TARGET} into ${OUTPUT_DIR}"
    VERBATIM
  )

  add_custom_target(
    ${TARGET} ALL
    DEPENDS ${OUTPUT_DIR}/${TARGET}.hpp ${OUTPUT_DIR}/${TARGET}.s ${OUTPUT_DIR}/${TARGET}.bin
    COMMENT "Custom target for JIT embed of ${TARGET}"
  )

  message(
    STATUS
      "JIT embed for target ${TARGET} will be generated into: ${OUTPUT_DIR}/${TARGET}.hpp ${OUTPUT_DIR}/${TARGET}.s ${OUTPUT_DIR}/${TARGET}.bin"
  )

  set(${TARGET}_INCLUDE_DIRS
      "${OUTPUT_DIR}"
      PARENT_SCOPE
  )

  set(${TARGET}_SOURCE_DIR
      ${OUTPUT_DIR}
      PARENT_SCOPE
  )

endfunction()
