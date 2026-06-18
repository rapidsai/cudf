# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(NOT TARGET zstd)
  message(FATAL_ERROR "zstd target is required for LIBRTCX embedding.")
endif()

if(NOT TARGET xxhash)
  message(FATAL_ERROR "xxhash target is required for LIBRTCX embedding.")
endif()

# This function initializes a target for JIT embedding. It must be called before any calls to
# embed_includes() or embed_blob() for the target. It creates a dedicated INTERFACE library target
# that is used to track registered files and dependencies via target properties. The TARGET argument
# specifies the name of the target being initialized.
function(add_embed TARGET)
  set(OPTIONS "")
  set(ONE_VALUE_ARGS)
  set(MULTI_VALUE_ARGS)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET argument is required")
  endif()

  add_library(${TARGET}__embed_props INTERFACE)
  set_property(TARGET ${TARGET}__embed_props PROPERTY EMBED_FILE_INDEX 0)
endfunction()

# This function registers a directory of include files to be embedded for JIT compilation.
function(embed_includes TARGET)
  set(OPTIONS "")
  set(ONE_VALUE_ARGS SOURCE_DIRECTORY # Source directory where files will be copied from
                     DEST_DIRECTORY # Destination directory where files will be copied to
  )
  set(MULTI_VALUE_ARGS
      FILES # Source files relative to SOURCE_DIRECTORY (optional, if not provided, all files under
            # SOURCE_DIRECTORY will be used)
      INCLUDE_DIRECTORIES # Include directories to be used when compiling with these files
  )
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT TARGET ${TARGET}__embed_props)
    message(FATAL_ERROR "embed target '${TARGET}' has not been initialized with add_embed()")
  endif()

  if(NOT ARG_SOURCE_DIRECTORY
     OR NOT ARG_DEST_DIRECTORY
     OR NOT ARG_INCLUDE_DIRECTORIES
  )
    message(
      FATAL_ERROR "SOURCE_DIRECTORY, DEST_DIRECTORY, and INCLUDE_DIRECTORIES arguments are required"
    )
  endif()

  if(NOT ARG_FILES)
    # gather all include files under the specified directory
    file(GLOB_RECURSE INCLUDE_FILES "${ARG_SOURCE_DIRECTORY}/*")

    # get their paths relative to the base include directory
    set(INCLUDE_FILES_RELATIVE_PATHS "")
    foreach(INCLUDE_FILE IN LISTS INCLUDE_FILES)
      file(RELATIVE_PATH INCLUDE_FILE_REL_PATH "${ARG_SOURCE_DIRECTORY}" "${INCLUDE_FILE}")
      list(APPEND INCLUDE_FILES_RELATIVE_PATHS "${INCLUDE_FILE_REL_PATH}")
    endforeach()

    set(ARG_FILES ${INCLUDE_FILES_RELATIVE_PATHS})
  endif()

  # check that each source file exists
  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    if(NOT EXISTS "${ARG_SOURCE_DIRECTORY}/${SOURCE_FILE}")
      message(FATAL_ERROR "Source file '${ARG_SOURCE_DIRECTORY}/${SOURCE_FILE}' does not exist")
    endif()
  endforeach(SOURCE_FILE)

  # Determine the starting index for new IDs from the current list length
  get_property(
    SOURCE_FILE_IDS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILE_IDS
  )
  list(LENGTH SOURCE_FILE_IDS IDX)

  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_SOURCE_FILE_IDS "include_${IDX}"
    )
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_SOURCE_FILES "${ARG_SOURCE_DIRECTORY}/${SOURCE_FILE}"
    )
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_SOURCE_FILE_DESTS "${ARG_DEST_DIRECTORY}/${SOURCE_FILE}"
    )
    math(EXPR IDX "${IDX} + 1")
  endforeach()

  set_property(
    TARGET ${TARGET}__embed_props
    APPEND
    PROPERTY EMBED_INCLUDE_DIRECTORIES ${ARG_INCLUDE_DIRECTORIES}
  )

  get_property(
    SOURCE_FILE_IDS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILE_IDS
  )
  list(LENGTH SOURCE_FILE_IDS IDX)

  set_property(TARGET ${TARGET}__embed_props PROPERTY EMBED_FILE_INDEX ${IDX})

endfunction()

# This function registers a single file to be embedded for JIT compilation.
function(embed_blob TARGET)
  set(OPTIONS)
  set(ONE_VALUE_ARGS ID FILE DEST)
  set(MULTI_VALUE_ARGS ARRAY_IDS ARRAY_VALUES)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT TARGET ${TARGET}__embed_props)
    message(FATAL_ERROR "embed target '${TARGET}' has not been initialized with add_embed()")
  endif()

  if(NOT ARG_ID
     OR NOT ARG_FILE
     OR NOT ARG_DEST
  )
    message(FATAL_ERROR "ID, FILE, and DEST arguments are required")
  endif()

  if(ARG_ARRAY_IDS)
    if(NOT ARG_ARRAY_VALUES)
      message(FATAL_ERROR "ARRAY_VALUES argument is required when ARRAY_IDS is provided")
    endif()

    list(LENGTH ARG_ARRAY_IDS ARG_ARRAY_IDS_LENGTH)
    list(LENGTH ARG_ARRAY_VALUES ARG_ARRAY_VALUES_LENGTH)

    if(NOT ARG_ARRAY_IDS_LENGTH EQUAL ARG_ARRAY_VALUES_LENGTH)
      message(FATAL_ERROR "ARRAY_IDS and ARRAY_VALUES must have the same length")
    endif()

    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_ARRAY_IDS ${ARG_ARRAY_IDS}
    )
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_ARRAY_VALUES ${ARG_ARRAY_VALUES}
    )
  endif()

  if(ARG_FILE MATCHES "\\$<TARGET_OBJECTS:([^>]+)>")
    # If the file is a generator expression for target objects add as dependency
    set_property(
      TARGET ${TARGET}__embed_props
      APPEND
      PROPERTY EMBED_TARGET_DEPS $<TARGET_OBJECTS:${CMAKE_MATCH_1}>
    )
  else()
    if(NOT EXISTS "${ARG_FILE}")
      message(FATAL_ERROR "Source file '${ARG_FILE}' does not exist")
    endif()
  endif()

  set_property(
    TARGET ${TARGET}__embed_props
    APPEND
    PROPERTY EMBED_SOURCE_FILE_IDS ${ARG_ID}
  )
  set_property(
    TARGET ${TARGET}__embed_props
    APPEND
    PROPERTY EMBED_SOURCE_FILES ${ARG_FILE}
  )
  set_property(
    TARGET ${TARGET}__embed_props
    APPEND
    PROPERTY EMBED_SOURCE_FILE_DESTS ${ARG_DEST}
  )

  get_property(
    SOURCE_FILE_IDS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILE_IDS
  )
  list(LENGTH SOURCE_FILE_IDS IDX)

  set_property(TARGET ${TARGET}__embed_props PROPERTY EMBED_FILE_INDEX ${IDX})

endfunction()

#[==[
# This function generates the necessary files and build targets to embed the registered source files
# for JIT compilation.
#]==]
# cmake-lint: disable=R0915
function(embed TARGET)
  set(OPTIONS "")
  set(ONE_VALUE_ARGS "COMPRESSION" "OUTPUT_DIRECTORY")
  set(MULTI_VALUE_ARGS "")
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT TARGET ${TARGET}__embed_props)
    message(FATAL_ERROR "embed target '${TARGET}' has not been initialized with add_embed()")
  endif()

  if(NOT DEFINED ARG_COMPRESSION)
    message(FATAL_ERROR "COMPRESSION argument is required")
  endif()

  if(NOT ARG_COMPRESSION STREQUAL "none" AND NOT ARG_COMPRESSION STREQUAL "zstd")
    message(FATAL_ERROR "COMPRESSION argument must be either none or zstd")
  endif()

  if(NOT DEFINED ARG_OUTPUT_DIRECTORY)
    message(FATAL_ERROR "OUTPUT_DIRECTORY argument is required")
  endif()

  get_property(
    EMBED_SOURCE_FILES
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILES
  )
  if(NOT EMBED_SOURCE_FILES)
    message(FATAL_ERROR "No source files registered for target '${TARGET}'")
  endif()

  get_property(
    EMBED_SOURCE_FILE_IDS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILE_IDS
  )
  get_property(
    EMBED_SOURCE_FILE_DESTS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_SOURCE_FILE_DESTS
  )
  get_property(
    EMBED_TARGET_DEPS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_TARGET_DEPS
  )
  get_property(
    EMBED_ARRAY_IDS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_ARRAY_IDS
  )
  get_property(
    EMBED_ARRAY_VALUES
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_ARRAY_VALUES
  )
  get_property(
    EMBED_INCLUDE_DIRS
    TARGET ${TARGET}__embed_props
    PROPERTY EMBED_INCLUDE_DIRECTORIES
  )

  set(OUTPUT_DIR "${ARG_OUTPUT_DIRECTORY}")
  set(EMBED_SCRIPT_TEMPLATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed.in.cpp")
  set(CONFIGURED_EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}__embed_cfg.cpp")
  set(EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}__embed.cpp")

  set(EMBED_SCRIPT__ID "${TARGET}")
  set(EMBED_SCRIPT__ARRAY_IDS "${EMBED_ARRAY_IDS}")
  set(EMBED_SCRIPT__ARRAY_VALUES "${EMBED_ARRAY_VALUES}")
  set(EMBED_SCRIPT__FILE_IDS "${EMBED_SOURCE_FILE_IDS}")
  set(EMBED_SCRIPT__FILE_PATHS "${EMBED_SOURCE_FILES}")
  set(EMBED_SCRIPT__FILE_DESTS "${EMBED_SOURCE_FILE_DESTS}")
  set(EMBED_SCRIPT__INCLUDE_DIRS "${EMBED_INCLUDE_DIRS}")
  set(EMBED_SCRIPT__COMPRESSION "${ARG_COMPRESSION}")
  set(EMBED_SCRIPT__OUTPUT_DIR "${OUTPUT_DIR}")

  configure_file(${EMBED_SCRIPT_TEMPLATE} ${CONFIGURED_EMBED_SCRIPT} @ONLY)
  file(
    GENERATE
    OUTPUT "${EMBED_SCRIPT}"
    INPUT "${CONFIGURED_EMBED_SCRIPT}"
  )

  set(RUNNER "${TARGET}__jit_embed_run")
  add_executable(${RUNNER} EXCLUDE_FROM_ALL "${EMBED_SCRIPT}")
  target_link_libraries(${RUNNER} PRIVATE ${CMAKE_DL_LIBS} zstd xxhash)
  target_include_directories(
    ${RUNNER} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR} ${ZSTD_INCLUDE_DIR}
  )
  set_target_properties(${RUNNER} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)

  add_custom_command(
    OUTPUT ${OUTPUT_DIR}/${TARGET}.hpp ${OUTPUT_DIR}/${TARGET}.s ${OUTPUT_DIR}/${TARGET}.bin
    COMMAND "${CMAKE_COMMAND}" -E env $<TARGET_FILE:${RUNNER}>
    DEPENDS "${EMBED_SCRIPT}" ${EMBED_SOURCE_FILES} ${EMBED_TARGET_DEPS}
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
