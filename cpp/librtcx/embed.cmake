# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(NOT TARGET zstd)
  message(
    FATAL_ERROR "zstd library is required for JIT embedding. Please ensure it is found by CMake."
  )
endif()

# This function initializes a target for JIT embedding. It must be called before any calls to
# embed_includes() or embed_blob() for the target. It sets up necessary variables and state to track
# the registered files and dependencies for the target. The TARGET argument specifies the name of
# the target being initialized.
function(add_embed TARGET)
  set(OPTIONS "")
  set(ONE_VALUE_ARGS)
  set(MULTI_VALUE_ARGS)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET argument is required")
  endif()

  set(${TARGET}_FILE_INDEX
      0
      PARENT_SCOPE
  )
  set(${TARGET}_INITIALIZED
      TRUE
      PARENT_SCOPE
  )
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

  if(NOT DEFINED TARGET OR NOT ${TARGET}_INITIALIZED)
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

  # Set scope variables to accumulate results

  set(SOURCE_FILE_IDS ${${TARGET}__embed__source_file_ids})
  set(SOURCE_FILES ${${TARGET}__embed__source_files})
  set(SOURCE_FILE_DESTS ${${TARGET}__embed__source_file_dests})
  set(INCLUDE_DIRECTORIES ${${TARGET}__embed__include_directories})

  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    list(LENGTH SOURCE_FILE_IDS SOURCE_FILE_IDS_LENGTH)
    list(APPEND SOURCE_FILE_IDS "include_${SOURCE_FILE_IDS_LENGTH}")
    list(APPEND SOURCE_FILES "${ARG_SOURCE_DIRECTORY}/${SOURCE_FILE}")
    list(APPEND SOURCE_FILE_DESTS "${ARG_DEST_DIRECTORY}/${SOURCE_FILE}")
  endforeach()

  list(APPEND INCLUDE_DIRECTORIES ${ARG_INCLUDE_DIRECTORIES})

  list(LENGTH SOURCE_FILE_IDS SOURCE_FILE_IDS_LENGTH)

  set(${TARGET}_FILE_INDEX
      ${SOURCE_FILE_IDS_LENGTH}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__source_file_ids
      ${SOURCE_FILE_IDS}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__source_files
      ${SOURCE_FILES}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__source_file_dests
      ${SOURCE_FILE_DESTS}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__include_directories
      ${INCLUDE_DIRECTORIES}
      PARENT_SCOPE
  )

endfunction()

# This function registers a single file to be embedded for JIT compilation.
function(embed_blob TARGET)
  set(OPTIONS)
  set(ONE_VALUE_ARGS ID FILE DEST)
  set(MULTI_VALUE_ARGS ARRAY_IDS ARRAY_VALUES)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT DEFINED TARGET OR NOT ${TARGET}_INITIALIZED)
    message(FATAL_ERROR "embed target '${TARGET}' has not been initialized with add_embed()")
  endif()

  if(NOT ARG_ID
     OR NOT ARG_FILE
     OR NOT ARG_DEST
  )
    message(FATAL_ERROR "ID, FILE, and DEST arguments are required")
  endif()

  set(SOURCE_FILE_IDS ${${TARGET}__embed__source_file_ids})
  set(SOURCE_FILES ${${TARGET}__embed__source_files})
  set(SOURCE_FILE_DESTS ${${TARGET}__embed__source_file_dests})
  set(TARGET_DEPS ${${TARGET}__embed__target_deps})
  set(ARRAY_IDS ${${TARGET}__embed__array_ids})
  set(ARRAY_VALUES ${${TARGET}__embed__array_values})

  if(ARG_ARRAY_IDS)
    if(NOT ARG_ARRAY_VALUES)
      message(FATAL_ERROR "ARRAY_VALUES argument is required when ARRAY_IDS is provided")
    endif()

    list(LENGTH ARG_ARRAY_IDS ARG_ARRAY_IDS_LENGTH)
    list(LENGTH ARG_ARRAY_VALUES ARG_ARRAY_VALUES_LENGTH)

    if(NOT ARG_ARRAY_IDS_LENGTH EQUAL ARG_ARRAY_VALUES_LENGTH)
      message(FATAL_ERROR "ARRAY_IDS and ARRAY_VALUES must have the same length")
    endif()

    list(APPEND ARRAY_IDS ${ARG_ARRAY_IDS})
    list(APPEND ARRAY_VALUES ${ARG_ARRAY_VALUES})
  endif()

  if(ARG_FILE MATCHES "\\$<TARGET_OBJECTS:([^>]+)>")
    # If the file is a generator expression for target objects add as dependency
    list(APPEND TARGET_DEPS $<TARGET_OBJECTS:${CMAKE_MATCH_1}>)
  else()
    if(NOT EXISTS "${ARG_FILE}")
      message(FATAL_ERROR "Source file '${ARG_FILE}' does not exist")
    endif()
  endif()
  list(APPEND SOURCE_FILE_IDS ${ARG_ID})
  list(APPEND SOURCE_FILES ${ARG_FILE})
  list(APPEND SOURCE_FILE_DESTS ${ARG_DEST})

  list(LENGTH SOURCE_FILE_IDS SOURCE_FILE_IDS_LENGTH)

  set(${TARGET}_FILE_INDEX
      ${SOURCE_FILE_IDS_LENGTH}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__source_file_ids
      ${SOURCE_FILE_IDS}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__source_files
      ${SOURCE_FILES}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__source_file_dests
      ${SOURCE_FILE_DESTS}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__target_deps
      ${TARGET_DEPS}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__array_ids
      ${ARRAY_IDS}
      PARENT_SCOPE
  )
  set(${TARGET}__embed__array_values
      ${ARRAY_VALUES}
      PARENT_SCOPE
  )

endfunction()

# This function generates the necessary files and build targets to embed the registered source files
# for JIT compilation.
function(embed TARGET)
  set(OPTIONS "")
  set(ONE_VALUE_ARGS "COMPRESSION")
  set(MULTI_VALUE_ARGS "")
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT DEFINED TARGET OR NOT ${TARGET}_INITIALIZED)
    message(FATAL_ERROR "embed target '${TARGET}' has not been initialized with add_embed()")
  endif()

  if(NOT DEFINED ARG_COMPRESSION)
    message(FATAL_ERROR "COMPRESSION argument is required")
  endif()

  if(NOT ARG_COMPRESSION STREQUAL "none" AND NOT ARG_COMPRESSION STREQUAL "zstd")
    message(FATAL_ERROR "COMPRESSION argument must be either none or zstd")
  endif()

  if(NOT DEFINED ${TARGET}__embed__source_files)
    message(FATAL_ERROR "No source files registered for target '${TARGET}'")
  endif()

  set(OUTPUT_DIR "${CUDF_GENERATED_INCLUDE_DIR}/rtcx_embed")
  set(EMBED_SCRIPT_TEMPLATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/embed.in.cpp")
  set(CONFIGURED_EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}__embed_cfg.cpp")
  set(EMBED_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}__embed.cpp")

  set(EMBED_SCRIPT__ID "${TARGET}")
  set(EMBED_SCRIPT__ARRAY_IDS "${${TARGET}__embed__array_ids}")
  set(EMBED_SCRIPT__ARRAY_VALUES "${${TARGET}__embed__array_values}")
  set(EMBED_SCRIPT__FILE_IDS "${${TARGET}__embed__source_file_ids}")
  set(EMBED_SCRIPT__FILE_PATHS "${${TARGET}__embed__source_files}")
  set(EMBED_SCRIPT__FILE_DESTS "${${TARGET}__embed__source_file_dests}")
  set(EMBED_SCRIPT__INCLUDE_DIRS "${${TARGET}__embed__include_directories}")
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
  target_include_directories(${RUNNER} PRIVATE ${ZSTD_INCLUDE_DIR})
  target_link_libraries(${RUNNER} PRIVATE ${CMAKE_DL_LIBS} zstd)
  set_target_properties(
    ${RUNNER} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES
  )
  target_include_directories(${RUNNER} PRIVATE ${CMAKE_CURRENT_FUNCTION_LIST_DIR})

  add_custom_command(
    OUTPUT ${OUTPUT_DIR}/${TARGET}.hpp ${OUTPUT_DIR}/${TARGET}.s ${OUTPUT_DIR}/${TARGET}.bin
    COMMAND "${CMAKE_COMMAND}" -E env $<TARGET_FILE:${RUNNER}>
    DEPENDS "${EMBED_SCRIPT}" ${${TARGET}__embed__source_files}
            ${${TARGET}__embed__target_deps}
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
