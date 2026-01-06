# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(add_jit_includes)
  set(IDENTIFIER ${ARGV0})
  set(OPTIONS "")
  set(ONE_VALUE_ARGS DIRECTORY INCLUDE_DIRECTORY)
  set(MULTI_VALUE_ARGS FILES)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT ARG_DIRECTORY AND NOT ARG_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "Either DIRECTORY or INCLUDE_DIRECTORY must be specified")
  endif()

  if(ARG_DIRECTORY AND ARG_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "Only one of DIRECTORY or INCLUDE_DIRECTORY can be specified")
  endif()

  if(ARG_INCLUDE_DIRECTORY AND ARG_FILES)
    message(FATAL_ERROR "FILES cannot be specified with INCLUDE_DIRECTORY")
  endif()

  # recursively gather all include files under the specified directory and get their paths relative
  # to the base include directory
  if(ARG_INCLUDE_DIRECTORY)
    file(GLOB_RECURSE INCLUDE_FILES "${ARG_INCLUDE_DIRECTORY}/*")
    set(INCLUDE_FILES_RELATIVE_PATHS "")
    foreach(INCLUDE_FILE IN LISTS INCLUDE_FILES)
      file(RELATIVE_PATH REL_PATH "${ARG_INCLUDE_DIRECTORY}" "${INCLUDE_FILE}")
      list(APPEND INCLUDE_FILES_RELATIVE_PATHS "${REL_PATH}")
    endforeach()
    set(ARG_DIRECTORY ${ARG_INCLUDE_DIRECTORY})
    set(ARG_FILES ${INCLUDE_FILES_RELATIVE_PATHS})
  endif()

  # check that each source file exists
  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    if(NOT EXISTS "${ARG_DIRECTORY}/${SOURCE_FILE}")
      message(FATAL_ERROR "Source file '${ARG_DIRECTORY}/${SOURCE_FILE}' does not exist")
    endif()
  endforeach(SOURCE_FILE)

  set(SOURCE_FILES ${${IDENTIFIER}_sources_file_paths})
  set(INCLUDE_NAMES ${${IDENTIFIER}_sources_include_names})

  foreach(SOURCE_FILE IN LISTS ARG_FILES)
    list(APPEND SOURCE_FILES "${ARG_DIRECTORY}/${SOURCE_FILE}")
    list(APPEND INCLUDE_NAMES "${SOURCE_FILE}")
  endforeach()

  set(${IDENTIFIER}_sources_file_paths
      ${SOURCE_FILES}
      PARENT_SCOPE
  )
  set(${IDENTIFIER}_sources_include_names
      ${INCLUDE_NAMES}
      PARENT_SCOPE
  )

endfunction()

function(add_jit_options)
  set(IDENTIFIER ${ARGV0})
  set(OPTIONS "")
  set(ONE_VALUE_ARGS "")
  set(MULTI_VALUE_ARGS "OPTIONS")
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  set(${IDENTIFIER}_options
      "${ARG_OPTIONS}"
      PARENT_SCOPE
  )

endfunction()

# pass the encoded args to the jit_embed.py script to generate the source and options maps
function(generate_jit_source_map)
  set(ONE_VALUE_ARGS "TARGET")
  cmake_parse_arguments(ARG "" "${ONE_VALUE_ARGS}" "" ${ARGN})

  list(LENGTH ${ARG_TARGET}_sources_include_names NUM_SOURCES)
  list(LENGTH ${ARG_TARGET}_options NUM_OPTIONS)

  set(TARGET_YAML " - id: \"${ARG_TARGET}_sources\"\n   type: \"sources\"\n   sources:\n")

  math(EXPR LAST_SOURCE_INDEX "${NUM_SOURCES} - 1")
  foreach(i RANGE 0 ${LAST_SOURCE_INDEX})
    list(GET ${ARG_TARGET}_sources_include_names ${i} INCLUDE_NAME)
    list(GET ${ARG_TARGET}_sources_file_paths ${i} SOURCE_FILE_PATH)
    set(TARGET_YAML "${TARGET_YAML}    - include_name: \"${INCLUDE_NAME}\"\n")
    set(TARGET_YAML "${TARGET_YAML}      file_path: \"${SOURCE_FILE_PATH}\"\n")
  endforeach()

  set(TARGET_YAML "${TARGET_YAML}\n\n")
  set(TARGET_YAML
      "${TARGET_YAML} - id: \"${ARG_TARGET}_options\"\n   type: \"options\"\n   options:\n"
  )

  math(EXPR LAST_OPTION_INDEX "${NUM_OPTIONS} - 1")
  foreach(i RANGE 0 ${LAST_OPTION_INDEX})
    list(GET ${ARG_TARGET}_options ${i} OPTION)
    set(TARGET_YAML "${TARGET_YAML}    - \"${OPTION}\"\n")
  endforeach()

  # write CONFIG to temp file and pass file path to script
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}.yaml" "${TARGET_YAML}")

  set(INCLUDE_DIR "${CUDF_GENERATED_INCLUDE_DIR}/include/jit_embed")
  set(HEADER "${INCLUDE_DIR}/${ARG_TARGET}.h")

  add_custom_command(
    OUTPUT ${HEADER}
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/jit_embed.py" --output
            "${HEADER}" --input-file "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}.yaml"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/jit_embed.py"
            ${${ARG_TARGET}_sources_file_paths}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating JIT source map for ${ARG_TARGET} into ${HEADER}"
    VERBATIM
  )

  add_custom_target(${ARG_TARGET} ALL DEPENDS "${HEADER}")

  message("Generated into: ${HEADER}")

  set(${ARG_TARGET}_INCLUDE_DIR
      ${INCLUDE_DIR}
      PARENT_SCOPE
  )
  set(${ARG_TARGET}_HEADER
      ${HEADER}
      PARENT_SCOPE
  )

endfunction()
