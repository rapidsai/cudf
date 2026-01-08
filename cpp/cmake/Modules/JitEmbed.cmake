# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter)

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

function(jit_add_options)
  set(TARGET ${ARGV0})
  set(OPTIONS "")
  set(ONE_VALUE_ARGS "")
  set(MULTI_VALUE_ARGS "OPTIONS")
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET argument is required")
  endif()

  if(NOT ARG_OPTIONS)
    message(FATAL_ERROR "OPTIONS argument is required")
  endif()

  set(options ${jitembed_options_${TARGET}_options})
  foreach(OPTION IN LISTS ARG_OPTIONS)
    list(APPEND options "${OPTION}")
  endforeach()

  set(jitembed_options_${TARGET}_options
      ${options}
      PARENT_SCOPE
  )

endfunction()

# pass the encoded args to the jit_embed.py script to generate the source and options maps
function(jit_embed)
  set(TARGET ${ARGV0})
  cmake_parse_arguments(ARG "" "${ONE_VALUE_ARGS}" "" ${ARGN})

  if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET argument is required")
  endif()

  string(APPEND TARGET_YAML "\"${TARGET}_sources\":\n")
  string(APPEND TARGET_YAML " type: \"sources\"\n")

  if(DEFINED jitembed_${TARGET}_incdir__source_files)

    # gather source files
    string(APPEND TARGET_YAML " sources:\n")
    list(LENGTH jitembed_${TARGET}_incdir__source_files NUM_SOURCES)

    math(EXPR LAST_SOURCE_INDEX "${NUM_SOURCES} - 1")
    foreach(i RANGE 0 ${LAST_SOURCE_INDEX})
      list(GET jitembed_${TARGET}_incdir__source_files ${i} SOURCE_FILE)
      list(GET jitembed_${TARGET}_incdir__source_file_dests ${i} SOURCE_FILE_DEST)
      string(APPEND TARGET_YAML "  - file: \"${SOURCE_FILE}\"\n")
      string(APPEND TARGET_YAML "    dest: \"${SOURCE_FILE_DEST}\"\n")
    endforeach()

    # gather include directories

    string(APPEND TARGET_YAML " include_directories:\n")
    list(LENGTH jitembed_${TARGET}_incdir__include_directories NUM_INCLUDE_DIRS)
    math(EXPR LAST_INCLUDE_DIR_INDEX "${NUM_INCLUDE_DIRS} - 1")
    foreach(i RANGE 0 ${LAST_INCLUDE_DIR_INDEX})
      list(GET jitembed_${TARGET}_incdir__include_directories ${i} INCLUDE_DIR)
      string(APPEND TARGET_YAML "  - \"${INCLUDE_DIR}\"\n")
    endforeach()

  endif()

  string(APPEND TARGET_YAML "\n\n")

  if(DEFINED jitembed_options_${TARGET}_options)

    string(APPEND TARGET_YAML "\"${TARGET}_options\":\n")
    string(APPEND TARGET_YAML " type: \"strings\"\n")

    # gather options
    string(APPEND TARGET_YAML " strings:\n")

    list(LENGTH jitembed_options_${TARGET}_options NUM_OPTIONS)
    math(EXPR LAST_OPTION_INDEX "${NUM_OPTIONS} - 1")
    foreach(i RANGE 0 ${LAST_OPTION_INDEX})
      list(GET jitembed_options_${TARGET}_options ${i} OPTION)
      string(APPEND TARGET_YAML "  - \"${OPTION}\"\n")
    endforeach()

  endif()

  set(YAML_FILE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.yaml")
  set(INCLUDE_DIR "${CUDF_GENERATED_INCLUDE_DIR}/include/jit_embed")
  set(HEADER "${INCLUDE_DIR}/${TARGET}.h")

  # write CONFIG to temp file and pass file path to script
  file(WRITE "${YAML_FILE_PATH}" "${TARGET_YAML}")

  add_custom_command(
    OUTPUT ${HEADER}
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/jit_embed.py" --output
            "${HEADER}" --input "${YAML_FILE_PATH}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/jit_embed.py" "${YAML_FILE_PATH}"
            ${jitembed_${TARGET}_incdir__source_files}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating JIT embed for ${TARGET} (YAML: ${YAML_FILE_PATH}) into ${HEADER}"
    VERBATIM
  )

  add_custom_target(${TARGET} ALL DEPENDS "${HEADER}")

  message(
    STATUS
      "JIT embed for target ${TARGET} (YAML: ${YAML_FILE_PATH}) will be generated into: ${HEADER}"
  )

  set(${TARGET}_INCLUDE_DIR
      ${INCLUDE_DIR}
      PARENT_SCOPE
  )
  set(${TARGET}_HEADER
      ${HEADER}
      PARENT_SCOPE
  )

endfunction()
