# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
include_guard(GLOBAL)

# Compile a single existing instantiation, keeping the definitions in the two original CUDA files.
function(cudf_row_fragment SYMBOL SOURCE INSTANTIATION)
  set(ID "row_${SYMBOL}")
  set(PATH "${CMAKE_CURRENT_BINARY_DIR}/row_fragments/${SYMBOL}.cu")
  configure_file("${CUDF_SOURCE_DIR}/src/row_operator/jit/fragment.cu.in" "${PATH}" @ONLY)
  add_fragment(cudf_fragments FRAGMENT ${ID} SOURCE "${PATH}")
  set_property(GLOBAL APPEND PROPERTY CUDF_ROW_SYMBOLS "${SYMBOL}")
endfunction()

# Expand one operator signature, including the predicate alternatives of IF_ELSE.
function(cudf_row_operator MACRO)
  set(ARGS ${ARGN})
  list(GET ARGS 0 OP)
  list(GET ARGS -1 TYPE)
  if(MACRO STREQUAL "CUDF_ROW_IF_ELSE")
    foreach(PRED IN LISTS CUDF_ROW_TYPES_INTEGRALS)
      set(SIGNATURE "${ARGS};${PRED}")
      list(JOIN SIGNATURE ", " SIGNATURE)
      cudf_row_fragment(
        "cudf_row_op_${OP}_${TYPE}_${TYPE}_${PRED}" src/row_operator/jit/operators.cu
        "CUDF_ROW_IF_ELSE_PREDICATE(${SIGNATURE})"
      )
    endforeach()
    return()
  elseif(MACRO MATCHES "^CUDF_ROW_UNARY_")
    set(SYMBOL "cudf_row_op_${OP}_${TYPE}")
  elseif(MACRO STREQUAL "CUDF_ROW_BINARY_RHS")
    list(GET ARGS 3 RHS)
    set(SYMBOL "cudf_row_op_${OP}_${TYPE}_${RHS}")
  elseif(MACRO MATCHES "^CUDF_ROW_BINARY_")
    set(SYMBOL "cudf_row_op_${OP}_${TYPE}_${TYPE}")
  else()
    message(FATAL_ERROR "Unknown row operator instantiation: ${MACRO}")
  endif()
  list(JOIN ARGS ", " SIGNATURE)
  cudf_row_fragment("${SYMBOL}" src/row_operator/jit/operators.cu "${MACRO}(${SIGNATURE})")
endfunction()

# Compile the loader and setter overloads for one payload type.
function(cudf_row_payload TYPE)
  set(OFFSETS i32)
  if(TYPE STREQUAL "string_view")
    set(OFFSETS i32 i64)
  endif()
  foreach(OFFSET IN LISTS OFFSETS)
    set(TAG "")
    set(SUFFIX "")
    if(TYPE STREQUAL "string_view")
      set(TAG ", type_tag<${OFFSET}>")
      set(SUFFIX "_offset_${OFFSET}")
    endif()
    foreach(ENTRY IN ITEMS "load|LOAD" "load_list|LOAD_LIST" "set|SET" "set_list|SET_LIST"
                           "load_nullable|LOAD_NULLABLE" "load_list_nullable|LOAD_LIST_NULLABLE"
    )
      string(REPLACE "|" ";" ENTRY "${ENTRY}")
      list(GET ENTRY 0 NAME)
      list(GET ENTRY 1 MACRO)
      string(REPLACE "_nullable" "" NAME "${NAME}")
      set(SYMBOL "${NAME}_${TYPE}")
      if(MACRO MATCHES "NULLABLE$")
        string(APPEND SYMBOL "_nullable")
      endif()
      cudf_row_fragment(
        "${SYMBOL}${SUFFIX}" src/row_operator/jit/access.cu
        "CUDF_ROW_${MACRO}(${TYPE}, ${OFFSET}, ${TAG})"
      )
      foreach(INDEX IN LISTS CUDF_ROW_TYPES_INTEGERS)
        string(REPLACE "_list" "" BASE "${NAME}")
        set(DICT_LIST "")
        if(NAME MATCHES "_list$")
          set(DICT_LIST "_list")
        endif()
        set(SYMBOL "${BASE}_dict${DICT_LIST}_${TYPE}")
        if(MACRO MATCHES "NULLABLE$")
          string(APPEND SYMBOL "_nullable")
        endif()
        string(REPLACE "_LIST" "" BASE_MACRO "${MACRO}")
        string(REPLACE "_NULLABLE" "" BASE_MACRO "${BASE_MACRO}")
        set(DICT_MACRO "${BASE_MACRO}_DICT")
        if(DICT_LIST)
          string(APPEND DICT_MACRO "_LIST")
        endif()
        if(MACRO MATCHES "NULLABLE$")
          string(APPEND DICT_MACRO "_NULLABLE")
        endif()
        cudf_row_fragment(
          "${SYMBOL}_index_${INDEX}${SUFFIX}" src/row_operator/jit/access.cu
          "CUDF_ROW_${DICT_MACRO}(${TYPE}, ${INDEX}, ${OFFSET}, ${TAG})"
        )
      endforeach()
    endforeach()
  endforeach()
endfunction()

# Register all accessor families, including validity and opaque column handles.
function(cudf_row_accessors)
  foreach(
    ENTRY IN
    ITEMS "cudf_row_input_column|INPUT_COLUMN"
          "cudf_row_output_column|OUTPUT_COLUMN"
          "load_list_size|LOAD_LIST_SIZE"
          "output_list_size|OUTPUT_LIST_SIZE"
          "load_list_size_nullable|LOAD_LIST_SIZE_NULLABLE"
          "set_string_view|SET_STRING_VIEW"
          "set_validity|SET_VALIDITY"
          "set_validity_list|SET_LIST_VALIDITY"
  )
    string(REPLACE "|" ";" ENTRY "${ENTRY}")
    list(GET ENTRY 0 SYMBOL)
    list(GET ENTRY 1 MACRO)
    cudf_row_fragment("${SYMBOL}" src/row_operator/jit/access.cu "CUDF_ROW_${MACRO}()")
  endforeach()
  foreach(INDEX IN LISTS CUDF_ROW_TYPES_INTEGERS)
    foreach(SUFFIX IN ITEMS "" "_nullable")
      string(TOUPPER "LOAD_DICT_LIST_SIZE${SUFFIX}" MACRO)
      cudf_row_fragment(
        "load_dict_list_size${SUFFIX}_index_${INDEX}" src/row_operator/jit/access.cu
        "CUDF_ROW_${MACRO}(${INDEX})"
      )
    endforeach()
  endforeach()
  foreach(TYPE IN LISTS CUDF_ROW_TYPES_ALL)
    cudf_row_payload("${TYPE}")
  endforeach()
  foreach(OFFSET IN ITEMS i32 i64)
    foreach(NAME IN ITEMS data size)
      string(TOUPPER "STRING_${NAME}" MACRO)
      cudf_row_fragment(
        "string_${NAME}_offset_${OFFSET}" src/row_operator/jit/access.cu
        "CUDF_ROW_${MACRO}(${OFFSET})"
      )
    endforeach()
  endforeach()

endfunction()

# Generate the sorted host-side symbol lookup table.
function(cudf_row_catalog)
  get_property(SYMBOLS GLOBAL PROPERTY CUDF_ROW_SYMBOLS)
  list(SORT SYMBOLS)
  list(LENGTH SYMBOLS COUNT)
  set(CONTENT
      "#pragma once\n#include <cudf_fragments.hpp>\n#include <array>\n#include <string_view>\nnamespace cudf::jit {\nstruct row_fragment { std::string_view symbol; std::size_t id; };\ninline constexpr std::array<row_fragment, ${COUNT}> row_fragments{{\n"
  )
  foreach(SYMBOL IN LISTS SYMBOLS)
    string(APPEND CONTENT "{\"${SYMBOL}\", cudf_fragments::row_${SYMBOL}},\n")
  endforeach()
  string(APPEND CONTENT "}};\n}\n")
  file(
    GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/embed/row_fragments.hpp"
    CONTENT "${CONTENT}"
  )
endfunction()

# Build the ABI symbol catalog from the existing operator and accessor macros.
function(cudf_add_row_fragments)
  set(OPS "${CUDF_SOURCE_DIR}/src/row_operator/jit/operators.cu")
  set(MACROS "${CUDF_SOURCE_DIR}/src/row_operator/jit/operator_macros.cuh")
  set(ACCESS "${CUDF_SOURCE_DIR}/src/row_operator/jit/access.cu")
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${OPS}" "${MACROS}" "${ACCESS}"
  )
  # Read the type sets and compact operator instantiation list instead of maintaining a second list.
  file(READ "${MACROS}" CONTENT)
  string(REGEX MATCHALL "#define CUDF_ROW_TYPES_[A-Z]+[^#]+" SETS "${CONTENT}")
  foreach(SET IN LISTS SETS)
    string(REGEX MATCH "#define (CUDF_ROW_TYPES_[A-Z]+)" UNUSED "${SET}")
    set(GROUP "${CMAKE_MATCH_1}")
    string(REGEX MATCHALL "M\\(__VA_ARGS__, [a-zA-Z0-9_]+\\)" TYPES "${SET}")
    foreach(TYPE IN LISTS TYPES)
      string(REGEX REPLACE "M\\(__VA_ARGS__, ([a-zA-Z0-9_]+)\\)" "\\1" TYPE "${TYPE}")
      list(APPEND ${GROUP} "${TYPE}")
    endforeach()
  endforeach()
  file(READ "${OPS}" CONTENT)
  string(REGEX MATCHALL "CUDF_ROW_[A-Z_]+\\([^)]*\\)" CALLS "${CONTENT}")
  foreach(CALL IN LISTS CALLS)
    string(REGEX REPLACE "[ \n\r\t]" "" CALL "${CALL}")
    string(REGEX MATCH "^[A-Z_]+" MACRO "${CALL}")
    string(REGEX REPLACE "^[^(]*\\(|\\)$" "" ARGS "${CALL}")
    string(REPLACE "," ";" ARGS "${ARGS}")
    if(MACRO MATCHES "^CUDF_ROW_TYPES_")
      list(POP_FRONT ARGS ENTRY)
      foreach(TYPE IN LISTS ${MACRO})
        cudf_row_operator("${ENTRY}" ${ARGS} "${TYPE}")
      endforeach()
    else()
      cudf_row_operator("${MACRO}" ${ARGS})
    endif()
  endforeach()

  cudf_row_accessors()

  cudf_row_catalog()
endfunction()
