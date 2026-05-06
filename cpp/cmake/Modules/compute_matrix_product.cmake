# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

function(compute_matrix_product output_var)
  set(options)
  set(one_value MATRIX_JSON_FILE MATRIX_JSON_STRING)
  set(multi_value)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  if(_JIT_LTO_MATRIX_JSON_FILE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compute_matrix_product.py"
              "${_JIT_LTO_MATRIX_JSON_FILE}" #
      OUTPUT_VARIABLE output COMMAND_ERROR_IS_FATAL ANY
    )
  else()
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E echo "${_JIT_LTO_MATRIX_JSON_STRING}"
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compute_matrix_product.py"
              -
      OUTPUT_VARIABLE output COMMAND_ERROR_IS_FATAL ANY
    )
  endif()

  set(${output_var}
      "${output}"
      PARENT_SCOPE
  )
endfunction()

function(populate_matrix_variables matrix_json_entry)
  string(JSON len LENGTH "${matrix_json_entry}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON key MEMBER "${matrix_json_entry}" "${i}")
    string(JSON value GET "${matrix_json_entry}" "${key}")
    set(${key}
        "${value}"
        PARENT_SCOPE
    )
  endforeach()
endfunction()
