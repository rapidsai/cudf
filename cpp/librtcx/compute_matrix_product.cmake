# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
# cmake-lint: disable=C0103,C0112,E1120

include_guard(GLOBAL)

#[=======================================================================[.rst:
compute_matrix_product
----------------------

Computes the Cartesian product of a JSON matrix file or string.

.. code-block:: cmake

  compute_matrix_product(<output_var>
                         (MATRIX_JSON_FILE <path> | MATRIX_JSON_STRING <json>))

The generated matrix product is returned as a JSON array string in ``<output_var>``.

``MATRIX_JSON_FILE``
  Path to a JSON matrix file.

``MATRIX_JSON_STRING``
  JSON matrix content.

Result Variables
^^^^^^^^^^^^^^^^

``<output_var>``
  JSON array containing the Cartesian product of the input matrix.

#]=======================================================================]
function(compute_matrix_product output_var)
  set(options)
  set(one_value MATRIX_JSON_FILE MATRIX_JSON_STRING)
  set(multi_value)

  cmake_parse_arguments(_CMP "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  if(_CMP_MATRIX_JSON_FILE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compute_matrix_product.py"
              "${_CMP_MATRIX_JSON_FILE}" #
      OUTPUT_VARIABLE output COMMAND_ERROR_IS_FATAL ANY
    )
  else()
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E echo "${_CMP_MATRIX_JSON_STRING}"
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

#[=======================================================================[.rst:
populate_matrix_variables
-------------------------

Populates CMake variables from a JSON matrix entry object.

.. code-block:: cmake

  populate_matrix_variables(<matrix_json_entry>)

Each member in ``<matrix_json_entry>`` is written as a variable in the caller's scope.

#]=======================================================================]
function(populate_matrix_variables matrix_json_entry)
  string(JSON len LENGTH "${matrix_json_entry}")
  math(EXPR last "${len} - 1")

  foreach(i RANGE "${last}")
    string(JSON key MEMBER "${matrix_json_entry}" "${i}")
    string(JSON value GET "${matrix_json_entry}" "${key}")
    set(${key}
        "${value}"
        PARENT_SCOPE
    )
  endforeach()
endfunction()
