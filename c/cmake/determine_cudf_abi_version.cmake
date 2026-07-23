# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
determine_cudf_abi_version
---------------------------

.. versionadded:: v26.02.00

Convert a calendar version to a cuDF ABI version.

  .. code-block:: cmake

    determine_cudf_abi_version(cal_ver MAJOR major_output_var MINOR minor_output_var)

Provides a consistent method to convert calendar-based version strings (YY.MM format) to
cuDF ABI version components.

``cal_ver``
    A calendar version string in YY.MM format (e.g., "26.02", "27.08").

``major_output_var``
    Contains the name of the variable that will be set in the parent scope to the computed
    ABI major version.

``minor_output_var``
    Contains the name of the variable that will be set in the parent scope to the computed
    ABI minor version.

#]=======================================================================]
# cmake-lint: disable=C0112
function(determine_cudf_abi_version cal_ver)
  set(options)
  set(one_value "MAJOR" "MINOR")
  set(multi_value)
  cmake_parse_arguments(_CUDF_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  rapids_cmake_parse_version(MAJOR ${cal_ver} cal_ver_major)
  rapids_cmake_parse_version(MINOR ${cal_ver} cal_ver_minor)

  set(current_major_abi_ver "1")
  set(abi_base_year "26")
  set(abi_base_month "02")

  if(cal_ver_major STREQUAL abi_base_year)
    math(EXPR computed_abi_minor "(${cal_ver_minor}-${abi_base_month})/2")
  else()
    math(EXPR first_year_count "(12-${abi_base_month})/2")
    math(EXPR extra_years "(${cal_ver_major} - ${abi_base_year} - 1) * 6")
    math(EXPR this_year_count "(${cal_ver_minor})/2")
    math(EXPR computed_abi_minor "${first_year_count} + ${extra_years} + ${this_year_count}")
  endif()

  set(${_CUDF_RAPIDS_MAJOR}
      ${current_major_abi_ver}
      PARENT_SCOPE
  )
  set(${_CUDF_RAPIDS_MINOR}
      ${computed_abi_minor}
      PARENT_SCOPE
  )
endfunction()
