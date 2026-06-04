# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(NOT DEFINED CUDF_BUILD_DIR)
  message(FATAL_ERROR "CUDF_BUILD_DIR must be provided")
endif()

set(cudf_targets "${CUDF_BUILD_DIR}/cudf-targets.cmake")
set(cudf_dependencies "${CUDF_BUILD_DIR}/cudf-dependencies.cmake")

foreach(file IN ITEMS "${cudf_targets}" "${cudf_dependencies}")
  if(NOT EXISTS "${file}")
    message(FATAL_ERROR "Expected generated cuDF export file does not exist: ${file}")
  endif()
endforeach()

file(READ "${cudf_targets}" targets_contents)
string(REGEX MATCH "INTERFACE_LINK_LIBRARIES \"([^\"]*)\"" _match "${targets_contents}")
if(NOT _match)
  message(FATAL_ERROR "Could not find cudf::cudf INTERFACE_LINK_LIBRARIES in ${cudf_targets}")
endif()
set(cudf_interface_libraries "${CMAKE_MATCH_1}")

foreach(forbidden IN ITEMS "rmm::rmm" "rapids_logger::rapids_logger" "spdlog::" "fmt::")
  if(cudf_interface_libraries MATCHES "${forbidden}")
    message(
      FATAL_ERROR
        "cudf::cudf INTERFACE_LINK_LIBRARIES exposes private dependency ${forbidden}: ${cudf_interface_libraries}"
    )
  endif()
endforeach()

file(READ "${cudf_dependencies}" dependencies_contents)
foreach(forbidden IN ITEMS rmm rapids_logger spdlog fmt)
  if(dependencies_contents MATCHES "find_dependency\\(${forbidden}"
     OR dependencies_contents MATCHES "NAME;${forbidden}(;|\")"
  )
    message(FATAL_ERROR "${cudf_dependencies} exposes private dependency package ${forbidden}")
  endif()
endforeach()
