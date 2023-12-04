# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
include_guard(GLOBAL)

# Making libraries available inside wheels by installing the associated targets.
function(install_aliased_imported_targets)
  list(APPEND CMAKE_MESSAGE_CONTEXT "install_aliased_imported_targets")

  set(options "")
  set(one_value "DESTINATION")
  set(multi_value "TARGETS")
  cmake_parse_arguments(_ "${options}" "${one_value}" "${multi_value}" ${ARGN})

  message(VERBOSE "Installing targets '${__TARGETS}' into lib_dir '${__DESTINATION}'")

  foreach(target IN LISTS __TARGETS)

    if(NOT TARGET ${target})
      message(VERBOSE "No target named ${target}")
      continue()
    endif()

    get_target_property(alias_target ${target} ALIASED_TARGET)
    if(alias_target)
      set(target ${alias_target})
    endif()

    get_target_property(is_imported ${target} IMPORTED)
    if(NOT is_imported)
      # If the target isn't imported, install it into the wheel
      install(TARGETS ${target} DESTINATION ${__DESTINATION})
      message(VERBOSE "install(TARGETS ${target} DESTINATION ${__DESTINATION})")
    else()
      # If the target is imported, make sure it's global
      get_target_property(type ${target} TYPE)
      if(${type} STREQUAL "UNKNOWN_LIBRARY")
        install(FILES $<TARGET_FILE:${target}> DESTINATION ${__DESTINATION})
        message(VERBOSE "install(FILES $<TARGET_FILE:${target}> DESTINATION ${__DESTINATION})")
      else()
        install(IMPORTED_RUNTIME_ARTIFACTS ${target} DESTINATION ${__DESTINATION})
        message(
          VERBOSE
          "install(IMPORTED_RUNTIME_ARTIFACTS $<TARGET_FILE:${target}> DESTINATION ${__DESTINATION})"
        )
      endif()
    endif()
  endforeach()
endfunction()
