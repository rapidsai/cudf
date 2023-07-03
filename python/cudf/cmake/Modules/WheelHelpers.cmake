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
function(add_target_libs_to_wheel)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_target_libs_to_wheel")

  set(options "")
  set(one_value "LIB_DIR")
  set(multi_value "TARGETS")
  cmake_parse_arguments(_ "${options}" "${one_value}" "${multi_value}" ${ARGN})

  message(VERBOSE "Installing targets '${__TARGETS}' into lib_dir '${__LIB_DIR}'")

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
      install(TARGETS ${target} DESTINATION ${__LIB_DIR})
      message(VERBOSE "install(TARGETS ${target} DESTINATION ${__LIB_DIR})")
    else()
      # If the target is imported, make sure it's global
      get_target_property(already_global ${target} IMPORTED_GLOBAL)
      if(NOT already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()

      # Find the imported target's library so we can copy it into the wheel
      set(lib_loc)
      foreach(prop IN ITEMS IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_DEBUG)
        get_target_property(lib_loc ${target} ${prop})
        if(lib_loc)
          message(VERBOSE "Found ${prop} for ${target}: ${lib_loc}")
          break()
        endif()
        message(VERBOSE "${target} has no value for property ${prop}")
      endforeach()

      if(NOT lib_loc)
        message(FATAL_ERROR "Found no libs to install for target ${target}")
      endif()

      # Copy the imported library into the wheel
      install(FILES ${lib_loc} DESTINATION ${__LIB_DIR})
      message(VERBOSE "install(FILES ${lib_loc} DESTINATION ${__LIB_DIR})")
    endif()
  endforeach()
endfunction()
