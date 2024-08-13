# =============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

# This function finds gtest and sets any additional necessary environment variables.
function(find_and_configure_gtest)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)

  # Mark all the non explicit googletest symbols as hidden. This ensures that libcudftestutil can be
  # used by consumers with a different shared gtest.
  set(gtest_hide_internal_symbols ON)

  # Find or install GoogleTest
  rapids_cpm_gtest(BUILD_STATIC)

  # Mark all the explicit googletest symbols as hidden. This ensures that libcudftestutil can be used
  # by consumers with a different shared gtest.
  if(TARGET gtest)
    target_compile_definitions(gtest PUBLIC "$<BUILD_LOCAL_INTERFACE:GTEST_API_=>")
  endif()
endfunction()

find_and_configure_gtest()
