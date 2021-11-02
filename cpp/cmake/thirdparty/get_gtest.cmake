# =============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

  # Find or install GoogleTest
  rapids_cpm_gtest(BUILD_EXPORT_SET cudf-testing-exports INSTALL_EXPORT_SET cudf-testing-exports)

  if(GTest_ADDED)
    rapids_export(
      BUILD GTest
      VERSION ${GTest_VERSION}
      EXPORT_SET GTestTargets
      GLOBAL_TARGETS gtest gmock gtest_main gmock_main
      NAMESPACE GTest::
    )

    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
      BUILD GTest [=[${CMAKE_CURRENT_LIST_DIR}]=] cudf-testing-exports
    )
  endif()

endfunction()

find_and_configure_gtest()
