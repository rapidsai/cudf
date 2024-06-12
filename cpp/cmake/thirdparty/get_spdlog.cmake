# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

# Use CPM to find or clone speedlog
function(find_and_configure_spdlog)

  include(${rapids-cmake-dir}/cpm/spdlog.cmake)
  rapids_cpm_spdlog(FMT_OPTION "EXTERNAL_FMT_HO" INSTALL_EXPORT_SET cudf-exports)
  rapids_export_package(BUILD spdlog cudf-exports)

  if(spdlog_ADDED)
    rapids_export(
      BUILD spdlog
      EXPORT_SET spdlog
      GLOBAL_TARGETS spdlog spdlog_header_only
      NAMESPACE spdlog::
    )
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
      BUILD spdlog [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cudf-exports
    )
  endif()
endfunction()

find_and_configure_spdlog()
