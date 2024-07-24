# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

# This function finds nanoarrow and sets any additional necessary environment variables.
function(find_and_configure_nanoarrow)
  # Currently we need to always build nanoarrow so we don't pickup a previous installed version
  set(CPM_DOWNLOAD_nanoarrow ON)
  rapids_cpm_find(
    nanoarrow 0.6.0.dev
    GLOBAL_TARGETS nanoarrow
    CPM_ARGS
    GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow.git
    GIT_TAG 1e2664a70ec14907409cadcceb14d79b9670bcdb
    GIT_SHALLOW FALSE
    OPTIONS "BUILD_SHARED_LIBS OFF" "NANOARROW_NAMESPACE cudf"
  )
  set_target_properties(nanoarrow PROPERTIES POSITION_INDEPENDENT_CODE ON)
  rapids_export_find_package_root(BUILD nanoarrow "${nanoarrow_BINARY_DIR}" EXPORT_SET cudf-exports)
endfunction()

find_and_configure_nanoarrow()
