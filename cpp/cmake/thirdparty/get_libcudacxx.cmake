# =============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

# This function finds libcudacxx and sets any additional necessary environment variables.
function(find_and_configure_libcudacxx)
  include(${rapids-cmake-dir}/cpm/libcudacxx.cmake)

  rapids_cpm_libcudacxx(
    BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports PATCH_COMMAND patch
    --reject-file=- -p1 -N < ${CUDF_SOURCE_DIR}/cmake/libcudacxx.patch || true
  )

  set(LIBCUDACXX_INCLUDE_DIR
      "${libcudacxx_SOURCE_DIR}/include"
      PARENT_SCOPE
  )
endfunction()

find_and_configure_libcudacxx()
