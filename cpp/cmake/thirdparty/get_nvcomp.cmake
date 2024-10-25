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

# This function finds nvcomp and sets any additional necessary environment variables.
function(find_and_configure_nvcomp)

  include(${rapids-cmake-dir}/cpm/nvcomp.cmake)
  set(export_args)
  if(CUDF_EXPORT_NVCOMP)
    set(export_args BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)
  endif()
  rapids_cpm_nvcomp(${export_args} USE_PROPRIETARY_BINARY ${CUDF_USE_PROPRIETARY_NVCOMP})

  # Per-thread default stream
  if(TARGET nvcomp AND CUDF_USE_PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(nvcomp PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
  endif()
endfunction()

find_and_configure_nvcomp()
