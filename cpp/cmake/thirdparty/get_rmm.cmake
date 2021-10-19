#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_rmm)
    include(${rapids-cmake-dir}/cpm/rmm.cmake)
    include(${rapids-cmake-dir}/cpm/package_override.cmake)

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rmm_override.json
[=[
{
  "packages" : {
    "rmm" : {
      "version" : "21.12",
      "git_url" : "https://github.com/robertmaynard/rmm.git",
      "git_tag" : "mark_optional_cuda_runtime_symbols_as_weak",
      "git_shallow" : true
    }
  }
}
]=]
        )
    rapids_cpm_package_override(${CMAKE_CURRENT_BINARY_DIR}/rmm_override.json)
    # Find or install RMM
    rapids_cpm_rmm(BUILD_EXPORT_SET cudf-exports
                   INSTALL_EXPORT_SET cudf-exports)

endfunction()

find_and_configure_rmm()
