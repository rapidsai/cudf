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

# Use CPM to find or clone flatbuffers
function(find_and_configure_flatbuffers VERSION)

  rapids_cpm_find(
    flatbuffers ${VERSION}
    GLOBAL_TARGETS flatbuffers
    CPM_ARGS
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v${VERSION}
    GIT_SHALLOW TRUE
  )

  rapids_export_find_package_root(
    BUILD flatbuffers "${flatbuffers_BINARY_DIR}" EXPORT_SET cudf-exports
  )

endfunction()

find_and_configure_flatbuffers(24.3.25)
