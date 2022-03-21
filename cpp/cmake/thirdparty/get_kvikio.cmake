# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

# This function finds KvikIO and sets `KvikIO_INCLUDE_DIR`
function(find_and_configure_kvikio)

  rapids_cpm_find(
    KvikIO 22.04
    GLOBAL_TARGETS kvikio::kvikio
    # CPM_ARGS GIT_REPOSITORY https://github.com/rapidsai/kvikio.git
    CPM_ARGS
    GIT_REPOSITORY https://github.com/madsbk/kvikio.git SOURCE_SUBDIR cpp
    GIT_TAG file_size # TODO: use version tags when they become available
    OPTIONS "KvikIO_BUILD_EXAMPLES FALSE" # No need to build the KvikIO example
  )
  set(KvikIO_INCLUDE_DIR
      ${KvikIO_SOURCE_DIR}/cpp/include
      PARENT_SCOPE
  )

endfunction()

find_and_configure_kvikio()
