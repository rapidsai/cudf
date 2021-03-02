#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

# Jitify doesn't have a version :/

function(find_and_configure_jitify)
    CPMFindPackage(NAME     jitify
            VERSION         1.0.0
            GIT_REPOSITORY  https://github.com/rapidsai/jitify.git
            GIT_TAG         cudf_0.16
            GIT_SHALLOW     TRUE
            DOWNLOAD_ONLY   TRUE)
    set(JITIFY_INCLUDE_DIR "${jitify_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

find_and_configure_jitify()
