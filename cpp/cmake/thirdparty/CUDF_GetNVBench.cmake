#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

# NVBench doesn't have a public release yet

function(find_and_configure_nvbench)

    if(TARGET nvbench::main)
        return()
    endif()

    CPMFindPackage(NAME nvbench
        GIT_REPOSITORY  https://github.com/NVIDIA/nvbench.git
        GIT_TAG         main
        GIT_SHALLOW     TRUE
        OPTIONS         "NVBench_ENABLE_EXAMPLES OFF"
                        "NVBench_ENABLE_TESTING OFF")

endfunction()

find_and_configure_nvbench()
