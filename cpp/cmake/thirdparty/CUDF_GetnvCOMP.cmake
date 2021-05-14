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

function(find_and_configure_nvcomp VERSION)

    if(TARGET nvCOMP::nvcomp)
        return()
    endif()

    # Find or install nvCOMP
    CPMFindPackage(NAME nvCOMP
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/NVIDIA/nvcomp.git
        GIT_TAG         v${VERSION}
        GIT_SHALLOW     TRUE)

    if(NOT TARGET nvCOMP::nvcomp)
        add_library(nvCOMP::nvcomp ALIAS nvcomp)
    endif()

    # Make sure consumers of cudf can also see nvCOMP::nvcomp target
    fix_cmake_global_defaults(nvCOMP::nvcomp)
endfunction()

set(CUDF_MIN_VERSION_nvCOMP 2.0.0)

find_and_configure_nvcomp(${CUDF_MIN_VERSION_nvCOMP})
