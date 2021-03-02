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

function(find_and_configure_dlpack VERSION)
    if(DLPACK_INCLUDE)
        set(DLPACK_INCLUDE_DIR "${DLPACK_INCLUDE_DIR}" PARENT_SCOPE)
        return()
    endif()
    find_path(DLPACK_INCLUDE_DIR "dlpack"
        HINTS "$ENV{DLPACK_ROOT}/include"
              "$ENV{CONDA_PREFIX}/include")
    if(DLPACK_INCLUDE_DIR)
        set(DLPACK_INCLUDE_DIR ${DLPACK_INCLUDE_DIR} PARENT_SCOPE)
        return()
    endif()
    CPMFindPackage(NAME dlpack
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/dmlc/dlpack.git
        GIT_TAG         v${VERSION}
        GIT_SHALLOW     TRUE
        DOWNLOAD_ONLY   TRUE
        OPTIONS         "BUILD_MOCK OFF")
    set(DLPACK_INCLUDE_DIR "${dlpack_SOURCE_DIR}/include" PARENT_SCOPE)
endfunction()

set(CUDF_MIN_VERSION_dlpack 0.3)

find_and_configure_dlpack(${CUDF_MIN_VERSION_dlpack})
