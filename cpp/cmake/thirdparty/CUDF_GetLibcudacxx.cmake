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

function(find_and_configure_libcudacxx VERSION)
    CPMFindPackage(NAME     libcudacxx
        VERSION             ${VERSION}
        GIT_REPOSITORY      https://github.com/NVIDIA/libcudacxx.git
        GIT_TAG             ${VERSION}
        GIT_SHALLOW         TRUE
        DOWNLOAD_ONLY       TRUE
    )
    set(LIBCUDACXX_DIR "${libcudacxx_SOURCE_DIR}" PARENT_SCOPE)
    set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include" PARENT_SCOPE)
    set(LIBCXX_DIR "${libcudacxx_SOURCE_DIR}/libcxx" PARENT_SCOPE)
    set(LIBCXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/libcxx/include" PARENT_SCOPE)
endfunction()

set(CUDF_MIN_VERSION_libcudacxx 1.4.0)

find_and_configure_libcudacxx(${CUDF_MIN_VERSION_libcudacxx})
