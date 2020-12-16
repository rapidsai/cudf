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

set(CUDF_MIN_VERSION_dlpack 0.3)

if (DLPACK_INCLUDE)
    set(dlpack_FOUND TRUE)
else()
    CPMFindPackage(NAME     dlpack
        VERSION             ${CUDF_MIN_VERSION_dlpack}
        GIT_REPOSITORY      https://github.com/dmlc/dlpack.git
        GIT_TAG             v${CUDF_MIN_VERSION_dlpack}
        GIT_SHALLOW         TRUE
        FIND_PACKAGE_ARGS   "CONFIG"
                            "HINTS $ENV{DLPACK_ROOT} ${CONDA_PREFIX}"
        OPTIONS             "BUILD_MOCK OFF"
    )

    message(STATUS "dlpack_ADDED: ${dlpack_ADDED}")
    message(STATUS "dlpack_FOUND: ${dlpack_FOUND}")
    
    if(NOT (dlpack_FOUND OR dlpack_ADDED))
        message(FATAL_ERROR "dlpack package not found")
    endif()
    
    message(STATUS "dlpack_SOURCE_DIR: ${dlpack_SOURCE_DIR}")
    message(STATUS "dlpack_BINARY_DIR: ${dlpack_BINARY_DIR}")
    
    set(DLPACK_INCLUDE "${dlpack_SOURCE_DIR}/include")
endif()

message(STATUS "DLPACK_INCLUDE: ${DLPACK_INCLUDE}")
