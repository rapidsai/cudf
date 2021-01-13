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

function(find_and_configure_thrust VERSION)
    CPMAddPackage(NAME Thrust
        VERSION         ${VERSION}
        GIT_REPOSITORY  https://github.com/NVIDIA/thrust.git
        GIT_TAG         ${VERSION}
        GIT_SHALLOW     TRUE
        PATCH_COMMAND   patch -p1 -N < ${CUDF_SOURCE_DIR}/cmake/thrust.patch || true)

    thrust_create_target(cudf::Thrust FROM_OPTIONS)
    set(THRUST_LIBRARY "cudf::Thrust" PARENT_SCOPE)
endfunction()

set(CUDF_MIN_VERSION_Thrust 1.10.0)

find_and_configure_thrust(${CUDF_MIN_VERSION_Thrust})
