#!/bin/bash

#
# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
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
#

set -ex
gcc --version

SKIP_JAVA_TESTS=${SKIP_JAVA_TESTS:-true}
BUILD_CPP_TESTS=${BUILD_CPP_TESTS:-OFF}
ENABLE_CUDA_STATIC_RUNTIME=${ENABLE_CUDA_STATIC_RUNTIME:-ON}
ENABLE_PTDS=${ENABLE_PTDS:-ON}
RMM_LOGGING_LEVEL=${RMM_LOGGING_LEVEL:-OFF}
ENABLE_NVTX=${ENABLE_NVTX:-ON}
ENABLE_GDS=${ENABLE_GDS:-OFF}
OUT=${OUT:-out}
CMAKE_GENERATOR=${CMAKE_GENERATOR:-Ninja}

SIGN_FILE=$1
#Set absolute path for OUT_PATH
OUT_PATH="$WORKSPACE/$OUT"

# set on Jenkins parameter
echo "SIGN_FILE: $SIGN_FILE,\
 SKIP_JAVA_TESTS: $SKIP_JAVA_TESTS,\
 BUILD_CPP_TESTS: $BUILD_CPP_TESTS,\
 ENABLE_CUDA_STATIC_RUNTIME: $ENABLE_CUDA_STATIC_RUNTIME,\
 ENABLED_PTDS: $ENABLE_PTDS,\
 ENABLE_NVTX: $ENABLE_NVTX,\
 ENABLE_GDS: $ENABLE_GDS,\
 RMM_LOGGING_LEVEL: $RMM_LOGGING_LEVEL,\
 OUT_PATH: $OUT_PATH"

INSTALL_PREFIX=/usr/local/rapids
export GIT_COMMITTER_NAME="ci"
export GIT_COMMITTER_EMAIL="ci@nvidia.com"
export CUDACXX=/usr/local/cuda/bin/nvcc
export LIBCUDF_KERNEL_CACHE_PATH=/rapids

###### Build libcudf ######
rm -rf "$WORKSPACE/cpp/build"
mkdir -p "$WORKSPACE/cpp/build"
cd "$WORKSPACE/cpp/build"
cmake .. -G"${CMAKE_GENERATOR}" \
         -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
         -DCUDA_STATIC_RUNTIME=$ENABLE_CUDA_STATIC_RUNTIME \
         -DUSE_NVTX=$ENABLE_NVTX \
         -DCUDF_LARGE_STRINGS_DISABLED=ON \
         -DCUDF_USE_ARROW_STATIC=ON \
         -DCUDF_ENABLE_ARROW_S3=OFF \
         -DBUILD_TESTS=$BUILD_CPP_TESTS \
         -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=$ENABLE_PTDS \
         -DRMM_LOGGING_LEVEL=$RMM_LOGGING_LEVEL \
         -DBUILD_SHARED_LIBS=OFF \
         -DCUDF_KVIKIO_REMOTE_IO=OFF

if [[ -z "${PARALLEL_LEVEL}" ]]; then
    cmake --build .
else
    cmake --build . --parallel $PARALLEL_LEVEL
fi
cmake --install .

###### Build cudf jar ######
BUILD_ARG="-Dmaven.repo.local=\"$WORKSPACE/.m2\"\
 -DskipTests=$SKIP_JAVA_TESTS\
 -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=$ENABLE_PTDS\
 -DCUDA_STATIC_RUNTIME=$ENABLE_CUDA_STATIC_RUNTIME\
 -DCUDF_JNI_LIBCUDF_STATIC=ON\
 -DUSE_GDS=$ENABLE_GDS -Dtest=*,!CuFileTest,!CudaFatalTest,!ColumnViewNonEmptyNullsTest"

if [ "$SIGN_FILE" == true ]; then
    # Build javadoc and sources only when SIGN_FILE is true
    BUILD_ARG="$BUILD_ARG -Prelease"
fi

if [ -f "$WORKSPACE/java/ci/settings.xml" ]; then
    # Build with an internal settings.xml
    BUILD_ARG="$BUILD_ARG -s \"$WORKSPACE/java/ci/settings.xml\""
fi

cd "$WORKSPACE/java"
mvn -B clean package $BUILD_ARG

###### Stash Jar files ######
rm -rf $OUT_PATH
mkdir -p $OUT_PATH
cp -f target/*.jar $OUT_PATH
