#!/bin/bash

set -e
gcc --version

SIGN_FILE=$1
#Set absolute path for OUT_PATH
OUT_PATH=$WORKSPACE/$OUT

# set on Jenkins parameter
if [ -z $RMM_VERSION ]
then
RMM_VERSION=`git describe --tags | grep -o -E '([0-9]+\.[0-9]+)'`
fi
echo "RMM_VERSION: $RMM_VERSION, SIGN_FILE: $SIGN_FILE, SKIP_JAVA_TESTS: $SKIP_JAVA_TESTS,\
 BUILD_CPP_TESTS: $BUILD_CPP_TESTS, ENABLED_PTDS: $ENABLE_PTDS, OUT_PATH: $OUT_PATH"

INSTALL_PREFIX=/usr/local/rapids
export GIT_COMMITTER_NAME="ci"
export GIT_COMMITTER_EMAIL="ci@nvidia.com"
export CUDACXX=/usr/local/cuda/bin/nvcc
export RMM_ROOT=$INSTALL_PREFIX
export DLPACK_ROOT=$INSTALL_PREFIX
export LIBCUDF_KERNEL_CACHE_PATH=/rapids

cd /rapids/
git clone --recurse-submodules https://github.com/rapidsai/rmm.git -b branch-$RMM_VERSION
git clone --recurse-submodules https://github.com/rapidsai/dlpack.git -b cudf

###### Build rmm/dlpack ######
mkdir -p /rapids/rmm/build
cd /rapids/rmm/build
echo "RMM SHA: `git rev-parse HEAD`"
cmake .. -DCMAKE_CXX11_ABI=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DBUILD_TESTS=$BUILD_CPP_TESTS
make -j$PARALLEL_LEVEL install

# Install spdlog headers from RMM build
(cd /rapids/rmm/build/_deps/spdlog-src && find include/spdlog | cpio -pmdv $INSTALL_PREFIX)

mkdir -p /rapids/dlpack/build
cd /rapids/dlpack/build
echo "DLPACK SHA: `git rev-parse HEAD`"
cmake .. -DCMAKE_CXX11_ABI=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DBUILD_TESTS=$BUILD_CPP_TESTS
make -j$PARALLEL_LEVEL install

###### Build libcudf ######
rm -rf $WORKSPACE/cpp/build
mkdir -p $WORKSPACE/cpp/build
cd $WORKSPACE/cpp/build
cmake .. -DCMAKE_CXX11_ABI=OFF -DUSE_NVTX=OFF -DARROW_STATIC_LIB=ON -DBoost_USE_STATIC_LIBS=ON -DBUILD_TESTS=$SKIP_CPP_TESTS -DPER_THREAD_DEFAULT_STREAM=$ENABLE_PTDS
make -j$PARALLEL_LEVEL
make install DESTDIR=$INSTALL_PREFIX

###### Build cudf jar ######
BUILD_ARG="-Dmaven.repo.local=$WORKSPACE/.m2 -DskipTests=$SKIP_JAVA_TESTS -PabiOff -DPER_THREAD_DEFAULT_STREAM=$ENABLE_PTDS"
if [ "$SIGN_FILE" == true ]; then
    # Build javadoc and sources only when SIGN_FILE is true
    BUILD_ARG="$BUILD_ARG -Prelease"
fi

cd $WORKSPACE/java
mvn -B clean package $BUILD_ARG

###### Stash Jar files ######
rm -rf $OUT_PATH
mkdir -p $OUT_PATH
cp -f target/*.jar $OUT_PATH
