#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/install-cuda-trusty.sh
#
# Install the core CUDA toolkit for a ubuntu-trusty (14.04) system. Requires the
# CUDA environment variable to be set to the required version.
#
# Since this script updates environment variables, to execute correctly you must
# 'source' this script, rather than executing it in a sub-process.
#
set -e

# CUDA 8 we can use the repo
if [ ${CUDA:0:1} == '8' ]
then
    travis_retry wget --progress=dot:mega http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_${CUDA}_amd64.deb
    travis_retry sudo dpkg -i cuda-repo-ubuntu1404_${CUDA}_amd64.deb
    travis_retry sudo apt-get update -qq
    export CUDA_APT=${CUDA:0:3}
    export CUDA_APT=${CUDA_APT/./-}
    # travis_retry sudo apt-get install -y cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} cuda-cufft-dev-${CUDA_APT}
    travis_retry sudo apt-get install -y cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT}
    travis_retry sudo apt-get clean
elif [ ${CUDA:0:3} == '9.0' ]
then
    # CUDA 9 we use the sh installer
    travis_retry wget --progress=dot:mega https://developer.nvidia.com/compute/cuda/${CUDA:0:3}/Prod/local_installers/cuda_${CUDA}_linux-run
    chmod +x cuda_*_linux-run
    sudo ./cuda_*_linux-run --silent --toolkit
elif [ ${CUDA:0:3} == '9.1' ]
then
    # CUDA 9.1 has a different filename pattern
    travis_retry wget --progress=dot:mega https://developer.nvidia.com/compute/cuda/${CUDA:0:3}/Prod/local_installers/cuda_${CUDA}_linux
    chmod +x cuda_*_linux
    sudo ./cuda_*_linux --silent --toolkit
elif [ ${CUDA:0:3} == '9.2' ]
then
    # Cuda 9.2 has a different url pattern
    travis_retry wget --progress=dot:mega https://developer.nvidia.com/compute/cuda/${CUDA:0:3}/Prod2/local_installers/cuda_${CUDA}_linux
    chmod +x cuda_*_linux
    sudo ./cuda_*_linux --silent --toolkit
else
    # Didn't match one of the expected CUDA builds, exit
    echo "CUDA version not specified or invalid version specified!"
    exit 1
fi
export CUDA_HOME=/usr/local/cuda-${CUDA:0:3}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
