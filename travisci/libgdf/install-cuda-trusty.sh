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

# Get CUDA release
if [ ${CUDA:0:1} == '9' ]
then
    # CUDA 9 release
    CUDA_REL=${CUDA:0:3}
elif [ ${CUDA:0:2} == '10' ]
then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
else
    # Didn't match one of the expected CUDA builds, exit
    echo "CUDA version not specified or invalid version specified!"
    exit 1
fi

# Default values
CUDA_PROD="Prod"
CUDA_FILE="cuda_${CUDA}_linux"

# Handle CUDA releases with non-default vals
if [ $CUDA_REL == '9.0' ]
then
    # CUDA 9 has different file name
    CUDA_FILE="cuda_${CUDA}_linux-run"
elif [ ${CUDA:0:3} == '9.2' ]
then
    # Cuda 9.2 has a different url pattern
    CUDA_PROD="Prod2"
fi

# Download and install CUDA
travis_retry wget --progress=dot:giga https://developer.nvidia.com/compute/cuda/${CUDA_REL}/${CUDA_PROD}/local_installers/${CUDA_FILE}
chmod +x cuda_*_linux*
sudo ./cuda_*_linux* --silent --toolkit

export CUDA_HOME=/usr/local/cuda-${CUDA_REL}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
