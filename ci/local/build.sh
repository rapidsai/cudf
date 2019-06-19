#!/bin/bash

DOCKER_IMAGE="gpuci/rapidsai-base:cuda10.0-ubuntu16.04-gcc5-py3.6"
REPO_PATH=${PWD}
RAPIDS_DIR_IN_CONTAINER="/rapids"
CPP_BUILD_DIR="cpp/build"
PYTHON_BUILD_DIR="python/build"
CONTAINER_SHELL_ONLY=0

SHORTHELP="$(basename $0) [-h] [-H] [-s] [-r <repo_dir>] [-i <image_name>]"
LONGHELP="${SHORTHELP}
Build and test your local repository using a base gpuCI Docker image

where:
    -H   Show this help text
    -r   Path to repository (defaults to working directory)
    -i   Use Docker image (default is ${DOCKER_IMAGE})
    -s   Skip building and testing and start an interactive shell in a container of the Docker image
"

# Limit GPUs available to container based on CUDA_VISIBLE_DEVICES
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    NVIDIA_VISIBLE_DEVICES="all"
else
    NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
fi

while getopts ":hHr:i:s" option; do
    case ${option} in
        r)
            REPO_PATH=${OPTARG}
            ;;
        i)
            DOCKER_IMAGE=${OPTARG}
            ;;
        s)
            CONTAINER_SHELL_ONLY=1
            ;;
        h)
            echo "${SHORTHELP}"
            exit 0
            ;;
        H)
            echo "${LONGHELP}"
            exit 0
            ;;
        *)
            echo "ERROR: Invalid flag"
            echo "${SHORTHELP}"
            exit 1
            ;;
    esac
done

REPO_PATH_IN_CONTAINER="${RAPIDS_DIR_IN_CONTAINER}/$(basename ${REPO_PATH})"
CPP_BUILD_DIR_IN_CONTAINER="${RAPIDS_DIR_IN_CONTAINER}/$(basename ${REPO_PATH})/${CPP_BUILD_DIR}"
PYTHON_BUILD_DIR_IN_CONTAINER="${RAPIDS_DIR_IN_CONTAINER}/$(basename ${REPO_PATH})/${PYTHON_BUILD_DIR}"


# BASE_CONTAINER_BUILD_DIR is named after the image name, allowing for
# multiple image builds to coexist on the local filesystem. This will
# be mapped to the typical BUILD_DIR inside of the container. Builds
# running in the container generate build artifacts just as they would
# in a bare-metal environment, and the host filesystem is able to
# maintain the host build in BUILD_DIR as well.
BASE_CONTAINER_BUILD_DIR=${REPO_PATH}/build_$(echo $(basename ${DOCKER_IMAGE})|sed -e 's/:/_/g')
CPP_CONTAINER_BUILD_DIR=${BASE_CONTAINER_BUILD_DIR}/cpp
PYTHON_CONTAINER_BUILD_DIR=${BASE_CONTAINER_BUILD_DIR}/python


BUILD_SCRIPT="#!/bin/bash
set -e
WORKSPACE=${REPO_PATH_IN_CONTAINER}
PREBUILD_SCRIPT=${REPO_PATH_IN_CONTAINER}/ci/gpu/prebuild.sh
BUILD_SCRIPT=${REPO_PATH_IN_CONTAINER}/ci/gpu/build.sh
cd ${WORKSPACE}
if [ -f \${PREBUILD_SCRIPT} ]; then
    source \${PREBUILD_SCRIPT}
fi
yes | source \${BUILD_SCRIPT}
"

if (( ${CONTAINER_SHELL_ONLY} == 0 )); then
    COMMAND="${CPP_BUILD_DIR_IN_CONTAINER}/build.sh || bash"
else
    COMMAND="bash"
fi

# Create the build dir for the container to mount, generate the build script inside of it
mkdir -p ${BASE_CONTAINER_BUILD_DIR}
mkdir -p ${CPP_CONTAINER_BUILD_DIR}
mkdir -p ${PYTHON_CONTAINER_BUILD_DIR}
echo "${BUILD_SCRIPT}" > ${CPP_CONTAINER_BUILD_DIR}/build.sh
chmod ugo+x ${CPP_CONTAINER_BUILD_DIR}/build.sh

# Run the generated build script in a container
docker pull ${DOCKER_IMAGE}
docker run --runtime=nvidia --rm -it -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
       --user $(id -u):$(id -g) \
       -v ${REPO_PATH}:${REPO_PATH_IN_CONTAINER} \
       -v ${CPP_CONTAINER_BUILD_DIR}:${CPP_BUILD_DIR_IN_CONTAINER} \
       -v ${PYTHON_CONTAINER_BUILD_DIR}:${PYTHON_BUILD_DIR_IN_CONTAINER} \
       --cap-add=SYS_PTRACE \
       ${DOCKER_IMAGE} bash -c "${COMMAND}"
