#!/bin/bash

DOCKER_IMAGE="gpuci/rapidsai-base:cuda10.0-ubuntu16.04-gcc5-py3.6"
REPO_PATH=${PWD}
RAPIDS_DIR_IN_CONTAINER="/rapids"
BUILD_DIR="cpp/build"
CONTAINER_SHELL_ONLY=0

SHORTHELP="$(basename $0) [-h] [-H] [-s] [-r <repo_dir>] [-i <image_name>]"
LONGHELP="${SHORTHELP}
Build and test your local repository using a base gpuCI Docker image

where:
    -H   Show this help text
    -r   Path to repository (defaults to working directory)
    -i   Use Docker image (default is ${DOCKER_IMAGE})
    -s   Start an interactive shell in a container of the Docker image
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
BUILD_DIR_IN_CONTAINER="${RAPIDS_DIR_IN_CONTAINER}/$(basename ${REPO_PATH})/${BUILD_DIR}"

# CONTAINER_BUILD_DIR is named after the image name, allowing for
# multiple image builds to coexist on the local filesystem. This will
# be mapped to the typical BUILD_DIR inside of the container. Builds
# running in the container generate build artifacts just as they would
# in a bare-metal environment, and the host filesystem is able to
# maintain the host build in BUILD_DIR as well.
CONTAINER_BUILD_DIR=${REPO_PATH}/build_$(echo $(basename ${DOCKER_IMAGE})|sed -e 's/:/_/g')

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
    COMMAND="${BUILD_DIR_IN_CONTAINER}/build.sh || bash"
else
    COMMAND="bash"
fi

# Create the build dir for the container to mount, generate the build script inside of it
mkdir -p ${CONTAINER_BUILD_DIR}
echo "${BUILD_SCRIPT}" > ${CONTAINER_BUILD_DIR}/build.sh
chmod ugo+x ${CONTAINER_BUILD_DIR}/build.sh

# Run the generated build script in a container
docker pull ${DOCKER_IMAGE}
docker run --runtime=nvidia --rm -it -e NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} \
       -v ${REPO_PATH}:${REPO_PATH_IN_CONTAINER} \
       -v ${CONTAINER_BUILD_DIR}:${BUILD_DIR_IN_CONTAINER} \
       ${DOCKER_IMAGE} bash -c "${COMMAND}"
