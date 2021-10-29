#!/bin/bash
set -e
WORKSPACE=/rapids/local
PREBUILD_SCRIPT=/rapids/local/ci/gpu/prebuild.sh
BUILD_SCRIPT=/rapids/local/ci/gpu/build.sh
if [ -f ${PREBUILD_SCRIPT} ]; then
    source ${PREBUILD_SCRIPT}
fi
yes | source ${BUILD_SCRIPT}

