# Copyright (c) 2018-2019, NVIDIA CORPORATION.

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    # This assumes the script is executed from the root of the repo directory
    ./build.sh -v libcudf --allgpuarch
else
    BUILD_GPU_ARCH=80 ./build.sh -v libcudf tests #TODO Add back --allgpuarch
fi