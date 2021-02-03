# Copyright (c) 2020, NVIDIA CORPORATION.

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    # This assumes the script is executed from the root of the repo directory
    ./build.sh -v libcudf_kafka
else
    ./build.sh -v libcudf_kafka tests
fi
