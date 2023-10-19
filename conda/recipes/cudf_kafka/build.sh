# Copyright (c) 2020-2023, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory
# Need to set CUDA_HOME inside conda environments because the hacked together
# setup.py for cudf-kafka searches that way.
# TODO: Remove after https://github.com/rapidsai/cudf/pull/14292 updates
# cudf_kafka to use scikit-build
CUDA_MAJOR=${RAPIDS_CUDA_VERSION%%.*}
if [[ ${CUDA_MAJOR} == "12" ]]; then
    target_name="x86_64-linux"
    if [[ ! $(arch) == "x86_64" ]]; then
        target_name="sbsa-linux"
    fi
    export CUDA_HOME="${PREFIX}/targets/${target_name}/"
fi
./build.sh -v cudf_kafka
