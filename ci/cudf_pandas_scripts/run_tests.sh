#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

# Function to display script usage
function display_usage {
    echo "Usage: $0 [--no-cudf]"
}

# Default value for the --no-cudf option
no_cudf=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cudf)
            no_cudf=true
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            display_usage
            exit 1
            ;;
    esac
done

if [ "$no_cudf" = true ]; then
    echo "Skipping cudf install"
else
    # Set the manylinux version used for downloading the wheels so that we test the
    # newer ABI wheels on the newer images that support their installation.
    # Need to disable pipefail for the head not to fail, see
    # https://stackoverflow.com/questions/19120263/why-exit-code-141-with-grep-q
    set +o pipefail
    glibc_minor_version=$(ldd --version | head -1 | grep -o "[0-9]\.[0-9]\+" | tail -1 | cut -d '.' -f2)
    set -o pipefail
    manylinux_version="2_17"
    if [[ ${glibc_minor_version} -ge 28 ]]; then
        manylinux_version="2_28"
    fi
    manylinux="manylinux_${manylinux_version}"

    RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
    RAPIDS_PY_WHEEL_NAME="cudf_${manylinux}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-cudf-dep
    python -m pip install $(ls ./local-cudf-dep/cudf*.whl)[test,cudf-pandas-tests]
fi

python -m pytest -p cudf.pandas ./python/cudf/cudf_pandas_tests/
