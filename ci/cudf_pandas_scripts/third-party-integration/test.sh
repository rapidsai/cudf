#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Common setup steps shared by Python test jobs

set -euo pipefail

# shellcheck source=ci/cudf_pandas_scripts/third-party-integration/timing.sh
source "$(dirname "${BASH_SOURCE[0]}")/timing.sh"

extract_lib_from_dependencies_yaml() {
    local file=$1
    # Parse all keys in dependencies.yaml under the "files" section,
    # extract all the keys that start with "test_", and extract the rest
    extracted_libs="$(yq -o json "$file" | jq -rc '.files | with_entries(select(.key | contains("test_"))) | keys | map(sub("^test_"; ""))')"
    echo "$extracted_libs"
}

main() {
    local dependencies_yaml="$1"

    LIBS=$(extract_lib_from_dependencies_yaml "$dependencies_yaml")
    LIBS=${LIBS#[}
    LIBS=${LIBS%]}

    integration_run_timed all package-manager-version mamba --version
    integration_run_timed all available-memory free -m
    integration_run_timed all available-disk df -h . /tmp

    if [ "$RAPIDS_BUILD_TYPE" == "pull-request" ]; then
        rapids-logger "Downloading artifacts from this pr jobs"
        CPP_CHANNEL=$(integration_run_timed all download-cpp rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_CHANNEL=$(integration_run_timed all download-python rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_NOARCH_CHANNEL=$(integration_run_timed all download-noarch rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --pure --cuda "$RAPIDS_CUDA_VERSION" --arch any)")
    fi

    ANY_FAILURES=0

    for lib in ${LIBS//,/ }; do
        lib=$(echo "$lib" | tr -d '""')
        echo "Running tests for library $lib"

        . /opt/conda/etc/profile.d/conda.sh
        # Check the value of RAPIDS_BUILD_TYPE
        if [ "$RAPIDS_BUILD_TYPE" == "pull-request" ]; then
            rapids-logger "Generate Python testing dependencies"
            integration_run_timed "$lib" dependencies rapids-dependency-file-generator \
                --config "$dependencies_yaml" \
                --output conda \
                --file-key "test_${lib}" \
                --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
                --prepend-channel "${CPP_CHANNEL}" \
                --prepend-channel "${PYTHON_CHANNEL}" \
                --prepend-channel "${PYTHON_NOARCH_CHANNEL}" | tee env.yaml
        else
            rapids-logger "Generate Python testing dependencies"
            integration_run_timed "$lib" dependencies rapids-dependency-file-generator \
                --config "$dependencies_yaml" \
                --output conda \
                --file-key "test_${lib}" \
                --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml
        fi

        integration_run_timed "$lib" environment rapids-mamba-retry env create --yes -f env.yaml -n test

        # Temporarily allow unbound variables for conda activation.
        set +u
        integration_run_timed "$lib" activation conda activate test
        set -u

        repo_root=$(git rev-parse --show-toplevel)
        TEST_DIR=${repo_root}/python/cudf/cudf_pandas_tests/third_party_integration_tests/tests

        integration_run_timed "$lib" environment-diagnostics rapids-print-env
        integration_run_timed "$lib" package-manifest conda list --explicit

        rapids-logger "Check GPU usage"
        integration_run_timed "$lib" gpu-diagnostics nvidia-smi

        rapids-logger "pytest ${lib}"

        NUM_PROCESSES=8
        serial_libraries=(
            "tensorflow"
        )
        for serial_library in "${serial_libraries[@]}"; do
            if [ "${lib}" = "${serial_library}" ]; then
                NUM_PROCESSES=1
            fi
        done

        EXITCODE=0
        trap "EXITCODE=1" ERR
        set +e

        TEST_DIR=${TEST_DIR} \
        NUM_PROCESSES=${NUM_PROCESSES} \
            integration_run_timed "$lib" tests timeout 45m \
            ci/cudf_pandas_scripts/third-party-integration/run-library-tests.sh "${lib}"

        set -e
        rapids-logger "Test script exiting with value: ${EXITCODE}"
        if [[ ${EXITCODE} != 0 ]]; then
            ANY_FAILURES=1
        fi
    done

    exit ${ANY_FAILURES}
}

main "$@"
