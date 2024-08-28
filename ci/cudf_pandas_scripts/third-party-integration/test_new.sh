#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs

set -euo pipefail

write_output() {
  local key="$1"
  local value="$2"
  echo "$key=$value" | tee --append "${GITHUB_OUTPUT:-/dev/null}"
}

extract_lib_from_dependencies_yaml() {
    local file=$1
    # Parse all keys in dependencies.yaml under the "files" section,
    # extract all the keys that start with "test_", and extract the rest
    local extracted_libs="$(yq -o json $file | jq -rc '.files | with_entries(select(.key | contains("test_"))) | keys | map(sub("^test_"; ""))')"
    echo $extracted_libs
}

main() {
    local dependencies_yaml="$1"

    LIBS=$(extract_lib_from_dependencies_yaml "$dependencies_yaml")
    LIBS=${LIBS#[}
    LIBS=${LIBS%]}

    for lib in ${LIBS//,/ }; do
        lib=$(echo "$lib" | tr -d '""')
        echo "Running tests for library $lib"

        CUDA_MAJOR=$(if [ "$lib" = "tensorflow" ]; then echo "11"; else echo "12"; fi)

        . /opt/conda/etc/profile.d/conda.sh

        rapids-logger "Generate Python testing dependencies"
        rapids-dependency-file-generator \
          --config "$dependencies_yaml" \
          --output conda \
          --file-key test_${lib} \
          --matrix "cuda=${CUDA_MAJOR};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

        rapids-mamba-retry env create --yes -f env.yaml -n test

        # Temporarily allow unbound variables for conda activation.
        set +u
        conda activate test
        set -u

        RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}

        mkdir -p "${RAPIDS_TESTS_DIR}"

        repo_root=$(git rev-parse --show-toplevel)
        TEST_DIR=${repo_root}/python/cudf/cudf_pandas_tests/third_party_integration_tests/tests

        rapids-print-env

        rapids-logger "Check GPU usage"
        nvidia-smi

        EXITCODE=0
        trap "EXITCODE=1" ERR
        set +e

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

        RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR} TEST_DIR=${TEST_DIR} NUM_PROCESSES=${NUM_PROCESSES} ci/cudf_pandas_scripts/third-party-integration/ci_run_library_tests.sh ${lib}

        rapids-logger "Test script exiting with value: ${EXITCODE}"
    done

    exit ${EXITCODE}
}

main "$@"
