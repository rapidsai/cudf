#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_cpp_common.sh

RUN_LIBCUDF_TESTS="${RUN_LIBCUDF_TESTS:-true}"
RUN_EXAMPLES_TESTS="${RUN_EXAMPLES_TESTS:-true}"
RUN_LIBCUDF_KAFKA_TESTS="${RUN_LIBCUDF_KAFKA_TESTS:-true}"
RUN_LIBCUDF_STREAMING_TESTS="${RUN_LIBCUDF_STREAMING_TESTS:-true}"

examples_timeout=30m
kafka_timeout=30m
if [[ ${RAPIDS_BUILD_TYPE:-} == pull-request ]]; then
    examples_timeout=5m
    kafka_timeout=2m
fi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run gtests from the libcudf-tests and libcudf-streaming-tests packages
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

SUITEERROR=0

if [[ "${RUN_LIBCUDF_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf gtests"
    timeout 30m ./ci/run_cudf_ctests.sh -j20
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_EXAMPLES_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf examples"
    timeout "${examples_timeout}" ./ci/run_cudf_examples.sh
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_LIBCUDF_KAFKA_TESTS}" == "true" ]]; then
    rapids-logger "Run libcudf_kafka gtests"
    timeout "${kafka_timeout}" ./ci/run_cudf_kafka_ctests.sh -j20
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )) && [[ "${RUN_LIBCUDF_STREAMING_TESTS}" == "true" ]]; then
    # ---- TEMPORARY UCX segfault debugging (not for merge) ----
    rapids-logger "UCX debug: driver / library information"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
    cat /proc/driver/nvidia/version || true
    uname -a
    ldconfig -p | grep -E "libcuda\.so" || true
    for lib in $(ldconfig -p | grep -E "libcuda\.so\.1" | awk '{print $NF}' | sort -u); do
        echo "${lib} -> $(readlink -f "${lib}")"
        readelf -n "${lib}" | grep -i "build id" || true
    done
    ucx_info -v || true
    ucx_info -d | grep -i -B2 -A6 "cuda_cpy" || true

    # Debug-level UCX logs from every rank go to files (they are huge); the tail is
    # printed below if the tests fail.
    export UCX_LOG_LEVEL=debug
    export UCX_LOG_FILE="/tmp/ucx_debug_%h_%p.log"

    rapids-logger "Run libcudf_streaming gtests"
    # cudf_streaming contains distributed tests, and running tests in
    # parallel results in resource starvation CI env.
    # Multiple MPI ranks execute the same GTest binary concurrently. GTest's
    # directory-mode XML output chooses file names without interprocess
    # synchronization, so ranks can corrupt a shared report. Keep XML reports
    # enabled for the non-distributed suites above, but disable them here.
    saved_gtest_output="${GTEST_OUTPUT}"
    unset GTEST_OUTPUT
    timeout 5m ./ci/run_cudf_streaming_ctests.sh -j1
    SUITEERROR=$?
    export GTEST_OUTPUT="${saved_gtest_output}"

    # The crash is intermittent, so always hammer the failing test under gdb,
    # whether or not the ctest run above passed.
    ctest_rc=${SUITEERROR}
    print_ucx_log_tails() {
        # the most recently written logs belong to the ranks that just ran
        for f in $(ls -t /tmp/ucx_debug_*.log | head -n 4); do
            echo "===== ${f} ($(wc -l < "${f}") lines) ====="
            tail -n 200 "${f}"
        done
    }
    if (( ctest_rc != 0 )); then
        rapids-logger "UCX debug: tail of UCX logs from the failed ctest run"
        print_ucx_log_tails
    fi

    rapids-logger "UCX debug: looping the ucxx StreamingShuffler tests under gdb"
    rapids-mamba-retry install -y -n test gdb || echo "WARNING: could not install gdb"
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    export OMPI_MCA_opal_cuda_support=1
    loop_deadline=$(( SECONDS + 20 * 60 ))
    loop_failed=0
    for i in $(seq 1 100); do
        if (( SECONDS >= loop_deadline )); then
            echo "UCX debug: time budget exhausted after $(( i - 1 )) iterations"
            break
        fi
        echo "UCX debug: gdb iteration ${i}"
        timeout 5m mpirun --oversubscribe --map-by node --bind-to none -np 2 \
            -x UCX_LOG_LEVEL -x UCX_LOG_FILE \
            "$(realpath ci/debug_ucx_gdb_wrap.sh)" \
            "${CONDA_PREFIX}/bin/gtests/libcudf_streaming/libcudf_streaming_ucxx_tests" \
            --gtest_filter='StreamingShuffler*' || { loop_failed=1; break; }
        # keep /tmp small: UCX debug logs of passing iterations are not interesting
        ls -t /tmp/ucx_debug_*.log 2> /dev/null | tail -n +9 | xargs -r rm -f
    done
    if (( loop_failed )); then
        rapids-logger "UCX debug: iteration ${i} failed, tail of UCX logs"
        print_ucx_log_tails
        SUITEERROR=1
        EXITCODE=1
    fi
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
