#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"$(dirname "$0")"}
mkdir -p "${RAPIDS_TESTS_DIR}/test-results"

repo_root=$(git rev-parse --show-toplevel)

TEST_DIR="${repo_root}/tests/" RAPIDS_TESTS_DIR="${RAPIDS_TESTS_DIR}" ${repo_root}/ci/ci_run_library_tests.sh "$@"
