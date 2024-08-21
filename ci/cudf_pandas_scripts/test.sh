#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail


DEPENDENCIES_PATH="../../dependencies.yaml"
package_name="pandas"

# Use grep to find the line containing the package name and version constraint
pandas_version_constraint=$(grep -oP "pandas>=\d+\.\d+,\<\d+\.\d+\.\d+dev\d+" $DEPENDENCIES_PATH)

# Function to display script usage
# function display_usage {
#     echo "Usage: $0 [--no-cudf] [pandas-version]"
# }


output=$(python fetch_pandas_versions.py $pandas_version_constraint)

# Remove the brackets and spaces from the output to get a comma-separated list
output=$(echo $output | tr -d "[] \'\'")

# Convert the comma-separated list into an array
IFS=',' read -r -a versions <<< "$output"

for version in "${versions[@]}"; do
    echo "Installing pandas version: $version"
    python -m pip install pandas==$version
done
# python -m pytest -p cudf.pandas \
#     --cov-config=./python/cudf/.coveragerc \
#     --cov=cudf \
#     --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-pandas-coverage.xml" \
#     --cov-report=term \
#     ./python/cudf/cudf_pandas_tests/
