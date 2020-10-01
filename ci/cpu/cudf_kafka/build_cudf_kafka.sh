#!/usr/bin/env bash
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

logger "Building cudf_kafka"
conda build conda/recipes/cudf_kafka --python=$PYTHON
