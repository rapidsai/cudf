#!/usr/bin/env bash
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

logger "Building custreamz"
conda build conda/recipes/custreamz --python=$PYTHON
