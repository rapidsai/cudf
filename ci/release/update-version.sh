#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
########################
# cuDF Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCX_PY_VERSION="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG}).*"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# cpp update
sed_runner 's/'"VERSION ${CURRENT_SHORT_TAG}.*"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/CMakeLists.txt

# cpp stream testing update
sed_runner 's/'"VERSION ${CURRENT_SHORT_TAG}.*"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/tests/utilities/identify_stream_usage/CMakeLists.txt

# Python update
sed_runner 's/'"cudf_version .*)"'/'"cudf_version ${NEXT_FULL_TAG})"'/g' python/cudf/CMakeLists.txt

# Strings UDF update
sed_runner 's/'"strings_udf_version .*)"'/'"strings_udf_version ${NEXT_FULL_TAG})"'/g' python/strings_udf/CMakeLists.txt

# cpp libcudf_kafka update
sed_runner 's/'"VERSION ${CURRENT_SHORT_TAG}.*"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/libcudf_kafka/CMakeLists.txt

# cpp cudf_jni update
sed_runner 's/'"VERSION ${CURRENT_SHORT_TAG}.*"'/'"VERSION ${NEXT_FULL_TAG}"'/g' java/src/main/native/CMakeLists.txt

# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' fetch_rapids.cmake

# cmake-format rapids-cmake definitions
sed_runner 's/'"branch-.*\/cmake-format-rapids-cmake.json"'/'"branch-${NEXT_SHORT_TAG}\/cmake-format-rapids-cmake.json"'/g' ci/checks/style.sh
sed_runner 's/'"branch-.*\/cmake-format-rapids-cmake.json"'/'"branch-${NEXT_SHORT_TAG}\/cmake-format-rapids-cmake.json"'/g' ci/check_style.sh

# doxyfile update
sed_runner 's/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = '${NEXT_FULL_TAG}'/g' cpp/doxygen/Doxyfile

# sphinx docs update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/cudf/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/cudf/source/conf.py

# bump rmm & dask-cuda
for FILE in conda/environments/*.yaml dependencies.yaml; do
  sed_runner "s/dask-cuda=${CURRENT_SHORT_TAG}/dask-cuda=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/rmm=${CURRENT_SHORT_TAG}/rmm=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/rmm-cu11=${CURRENT_SHORT_TAG}/rmm-cu11=${NEXT_SHORT_TAG}/g" ${FILE};
done

# Doxyfile update
sed_runner "s|\(TAGFILES.*librmm/\).*|\1${NEXT_SHORT_TAG}|" cpp/doxygen/Doxyfile

# README.md update
sed_runner "s/version == ${CURRENT_SHORT_TAG}/version == ${NEXT_SHORT_TAG}/g" README.md
sed_runner "s/cudf=${CURRENT_SHORT_TAG}/cudf=${NEXT_SHORT_TAG}/g" README.md

# Libcudf examples update
sed_runner "s/CUDF_TAG branch-${CURRENT_SHORT_TAG}/CUDF_TAG branch-${NEXT_SHORT_TAG}/" cpp/examples/basic/CMakeLists.txt
sed_runner "s/CUDF_TAG branch-${CURRENT_SHORT_TAG}/CUDF_TAG branch-${NEXT_SHORT_TAG}/" cpp/examples/strings/CMakeLists.txt

# ucx-py version update
sed_runner "s/export UCX_PY_VERSION=.*/export UCX_PY_VERSION='${NEXT_UCX_PY_VERSION}'/g" ci/gpu/build.sh
sed_runner "s/export UCX_PY_VERSION=.*/export UCX_PY_VERSION='${NEXT_UCX_PY_VERSION}'/g" ci/gpu/java.sh

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

# Wheel builds install intra-RAPIDS dependencies from same release
sed_runner "s/rmm{cuda_suffix}.*\",/rmm{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/cudf/setup.py
sed_runner "s/cudf{cuda_suffix}==.*\",/cudf{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/dask_cudf/setup.py

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
