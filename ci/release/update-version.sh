#!/bin/bash
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
NEXT_PATCH=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCX_PY_VERSION="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG}).*"

# Need to distutils-normalize the versions for some use cases
CURRENT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${CURRENT_SHORT_TAG}'))")
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")
PATCH_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_PATCH}'))")
echo "current is ${CURRENT_SHORT_TAG_PEP440}, next is ${NEXT_SHORT_TAG_PEP440}"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

# Wheel testing script
sed_runner "s/branch-.*/branch-${NEXT_SHORT_TAG}/g" ci/test_wheel_dask_cudf.sh

DEPENDENCIES=(
  cudf
  cudf_kafka
  custreamz
  dask-cuda
  dask-cudf
  kvikio
  libkvikio
  librmm
  rapids-dask-dependency
  rmm
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*\"/g" ${FILE}
  done
done

# README.md update
sed_runner "s/version == ${CURRENT_SHORT_TAG}/version == ${NEXT_SHORT_TAG}/g" README.md
sed_runner "s/cudf=${CURRENT_SHORT_TAG}/cudf=${NEXT_SHORT_TAG}/g" README.md

# Libcudf examples update
sed_runner "s/CUDF_TAG branch-${CURRENT_SHORT_TAG}/CUDF_TAG branch-${NEXT_SHORT_TAG}/" cpp/examples/versions.cmake

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  sed_runner "s/dask-cuda.git@branch-[^\"\s]\+/dask-cuda.git@branch-${NEXT_SHORT_TAG}/g" ${FILE};
done

# Java files
NEXT_FULL_JAVA_TAG="${NEXT_SHORT_TAG}.${PATCH_PEP440}-SNAPSHOT"
sed_runner "s|<version>.*-SNAPSHOT</version>|<version>${NEXT_FULL_JAVA_TAG}</version>|g" java/pom.xml
sed_runner "s/branch-.*/branch-${NEXT_SHORT_TAG}/g" java/ci/README.md
sed_runner "s/cudf-.*-SNAPSHOT/cudf-${NEXT_FULL_JAVA_TAG}/g" java/ci/README.md

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-[0-9.]*@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done
