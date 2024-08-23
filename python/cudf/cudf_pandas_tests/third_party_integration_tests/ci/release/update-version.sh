#!/bin/bash
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
###########################################
# cudf.pandas integration Version Updater #
###########################################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Need to distutils-normalize the versions for some use cases
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done

DEPENDENCIES=(
  cugraph
  cudf
  cuml
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" "${FILE}"
  done
done
