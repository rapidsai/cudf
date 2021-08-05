#!/bin/bash
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

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# cpp update
sed_runner 's/'"CUDF VERSION .* LANGUAGES"'/'"CUDF VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' cpp/CMakeLists.txt

# cpp libcudf_kafka update
sed_runner 's/'"CUDA_KAFKA VERSION .* LANGUAGES"'/'"CUDA_KAFKA VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' cpp/libcudf_kafka/CMakeLists.txt

# cpp cudf_jni update
sed_runner 's/'"CUDF_JNI VERSION .* LANGUAGES"'/'"CUDF_JNI VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' java/src/main/native/CMakeLists.txt

# doxyfile update
sed_runner 's/PROJECT_NUMBER         = .*/PROJECT_NUMBER         = '${NEXT_FULL_TAG}'/g' cpp/doxygen/Doxyfile

# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/cudf/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/cudf/source/conf.py

# bump rmm
for FILE in conda/environments/*.yml; do
  sed_runner "s/rmm=${CURRENT_SHORT_TAG}/rmm=${NEXT_SHORT_TAG}/g" ${FILE};
done

# Doxyfile update
sed_runner "s|\(TAGFILES.*librmm/\).*|\1${NEXT_SHORT_TAG}|" cpp/doxygen/Doxyfile

# README.md update
sed_runner "s/version == ${CURRENT_SHORT_TAG}/version == ${NEXT_SHORT_TAG}/g" README.md
sed_runner "s/cudf=${CURRENT_SHORT_TAG}/cudf=${NEXT_SHORT_TAG}/g" README.md

# Libcudf examples update
sed_runner "s/CUDF_TAG branch-${CURRENT_SHORT_TAG}/CUDF_TAG branch-${NEXT_SHORT_TAG}/" cpp/examples/basic/CMakeLists.txt
