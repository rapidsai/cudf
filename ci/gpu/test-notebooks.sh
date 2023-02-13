#!/bin/bash
# Copyright (c) 2020-2022, NVIDIA CORPORATION.

NOTEBOOKS_DIR="$WORKSPACE/notebooks"
NBTEST="$WORKSPACE/ci/utils/nbtest.sh"
LIBCUDF_KERNEL_CACHE_PATH="$WORKSPACE/.jitcache"

cd ${NOTEBOOKS_DIR}
TOPLEVEL_NB_FOLDERS=$(find . -name *.ipynb |cut -d'/' -f2|sort -u)

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)

SKIPNBS=""

## Check env
env

EXITCODE=0

# Always run nbtest in all TOPLEVEL_NB_FOLDERS, set EXITCODE to failure
# if any run fails

cd ${NOTEBOOKS_DIR}
for nb in $(find . -name "*.ipynb"); do
    nbBasename=$(basename ${nb})
    # Skip all NBs that use dask (in the code or even in their name)
    if ((echo ${nb}|grep -qi dask) || \
        (grep -q dask ${nb})); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (suspected Dask usage, not currently automatable)"
        echo "--------------------------------------------------------------------------------"
    elif (echo " ${SKIPNBS} " | grep -q " ${nbBasename} "); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (listed in skip list)"
        echo "--------------------------------------------------------------------------------"
    else
        nvidia-smi
        ${NBTEST} ${nbBasename}
        EXITCODE=$((EXITCODE | $?))
        rm -rf ${LIBCUDF_KERNEL_CACHE_PATH}/*
    fi
done


nvidia-smi

exit ${EXITCODE}
