#!/bin/bash

NOTEBOOKS_DIR="${GITHUB_WORKSPACE}/notebooks"
NBTEST="${GITHUB_WORKSPACE}/ci/utils/nbtest.sh"

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)
SKIPNBS=""

cd "${NOTEBOOKS_DIR}" || exit 1
EXITCODE=0
for nb in $(find . -name "*.ipynb"); do
    nbBasename=$(basename "${nb}")
    # Skip all NBs that use dask (in the code or even in their name)
    if ( (echo "${nb}"|grep -qi dask) || \
        (grep -q dask "${nb}")); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (suspected Dask usage, not currently automatable)"
        echo "--------------------------------------------------------------------------------"
    elif (echo " ${SKIPNBS} " | grep -q " ${nbBasename} "); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (listed in skip list)"
        echo "--------------------------------------------------------------------------------"
    else
        nvidia-smi
        "${NBTEST}" "${nbBasename}"
        EXITCODE=$((EXITCODE | $?))
    fi
done

nvidia-smi

exit ${EXITCODE}
