#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eEuo pipefail

echo "checking for symbol visibility issues"

LIBRARY="${1}"

echo ""
echo "Checking exported symbols in '${LIBRARY}'"
symbol_file="./symbols.txt"
readelf --dyn-syms --wide "${LIBRARY}" \
    | c++filt \
    > "${symbol_file}"

lib=ZSTD
echo "Checking for '${lib}' symbols..."
if grep -E "${lib}" "${symbol_file}"; then
    echo "ERROR: Found some exported symbols in ${LIBRARY} matching the pattern ${lib}."
    rm "${symbol_file}"
    exit 1
fi

rm "${symbol_file}"
echo "No symbol visibility issues found in ${LIBRARY}"
