#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fail if a cudf classifier JAR's libcudf.so DT_NEEDs libspdlog or libfmt.
#
# Usage: check_jar_libcudf_needed.sh <cudf-*-cuda*.jar>

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cudf-classifier.jar>" >&2
  exit 2
fi

JAR="$(realpath "$1")"
if [[ ! -f ${JAR} ]]; then
  echo "Error: JAR not found: ${JAR}" >&2
  exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
  echo "Error: readelf not found (install binutils)" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip not found" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

MEMBER="$(unzip -Z1 "${JAR}" | grep -E '(^|/)libcudf\.so$' | head -1 || true)"
if [[ -z ${MEMBER} ]]; then
  echo "Error: ${JAR} does not contain libcudf.so" >&2
  exit 1
fi

# -j flattens the JAR-internal path (e.g. native/.../libcudf.so) into ${TMP}.
unzip -j -o -q "${JAR}" "${MEMBER}" -d "${TMP}"

SO="${TMP}/libcudf.so"
if [[ ! -f ${SO} ]]; then
  echo "Error: failed to extract libcudf.so from ${JAR}" >&2
  exit 1
fi

if readelf -d "${SO}" | grep -E 'NEEDED.*\[(libspdlog|libfmt)\.so'; then
  echo "Error: ${JAR} libcudf.so still depends on shared spdlog/fmt:" >&2
  readelf -d "${SO}" | grep NEEDED >&2 || true
  exit 1
fi

echo "OK: no libspdlog/libfmt DT_NEEDED in $(basename "${JAR}")"
