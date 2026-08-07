#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install PIC static libfmt.a / libspdlog.a into the static libcudf install
# prefix (same tree as rmm / rapids_logger). Versions match the shared libs
# from the ci-conda build_java env (libfmt.so.12.1.0, libspdlog.so.1.17.0).
#
# Usage:
#   . install_static_logging_libs.sh
#   install_static_logging_libs "${INSTALL_PREFIX}"

install_static_libfmt() {
  local prefix="$1"
  local src
  src="$(mktemp -d /tmp/cudf-java-fmt.XXXXXX)"

  rapids-logger "Installing static libfmt.a (12.1.0) into ${prefix}"
  curl -fsSL "https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz" \
    | tar -xz -C "${src}"
  cmake -S "${src}/fmt-12.1.0" -B "${src}/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${prefix}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DFMT_DOC=OFF \
    -DFMT_TEST=OFF
  cmake --build "${src}/build" --parallel "${PARALLEL_LEVEL}"
  cmake --install "${src}/build"
  rm -rf "${src}"
  [[ -f ${prefix}/lib/libfmt.a ]] \
    || { echo "Error: failed to install ${prefix}/lib/libfmt.a" >&2; return 1; }
}

install_static_libspdlog() {
  local prefix="$1"
  local src
  src="$(mktemp -d /tmp/cudf-java-spdlog.XXXXXX)"

  rapids-logger "Installing static libspdlog.a (v1.17.0) into ${prefix}"
  curl -fsSL "https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz" \
    | tar -xz -C "${src}"
  cmake -S "${src}/spdlog-1.17.0" -B "${src}/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${prefix}" \
    -DCMAKE_PREFIX_PATH="${prefix}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DSPDLOG_BUILD_SHARED=OFF \
    -DSPDLOG_BUILD_PIC=ON \
    -DSPDLOG_FMT_EXTERNAL=ON \
    -DSPDLOG_BUILD_EXAMPLE=OFF \
    -DSPDLOG_BUILD_TESTS=OFF \
    -DSPDLOG_BUILD_BENCH=OFF
  cmake --build "${src}/build" --parallel "${PARALLEL_LEVEL}"
  cmake --install "${src}/build"
  rm -rf "${src}"
  [[ -f ${prefix}/lib/libspdlog.a ]] \
    || { echo "Error: failed to install ${prefix}/lib/libspdlog.a" >&2; return 1; }
}

install_static_logging_libs() {
  local prefix="$1"
  if [[ -z ${prefix} ]]; then
    echo "Error: install_static_logging_libs requires an install prefix" >&2
    return 1
  fi
  mkdir -p "${prefix}"

  [[ -f ${prefix}/lib/libfmt.a ]] || install_static_libfmt "${prefix}"
  [[ -f ${prefix}/lib/libspdlog.a ]] || install_static_libspdlog "${prefix}"
}
