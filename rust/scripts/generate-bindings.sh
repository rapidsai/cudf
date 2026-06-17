#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  rust/scripts/generate-bindings.sh
  rust/scripts/generate-bindings.sh --check

Regenerate the cudf-sys bindgen output and either:
  - copy it into rust/cudf-sys/src/bindings.rs (default), or
  - verify that the checked-in bindings.rs is up to date (--check)
EOF
}

sha256_file() {
  local file="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | cut -d' ' -f1
    return
  fi

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file}" | cut -d' ' -f1
    return
  fi

  echo "unavailable"
}

print_file_summary() {
  local label="$1"
  local file="$2"
  local lines bytes sha

  lines="$(wc -l < "${file}" | tr -d ' ')"
  bytes="$(wc -c < "${file}" | tr -d ' ')"
  sha="$(sha256_file "${file}")"

  echo "${label}: ${file}" >&2
  echo "  lines=${lines} bytes=${bytes} sha256=${sha}" >&2
}

print_diff_preview() {
  local old_file="$1"
  local new_file="$2"
  local diff_file first_hunk

  diff_file="$(mktemp "${target_dir}/cudf-bindings-check.XXXXXX.diff")"
  if ! diff -u "${old_file}" "${new_file}" > "${diff_file}"; then
    first_hunk="$(sed -n '/^@@/ {p;q}' "${diff_file}")"
    if [[ -n "${first_hunk}" ]]; then
      echo "First diff hunk: ${first_hunk}" >&2
    fi

    echo "Diff preview (first 80 lines):" >&2
    sed -n '1,80p' "${diff_file}" >&2
    echo "Full diff written to: ${diff_file}" >&2
  fi
}

mode="write"
case "${1:-}" in
  "")
    ;;
  --check)
    mode="check"
    shift
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rust_dir="$(cd -- "${script_dir}/.." && pwd)"
bindings_file="${rust_dir}/cudf-sys/src/bindings.rs"

target_dir="$(
  cargo metadata \
    --format-version 1 \
    --no-deps \
    --manifest-path "${rust_dir}/Cargo.toml" \
  | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p'
)"

if [[ -z "${target_dir}" ]]; then
  echo "Failed to determine Cargo target directory" >&2
  exit 1
fi

cargo clean \
  -p cudf-sys \
  --manifest-path "${rust_dir}/Cargo.toml"

cargo build \
  -p cudf-sys \
  --features generate-bindings \
  --manifest-path "${rust_dir}/Cargo.toml"

generated_file="$(
  find "${target_dir}/debug/build" \
    -path '*/cudf-sys-*/out/cudf_bindings.rs' \
    -printf '%T@ %p\n' \
  | sort -n \
  | tail -n 1 \
  | cut -d' ' -f2-
)"

if [[ -z "${generated_file}" || ! -f "${generated_file}" ]]; then
  echo "Could not locate generated cudf_bindings.rs in ${target_dir}/debug/build" >&2
  exit 1
fi

echo "Generated: ${generated_file}"

if [[ "${mode}" == "check" ]]; then
  if cmp -s "${generated_file}" "${bindings_file}"; then
    echo "Checked-in bindings are up to date: ${bindings_file}"
    exit 0
  fi

  echo "Checked-in bindings are stale: ${bindings_file}" >&2
  print_file_summary "Checked-in bindings" "${bindings_file}"
  print_file_summary "Generated bindings" "${generated_file}"
  print_diff_preview "${bindings_file}" "${generated_file}"
  echo "Regenerate them with: ${script_dir}/generate-bindings.sh" >&2
  exit 1
fi

cp "${generated_file}" "${bindings_file}"
echo "Updated ${bindings_file}"
