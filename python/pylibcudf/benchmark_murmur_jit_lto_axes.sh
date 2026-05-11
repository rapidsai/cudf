#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Murmur JIT+LTO link benchmark: sweep nvJitLink *extra* flags via CUDF_JIT_LTO_NVJITLINK_OPTIONS.
# libcudf always passes -lto; embedded fragments are LTO-IR, so omitting -lto is not supported.
#
# Skipped preset:
#   - O0  (nvJitLink ``-lto`` + ``-O0`` → ``cudaErrorLaunchFailure`` on observed stacks)
#
# Each profile clears ~/.nv/ComputeCache once, then runs two Python processes (cold/warm disk);
# each run does one in-process table sweep. All timings append to one CSV.
#
# Usage:
#   conda activate cudf_2606   # or set CONDA_ENV
#   ./python/pylibcudf/benchmark_murmur_jit_lto_axes.sh [PROFILE ...]
#
# No arguments: lto, split-compile, split-compile-extended. (-O3 is nvJitLink default with -lto;
# pass ``O3`` explicitly if needed.) ``O0`` is not run (see above).
#
# Environment:
#   CONDA_ENV   conda env name (default: cudf_2606)
#   REPO_ROOT   cuDF git root (default: inferred from this script)
#   OUTDIR      output directory (default: $REPO_ROOT/jit_lto_bench_out)
#   CSV_OUT      raw per-link CSV (default: $OUTDIR/murmur_jit_lto_bench.csv); removed at start
#   SUMMARY_CSV  pivot summary for analysis (default: $OUTDIR/murmur_jit_lto_bench_summary.csv)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-cudf_2606}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/jit_lto_bench_out}"
CSV_OUT="${CSV_OUT:-${OUTDIR}/murmur_jit_lto_bench.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTDIR}/murmur_jit_lto_bench_summary.csv}"
BENCH_PY="${REPO_ROOT}/python/pylibcudf/benchmark_murmur_jit_lto_link.py"

mkdir -p "${OUTDIR}"
rm -f "${CSV_OUT}"

declare -A NVJITLINK_PROFILE_OPTS=(
  [lto]=""
  [O0]="-O0"
  [O3]="-O3"
  [split-compile]="-split-compile=0"
  [split-compile-extended]="-split-compile-extended=0"
)

DEFAULT_PROFILES=(lto split-compile split-compile-extended)

filter_profiles() {
  local -a in=( "$@" )
  RUN_PROFILES=()
  local p
  for p in "${in[@]}"; do
    if [[ "${p}" == O0 ]]; then
      echo "Skipping profile O0: nvJitLink -lto -O0 fails at kernel launch on observed stacks." >&2
      continue
    fi
    RUN_PROFILES+=( "${p}" )
  done
}

if [[ ${#} -gt 0 ]]; then
  filter_profiles "$@"
else
  RUN_PROFILES=( "${DEFAULT_PROFILES[@]}" )
fi

if [[ ${#RUN_PROFILES[@]} -eq 0 ]]; then
  echo "No profiles left to run after filtering." >&2
  exit 1
fi

for p in "${RUN_PROFILES[@]}"; do
  if [[ ! -v NVJITLINK_PROFILE_OPTS[$p] ]]; then
    echo "Unknown profile '${p}'. Valid: ${!NVJITLINK_PROFILE_OPTS[*]}" >&2
    exit 1
  fi
done

unset CUDF_JIT_LTO_DISABLE_LTO

# Conda's cuda-nvcc activate hook uses NVCC_PREPEND_FLAGS without a default; ``set -u`` makes that
# an error. Allow unset variables only while initializing conda.
set +u
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

echo "REPO_ROOT=${REPO_ROOT}"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"
echo "OUTDIR=${OUTDIR}"
echo "CSV_OUT=${CSV_OUT}"
echo "SUMMARY_CSV=${SUMMARY_CSV}"
echo "PROFILES=${RUN_PROFILES[*]}"
echo ""

for tag in "${RUN_PROFILES[@]}"; do
  extra="${NVJITLINK_PROFILE_OPTS[$tag]}"
  if [[ -n "${extra}" ]]; then
    export CUDF_JIT_LTO_NVJITLINK_OPTIONS="${extra}"
  else
    unset CUDF_JIT_LTO_NVJITLINK_OPTIONS
  fi

  echo "======== Profile ${tag} (CUDF_JIT_LTO_NVJITLINK_OPTIONS=${extra:-<unset>}, always -lto) ========"
  echo "======== Clear CUDA compute cache ========"
  rm -rf "${HOME}/.nv/ComputeCache"

  for inv in 1 2; do
    echo "-------- Python: profile=${tag} script_invocation=${inv} (append ${CSV_OUT}) --------"
    python "${BENCH_PY}" \
      --nvjitlink-optset "${tag}" \
      --script-invocation "${inv}" \
      --output-csv "${CSV_OUT}"
  done
  echo ""
done

python "${BENCH_PY}" --summarize-bench-csv "${CSV_OUT}" --summary-output "${SUMMARY_CSV}"

echo "Done. Raw: ${CSV_OUT}  Summary: ${SUMMARY_CSV}"
