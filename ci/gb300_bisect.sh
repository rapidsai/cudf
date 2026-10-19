#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDA_MAJOR="${RAPIDS_PY_CUDA_SUFFIX#cu}"

rapids-generate-pip-constraints py_test_cudf "${PIP_CONSTRAINT}" constraints

COMMITS=(
  3f1e3747a38a7cc18ea678769adc258127cde560:38686258067
  e7b312bd794ec354ef10e0f01c9452b2c05a89ec:38730740524
  35eade248ca0af75fb85860791215e4f294294ee:38742664042
  bf13ff9a1a4d51e2e63f9c71abb094ece04e5ef1:38745904459
  8922a83d766acc507f3ede7bad0609707969c898:38773569369
  14480a10aa38608873f1b7ef57ec7c7a44026c41:38775126474
  90f9065d4e2afd5cf27134bed30e8e19bb1a02d6:38790440371
  672a8eb4a9184ad3537c7979bfcce2940d4d4ed0:38806939685
  3789ec1e3b539697359a79e9eecfda2f967f57f3:38812971103
  b46b4b4e27646d39a0842b4b889e39843c716645:38818205479
  7f567e12dabb6f83811afa529f93282eb535a5c6:38825622658
  a0203ee9b40df2a5aa45b0d5fa3850c6fdfb90bd:38855235976
  a4c2737246e068fe240e67c248564bf9e26ff72f:38907149922
  70a72ddfa1c79f627ee07672fa34a3e4dc2f0791:38942900229
  73937cc6bdd711eeec5909bf65a3b21146f0f233:38950206907
  e3dababf9ff7813cd422101506ea7c30e9fee67a:38953548670
  b1dda833c49843fd9ce925c2104b09f2f2558d0f:38961358366
  439521a16224c47a5b2c63014d9038d1bcf18d2e:38964793295
  f2412d85df27fb47709c27eff0a16114099a3258:38969621274
  e4addf8d58d59bb504556c8b815303ff5f2457c5:38980141687
  816f682244c080b438792eec6bc8eda6c3b0583a:39100160924
)

SUMMARY=()
WORK="$(mktemp -d)"

for entry in "${COMMITS[@]}"; do
  sha="${entry%%:*}"
  run_id="${entry##*:}"
  dir="${WORK}/${sha}"
  mkdir -p "${dir}/wheels" "${dir}/src"

  rapids-logger "Bisect ${sha} (build run ${run_id})"

  if ! gh run download "${run_id}" --repo NVIDIA/cudf \
      --name "cudf_wheel_cpp_libcudf_aarch64_cu${CUDA_MAJOR}" \
      --name "cudf_wheel_python_pylibcudf_aarch64_abi3_cu${CUDA_MAJOR}" \
      --dir "${dir}/wheels"; then
    SUMMARY+=("${sha} DOWNLOAD_FAILED")
    continue
  fi
  git archive "${sha}" python/pylibcudf/tests | tar -x -C "${dir}/src"

  python -m venv "${dir}/env"
  # shellcheck disable=SC1091
  . "${dir}/env/bin/activate"
  rapids-pip-retry install \
      --prefer-binary \
      --constraint "${PIP_CONSTRAINT}" \
      "$(find "${dir}/wheels" -name "libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")" \
      "$(find "${dir}/wheels" -name "pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")[test]"

  for attempt in 1 2; do
    pushd "${dir}/src/python/pylibcudf/tests" > /dev/null
    set +e
    timeout 15m python -m pytest \
      --cache-clear \
      --numprocesses=8 \
      --dist=worksteal \
      -q \
      . 2>&1 | tee "${dir}/pytest-${attempt}.log"
    status=${PIPESTATUS[0]}
    set -e
    popd > /dev/null
    crashed=$(grep -o "crashed while running '[^']*'" "${dir}/pytest-${attempt}.log" | tr '\n' ' ' || true)
    SUMMARY+=("${sha} attempt=${attempt} exit=${status} ${crashed}")
  done

  deactivate
  rm -rf "${dir}/env" "${dir}/wheels"
done

rapids-logger "GB300 bisect summary (oldest first)"
printf '%s\n' "${SUMMARY[@]}"
