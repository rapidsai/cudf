#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In-container packaging of the cuDF Java JAR for a single classifier.
#
# This script runs inside the rapidsai/ci-conda container launched by
# java/ci/build_cudf_java_jar.sh. It generates the build_java conda toolchain
# environment, compiles the JNI layer against a prebuilt static libcudf
# (mounted at /libcudf), and packages the cuDF Java JAR. The resulting
# classifier JAR and its POM are copied to /output. Then chowns /output and
# /repo/java/target to HOST_UID:HOST_GID so the host user owns the outputs.
#
# Inputs (environment variables):
#   RAPIDS_CUDA_VERSION        CUDA version, e.g. 12.9 or 12.9.1 (required).
#   PARALLEL_LEVEL             Build parallelism (default: nproc).
#   CMAKE_CUDA_ARCHITECTURES   Optional override for -DCMAKE_CUDA_ARCHITECTURES.
#   HOST_UID / HOST_GID        Chown target for /output and /repo/java/target
#                              (both required).

set -e

OUTPUT_DIR=/output
REPO_ROOT=/repo
CUDF_INSTALL_DIR=/libcudf

. /opt/conda/etc/profile.d/conda.sh

if [[ -z ${RAPIDS_CUDA_VERSION} ]]; then
  echo "Error: RAPIDS_CUDA_VERSION must be set" >&2
  exit 1
fi

if [[ -z ${PARALLEL_LEVEL} ]]; then
  PARALLEL_LEVEL=$(nproc)
fi

CUDA_MAJOR_MINOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d. -f1,2)

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generating build_java conda environment (cuda=${CUDA_MAJOR_MINOR}, arch=$(arch))"
ENV_YAML_DIR="$(mktemp -d)"
rapids-dependency-file-generator \
  --output conda \
  --file-key build_java \
  --matrix "cuda=${CUDA_MAJOR_MINOR};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n build_java
conda activate build_java

rapids-print-env

if [[ -z ${CUDACXX} ]]; then
  export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
fi
if [[ -z ${LIBCUDF_KERNEL_CACHE_PATH} ]]; then
  export LIBCUDF_KERNEL_CACHE_PATH=/tmp/rapids-kernel-cache
fi

BUILD_ARG=(
  -B
  "-Dmaven.repo.local=/tmp/.m2"
  "-Dparallel.level=${PARALLEL_LEVEL}"
  "-DskipTests=true"
  "-DCUDF_USE_PER_THREAD_DEFAULT_STREAM=ON"
  "-DCUDF_JNI_LIBCUDF_STATIC=ON"
  "-DUSE_GDS=OFF"
)

if [[ -n ${CMAKE_CUDA_ARCHITECTURES} ]]; then
  BUILD_ARG+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
fi

cd "${REPO_ROOT}/java"

CUDF_VERSION="$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout "${BUILD_ARG[@]}")"
rapids-logger "Packaging cuDF Java JAR version ${CUDF_VERSION} (libcudf: ${CUDF_INSTALL_DIR})"

# The `clean` goal is intentionally omitted: /repo/java/target is a
# bind-mount point, so when `mvn clean` attempts to remove the directory,
# it fails with EBUSY. The host wrapper (build_cudf_java_jar.sh) recreates
# the scratch dir before each container launch to guarantee target/ starts empty.
CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}" mvn package "${BUILD_ARG[@]}"

MAIN_JAR=""
for candidate in target/cudf-"${CUDF_VERSION}"-*.jar; do
  case "${candidate}" in
    *-tests.jar|*-sources.jar|*-javadoc.jar)
      continue
      ;;
  esac
  if [[ -f ${candidate} ]]; then
    MAIN_JAR=${candidate}
    break
  fi
done

if [[ -z ${MAIN_JAR} ]]; then
  echo "Error: no cuDF classifier JAR produced under target/"
  ls -l target/ || true
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cp -f "${MAIN_JAR}" "${OUTPUT_DIR}/"
cp -f pom.xml "${OUTPUT_DIR}/cudf-${CUDF_VERSION}.pom"

rapids-logger "Emitted $(basename "${MAIN_JAR}") + cudf-${CUDF_VERSION}.pom to ${OUTPUT_DIR}"

if [[ -n ${HOST_UID} && -n ${HOST_GID} ]]; then
  rapids-logger "Chowning ${OUTPUT_DIR} and ${REPO_ROOT}/java/target to ${HOST_UID}:${HOST_GID}"
  chown -R "${HOST_UID}:${HOST_GID}" "${OUTPUT_DIR}"
  chown -R "${HOST_UID}:${HOST_GID}" "${REPO_ROOT}/java/target"
fi
