#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${RAPIDS_RELEASE_CANDIDATE:-false}" != "true" ]]; then
  exit 0
fi

candidate_version="$(jq -er '.unit_package_versions["maven:cudf-java"] // .package_versions.maven' <<<"${RAPIDS_RELEASE_CANDIDATE_UNIT_VERSIONS:-}")"
actual_version="$(mvn -q -f "${repo_root}/java/pom.xml" help:evaluate -Dexpression=project.version -DforceStdout)"
if [[ "${actual_version}" != "${candidate_version}" && "${actual_version}" != "${candidate_version}-SNAPSHOT" ]]; then
  echo "cuDF Java Maven version ${actual_version} does not match frozen candidate version ${candidate_version}" >&2
  exit 1
fi

classifier="$(bash -c 'source "$1"; cudf_java_maven_classifier "$2"' _ \
  "${repo_root}/java/ci/java_classifier.sh" "${RAPIDS_CUDA_VERSION:?}")"
output_dir="${repo_root}/.release-candidate-artifacts/maven"
archive="${output_dir}/cudf-java.tar.gz"
identity="${output_dir}/cudf-java.identity.json"
classifier_dir="${repo_root}/output_jars/${classifier}"
repo_dir="${output_dir}/repository"
version_dir="${repo_dir}/ai/rapids/cudf/${candidate_version}"

if [[ ! -d "${classifier_dir}" ]]; then
  echo "cuDF Java classifier output is missing: ${classifier_dir}" >&2
  exit 1
fi

classifier_jars=("${classifier_dir}"/cudf-"${candidate_version}"-"${classifier}".jar)
pom_files=("${classifier_dir}"/cudf-"${candidate_version}".pom)
if [[ ! -f "${classifier_jars[0]}" ]]; then
  echo "cuDF Java classifier JAR is missing for ${classifier}" >&2
  exit 1
fi
if [[ ! -f "${pom_files[0]}" ]]; then
  echo "cuDF Java POM is missing for ${candidate_version}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
rm -rf "${repo_dir}" "${archive}"
mkdir -p "${version_dir}"
cp "${classifier_jars[0]}" "${version_dir}/"
# Maven consumers without a classifier resolve the primary JAR. For each
# candidate platform record, make that primary JAR the exact classifier build.
cp "${classifier_jars[0]}" "${version_dir}/cudf-${candidate_version}.jar"
cp "${pom_files[0]}" "${version_dir}/"
for attachment in sources javadoc; do
  attachment_file="${classifier_dir}/cudf-${candidate_version}-${attachment}.jar"
  if [[ -f "${attachment_file}" ]]; then
    cp "${attachment_file}" "${version_dir}/"
  fi
done
COPYFILE_DISABLE=1 tar -czf "${archive}" -C "${repo_dir}" .
arch="${RAPIDS_CONDA_ARCH:?}"
cuda_major="${RAPIDS_CUDA_VERSION%%.*}"
jq -cnS \
  --arg version "${candidate_version}" \
  --arg platform "${arch}-cuda${cuda_major}" \
  '{ecosystem: "maven", name: "ai.rapids:cudf", version: $version, platform: $platform}' \
  >"${identity}"
