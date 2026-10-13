// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";

// The user-selected canary artifact must match the exact Charon route grant.
// rapids-pre-commit-hooks: disable-next-line
const wheel = "cucim_cu12-26.12.0a24.post261008050959-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl";
const artifactRepo = "sw-rapids-release-mirror-pypi-local";
const targets = [
  { name: "gitlab-service-identity", path: "/gitlab/api/v4/user",
    valid: (value) => value.username === "svc-rapids" },
  { name: "gitlab-project-read",
    path: "/gitlab/api/v4/projects/RAPIDS%2Fsecurity%2Fpulse-scanning/repository/branches?per_page=1",
    valid: (value) => Array.isArray(value) && value.length > 0 &&
      value.every((branch) => typeof branch.name === "string") },
  { name: "artifactory-artifact-metadata",
    path: `/artifactory/artifactory/api/storage/${artifactRepo}/${wheel}`,
    valid: (value) => value.repo === artifactRepo && value.path === `/${wheel}` },
  { name: "nspect-program-versions",
    path: "/nspect-rapids-poc/pm/api/v1.0/public/programs/NSPECT-1RDT-NZVP/programVersions",
    valid: (value) => value !== null && typeof value === "object" && !value.error },
];

async function boundedJson(response, label) {
  if (!(response.headers.get("content-type") || "").includes("application/json")) {
    throw new Error(`${label}: expected a JSON API response`);
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error(`${label}: empty API response`);
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 1024 * 1024) throw new Error(`${label}: API response exceeded 1 MiB`);
      chunks.push(Buffer.from(value));
    }
  } finally {
    await reader.cancel();
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new Error(`${label}: invalid JSON API response`);
  }
}

module.exports = async function runReads(token, { expectDenied = false, fetcher = fetch } = {}) {
  const results = [];
  for (const target of targets) {
    let response;
    try {
      response = await fetcher("http://127.0.0.1:8888" + target.path, {
        method: "GET",
        headers: { "X-Charon-GHA-Token": token },
        redirect: "manual",
        signal: AbortSignal.timeout(30000),
      });
    } catch {
      throw new Error(`${target.name}: network/tunnel failure`);
    }
    if (expectDenied) {
      await response.body?.cancel();
      if (response.status !== 403) {
        throw new Error(`${target.name}: expected denial HTTP 403, received ${response.status}`);
      }
    } else {
      if (response.status !== 200) {
        await response.body?.cancel();
        throw new Error(`${target.name}: expected HTTP 200, received ${response.status}`);
      }
      const value = await boundedJson(response, target.name);
      if (!target.valid(value)) throw new Error(`${target.name}: unexpected API response identity`);
    }
    results.push({ service: target.name, result: expectDenied ? "denied" : "read-passed" });
  }
  return results;
};
