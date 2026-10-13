// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const validate = require("./charon-poc-identity.cjs");

const expected = {
  repository: "rapidsai/rapids-logger",
  repositoryId: "891001991",
  owner: "rapidsai",
  ownerId: "43887749",
  ref: "refs/heads/testing/charon-poc",
  sha: "abc123",
  event: "push",
};
const claims = {
  iss: "https://token.actions.githubusercontent.com",
  aud: "charon.nvidia.com",
  enterprise: "nvidia",
  enterprise_id: "6988",
  repository: expected.repository,
  repository_id: expected.repositoryId,
  repository_owner: expected.owner,
  repository_owner_id: expected.ownerId,
  ref: expected.ref,
  sha: expected.sha,
  workflow_ref: `${expected.repository}/.github/workflows/charon-poc.yml@${expected.ref}`,
  event_name: expected.event,
};
function token(payload) {
  return `header.${Buffer.from(JSON.stringify(payload)).toString("base64url")}.signature`;
}

test("accepts the exact contract without returning arbitrary claims", () => {
  const output = validate(token({ ...claims, secret: "dummy-secret" }), expected);
  assert.equal(output.repository_id, "891001991");
  assert.ok(!JSON.stringify(output).includes("dummy-secret"));
  assert.ok(!JSON.stringify(output).includes("signature"));
});

for (const name of Object.keys(claims)) {
  test(`rejects a mismatched ${name} without echoing the value`, () => {
    const input = token({ ...claims, [name]: "dummy-sensitive-value" });
    assert.throws(() => validate(input, expected), (error) => {
      assert.ok(error.message.includes(name));
      assert.ok(!error.message.includes("dummy-sensitive-value"));
      return true;
    });
  });
}

test("rejects missing immutable IDs", () => {
  const { repository_id, repository_owner_id, ...missing } = claims;
  assert.throws(() => validate(token(missing), expected), /repository_id/);
});

test("accepts audience arrays containing the Charon audience", () => {
  validate(token({ ...claims, aud: ["other", "charon.nvidia.com"] }), expected);
});

test("rejects malformed tokens without echoing their content", () => {
  assert.throws(() => validate("dummy-secret", expected), /format/);
  assert.throws(() => validate("header.not-json.signature", expected), /payload/);
});
