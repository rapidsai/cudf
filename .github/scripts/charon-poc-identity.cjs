// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";

const AUDIENCE = "charon.nvidia.com";

// This validates the expected identity contract for diagnostics.
// Charon must independently verify the JWT signature and authorize every request.
module.exports = function validateIdentity(token, expected) {
  const parts = token.split(".");
  if (parts.length !== 3) throw new Error("Invalid GitHub OIDC token format");
  let claims;
  try {
    claims = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8"));
  } catch {
    throw new Error("Invalid GitHub OIDC token payload");
  }
  if (!claims || typeof claims !== "object" || Array.isArray(claims)) {
    throw new Error("Invalid GitHub OIDC claim object");
  }
  const required = {
    iss: "https://token.actions.githubusercontent.com",
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
  for (const [name, value] of Object.entries(required)) {
    if (typeof claims[name] !== "string" || claims[name] !== value) {
      throw new Error(`GitHub OIDC identity mismatch: ${name}`);
    }
  }
  if (claims.aud !== AUDIENCE &&
      !(Array.isArray(claims.aud) && claims.aud.includes(AUDIENCE))) {
    throw new Error("GitHub OIDC identity mismatch: aud");
  }
  // Emit only selected non-secret metadata. Never return the token.
  return { ...required, aud: AUDIENCE };
};
