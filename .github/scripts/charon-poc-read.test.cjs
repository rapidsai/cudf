// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const run = require("./charon-poc-read.cjs");
// The user-selected canary artifact must match the exact Charon route grant.
// rapids-pre-commit-hooks: disable-next-line
const wheel = "cucim_cu12-26.12.0a24.post261008050959-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl";
function body(url) {
  if (url.endsWith("/user")) return { username: "svc-rapids", email: "private-profile" };
  if (url.includes("repository/branches")) return [{ name: "main" }];
  if (url.includes("api/storage")) return { repo: "sw-rapids-release-mirror-pypi-local", path: "/" + wheel };
  return { programVersions: [] };
}
test("makes only four exact GETs through the local tunnel and emits no response bodies", async () => {
  const calls = [];
  const results = await run("dummy-token", { fetcher: async (url, options) => {
    calls.push({ url, options });
    return Response.json(body(url));
  } });
  assert.equal(calls.length, 4);
  for (const { url, options } of calls) {
    assert.ok(url.startsWith("http://127.0.0.1:8888/"));
    assert.equal(options.method, "GET");
    assert.equal(options.redirect, "manual");
    assert.deepEqual(options.headers, { "X-Charon-GHA-Token": "dummy-token" });
  }
  assert.ok(!JSON.stringify(results).includes("dummy-token"));
  assert.ok(!JSON.stringify(results).includes("private-profile"));
});
test("requires actual HTTP 403 on all denied requests", async () => {
  const results = await run("dummy-token", {
    expectDenied: true, fetcher: async () => new Response("private error", { status: 403 }),
  });
  assert.equal(results.length, 4);
  for (const status of [200, 301, 401, 404, 500, 503]) {
    await assert.rejects(run("dummy-token", {
      expectDenied: true, fetcher: async () => new Response("private error", { status }),
    }), /expected denial/);
  }
});
test("fails closed on redirect and wrong service identity", async () => {
  await assert.rejects(run("dummy-token", {
    fetcher: async () => new Response("", { status: 302 }),
  }), /received 302/);
  await assert.rejects(run("dummy-token", {
    fetcher: async () => Response.json({ username: "wrong-account" }),
  }), /unexpected API response identity/);
});
test("sanitizes network errors and rejects HTML or oversized responses", async () => {
  await assert.rejects(run("dummy-token", {
    fetcher: async () => { throw new Error("dummy-token"); },
  }), { message: "gitlab-service-identity: network/tunnel failure" });
  await assert.rejects(run("dummy-token", {
    fetcher: async () => new Response("<html>login</html>"),
  }), /JSON API response/);
  await assert.rejects(run("dummy-token", {
    fetcher: async () => new Response("x".repeat(1024 * 1024 + 1), {
      headers: { "content-type": "application/json" },
    }),
  }), /exceeded/);
});
