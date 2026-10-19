# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import tempfile
import time

NUM_PROCESSES = 8
ROUNDS = 2
TIME_BUDGET_SECONDS = 90 * 60

RUNNER = """
import sys

import pytest

ids = open(sys.argv[1]).read().splitlines()
sys.exit(pytest.main(['-p', 'no:xdist', '-p', 'no:cacheprovider', '-q', *ids]))
"""

all_ids = open(sys.argv[1]).read().splitlines()
target = all_ids[-1]
start_time = time.monotonic()
probes = 0


def crashes(prefix, num_processes=NUM_PROCESSES, rounds=ROUNDS):
    global probes
    probes += 1
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join([*prefix, target]))
        list_file = f.name
    for _ in range(rounds):
        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-X",
                    "faulthandler",
                    "-c",
                    RUNNER,
                    list_file,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            for _ in range(num_processes)
        ]
        outputs = [p.communicate()[0] for p in procs]
        if any(b"Fatal Python error" in out for out in outputs):
            return True
    return False


def out_of_time():
    return time.monotonic() - start_time > TIME_BUDGET_SECONDS


def ddmin(items, expand):
    n = 2
    while len(items) >= 2 and not out_of_time():
        chunk_size = (len(items) + n - 1) // n
        chunks = [
            items[i : i + chunk_size] for i in range(0, len(items), chunk_size)
        ]
        reduced = False
        for i in range(len(chunks)):
            complement = [x for j, c in enumerate(chunks) if j != i for x in c]
            if crashes(expand(complement)):
                print(
                    f"[{probes} probes] reduced {len(items)} -> {len(complement)}",
                    flush=True,
                )
                items = complement
                n = max(n - 1, 2)
                reduced = True
                break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), 2 * n)
    return items


prefix = all_ids[:-1]
print(f"target: {target}", flush=True)
print(
    f"full prefix ({len(prefix)} tests) crashes: {crashes(prefix)}", flush=True
)
print(f"empty prefix crashes: {crashes([])}", flush=True)

modules = list(dict.fromkeys(i.split("::")[0] for i in prefix))


def expand_modules(mods):
    keep = set(mods)
    return [i for i in prefix if i.split("::")[0] in keep]


modules = ddmin(modules, expand_modules)
print(f"minimal modules ({len(modules)}): {modules}", flush=True)

tests = ddmin(expand_modules(modules), lambda ts: ts)
print(f"minimal prefix ({len(tests)} tests):", flush=True)
for t in tests:
    print(f"  {t}", flush=True)

print(
    "minimal sequence serial (1 process, 3 rounds) crashes:",
    crashes(tests, num_processes=1, rounds=3),
    flush=True,
)
print(
    "minimal sequence concurrent (8 processes) crashes:",
    crashes(tests),
    flush=True,
)
print(f"total probes: {probes}, elapsed: {time.monotonic() - start_time:.0f}s")
