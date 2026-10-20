# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import sys

lines = open(sys.argv[1], errors="replace").read().splitlines()
calls = [ln for ln in lines if ln.startswith("GETHANDLE")]
print(f"GETHANDLE calls: {len(calls)}")
print("\n".join(calls[-6:]))
start = next(
    i for i, ln in enumerate(lines) if ln.startswith("Program received signal")
)
print("\n".join(lines[start : start + 3]))
for ln in lines:
    if ln.startswith("STOPPED"):
        print(ln)
bt = [i for i, ln in enumerate(lines) if re.match(r"#0 ", ln)]
if bt:
    print("\n".join(lines[bt[0] : bt[0] + 30]))
for ln in lines:
    if re.match(r"x([0-9]|1[0-9])\s", ln) or ln.startswith(
        ("pc ", "sp ", "lr ")
    ):
        print(ln)
for i, ln in enumerate(lines):
    if "=>" in ln and "0x" in ln:
        print("\n".join(lines[max(0, i - 4) : i + 5]))
        break
if calls:
    m = re.search(r"addr=0x([0-9a-f]+) len=(\d+)", calls[-1])
    a = int(m.group(1), 16)
    for ln in lines:
        parts = ln.split()
        if (
            len(parts) >= 4
            and parts[0].startswith("0x")
            and parts[1].startswith("0x")
        ):
            lo, hi = int(parts[0], 16), int(parts[1], 16)
            if lo <= a < hi:
                print("mapping containing last GETHANDLE addr:", ln)
