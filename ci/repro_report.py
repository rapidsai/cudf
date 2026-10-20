# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re
import sys

lines = open(sys.argv[1], errors="replace").read().splitlines()
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
    if re.match(r"x([0-9]|[12][0-9]|30)\s", ln) or ln.startswith(
        ("pc ", "sp ", "lr ")
    ):
        print(ln)
for i, ln in enumerate(lines):
    if "=>" in ln and "0x" in ln:
        print("\n".join(lines[max(0, i - 24) : i + 6]))
        break
pc = None
for ln in lines:
    if ln.startswith("pc "):
        pc = int(ln.split()[1], 16)
for ln in lines:
    parts = ln.split()
    if (
        len(parts) >= 4
        and parts[0].startswith("0x")
        and parts[1].startswith("0x")
    ):
        try:
            lo, hi = int(parts[0], 16), int(parts[1], 16)
        except ValueError:
            continue
        if pc is not None and lo <= pc < hi:
            print("mapping containing pc:", ln)
