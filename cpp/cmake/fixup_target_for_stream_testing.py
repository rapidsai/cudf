# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import lief

# This list must be kept in sync with cpp/tests/utilities/identify_stream_usage.cpp
SYMBOLS_TO_REWRITE = [
    "cudaEventRecord",
    "cudaEventRecordWithFlags",
    "cudaLaunchKernel",
    "__cudaLaunchKernel",
    "__cudaLaunchKernel_ptsz",
    "cudaMemPrefetchAsync",
    "cudaMemcpy2DAsync",
    "cudaMemcpy2DFromArrayAsync",
    "cudaMemcpy2DToArrayAsync",
    "cudaMemcpy3DAsync",
    "cudaMemcpy3DPeerAsync",
    "cudaMemcpyAsync",
    "cudaMemcpyFromSymbolAsync",
    "cudaMemcpyToSymbolAsync",
    "cudaMemset2DAsync",
    "cudaMemset3DAsync",
    "cudaMemsetAsync",
    "cudaFreeAsync",
    "cudaMallocAsync",
    "cudaMallocFromPoolAsync",
]


for filename in sys.argv[1:]:
    elf = lief.ELF.parse(filename)

    for symbol_name in SYMBOLS_TO_REWRITE:
        if symbol := elf.get_symbol(symbol_name):
            symbol.name = f"cudf_{symbol.name}"

    elf.write(filename)
