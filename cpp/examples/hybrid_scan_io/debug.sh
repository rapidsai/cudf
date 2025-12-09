#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# gdb --ex start --args ./build/hybrid_scan_io example.parquet string_col 0000001 FILEPATH
# ./build/hybrid_scan_io example.parquet string_col 0000001 FILEPATH
# ./build/hybrid_scan_io example.parquet string_col 0000001 HOST_BUFFER
./build/hybrid_scan_io example.parquet string_col 0000001 PINNED_BUFFER
