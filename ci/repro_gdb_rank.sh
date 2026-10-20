#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

exec gdb -batch -x "$REPRO_GDB_CMDS" --args "$@" > "$REPRO_LOG_DIR/rank.${OMPI_COMM_WORLD_RANK:-0}.log" 2>&1
