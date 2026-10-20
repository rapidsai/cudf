#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

exec gdb -batch -x "$REPRO_GDB_CMDS" -ex run \
  -ex 'printf "STOPPED si_addr=%p\n", $_siginfo._sifields._sigfault.si_addr' \
  -ex 'bt 30' -ex 'info registers' -ex 'x/8i $pc-16' -ex 'info proc mappings' \
  --args "$@" > "$REPRO_LOG_DIR/rank.${OMPI_COMM_WORLD_RANK:-0}.log" 2>&1
