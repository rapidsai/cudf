#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../  || exit 1
source ./ci/test_python_common.sh test_python_other

nvidia-smi
uname -a
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
ldconfig -p | grep -E "libcuda\.so" || true
ls -la /usr/local/cuda*/compat/libcuda* || true
find /usr/lib /usr/lib64 -name 'libcuda.so*' 2>/dev/null
conda list 2>/dev/null | grep -E "^(ucx|ucxx|libucxx|rapidsmpf|librapidsmpf|cuda-version|cuda-cudart |libcudf)" || true
which gcc cc gdb || true

python ci/repro_ucx_dmabuf.py
echo "repro script finished"
exit 0
