# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import rmm.mr
import rmm.statistics

import cudf
from cudf.utils.performance_tracking import (
    get_memory_records,
    print_memory_report,
)

# Reset RMM
rmm.mr.set_current_device_resource(rmm.mr.CudaMemoryResource())

df1 = cudf.DataFrame({"a": [1, 2, 3]})
assert len(get_memory_records()) == 0

rmm.statistics.enable_statistics()
cudf.set_option("memory_profiling", True)

df1.merge(df1)

assert len(get_memory_records()) > 0

out = StringIO()
print_memory_report(file=out)
assert "DataFrame.merge" in out.getvalue()
