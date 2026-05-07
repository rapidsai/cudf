# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.column import Column
from pylibcudf.types import DataType
from pylibcudf.utils import CudaStreamLike

def get_variant_field(
    variant_column: Column,
    field_name: str,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def cast_variant(
    variant_column: Column,
    desired_type: DataType,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def extract_variant_field(
    variant_column: Column,
    field_name: str,
    desired_type: DataType,
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
