# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource

from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.io.text import ByteRangeInfo
from pylibcudf.io.types import SourceInfo
from pylibcudf.utils import CudaStreamLike

__all__ = ["fetch_byte_ranges_to_device"]

def fetch_byte_ranges_to_device(
    source_info: SourceInfo,
    byte_ranges: list[ByteRangeInfo],
    stream: CudaStreamLike | None = None,
    mr: DeviceMemoryResource | None = None,
) -> list[gpumemoryview]: ...
