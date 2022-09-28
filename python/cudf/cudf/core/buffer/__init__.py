# Copyright (c) 2018-2022, NVIDIA CORPORATION.

from cudf.core.buffer.buffer import Buffer, DeviceBufferLike
from cudf.core.buffer.utils import (
    as_device_buffer_like,
    mark_columns_as_read_only_inplace,
)
