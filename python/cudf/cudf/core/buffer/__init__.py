# Copyright (c) 2018-2022, NVIDIA CORPORATION.

from cudf.core.buffer.buffer import Buffer, DeviceBufferLike
from cudf.core.buffer.utils import (
    as_device_buffer_like,
    get_spill_lock,
    mark_columns_as_read_only_inplace,
    with_spill_lock,
)
