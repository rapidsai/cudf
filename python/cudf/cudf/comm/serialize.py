# SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf  # noqa: F401
from cudf.core.abc import Serializable

try:
    from distributed.protocol import dask_deserialize, dask_serialize
    from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
    from distributed.utils import log_errors

    @cuda_serialize.register(Serializable)
    def cuda_serialize_cudf_object(x):
        with log_errors():
            return x.device_serialize()

    @dask_serialize.register(Serializable)
    def dask_serialize_cudf_object(x):
        with log_errors():
            return x.host_serialize()

    @cuda_deserialize.register(Serializable)
    def cuda_deserialize_cudf_object(header, frames):
        with log_errors():
            return Serializable.device_deserialize(header, frames)

    @dask_deserialize.register(Serializable)
    def dask_deserialize_cudf_object(header, frames):
        with log_errors():
            return Serializable.host_deserialize(header, frames)

except ImportError:
    # distributed is probably not installed on the system
    pass
