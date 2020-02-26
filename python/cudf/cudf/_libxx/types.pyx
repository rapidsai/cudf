# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
cimport cudf._libxx.cpp.types as cudf_types


np_to_cudf_types = {
    np.dtype("int8"): cudf_types.INT8,
    np.dtype("int16"): cudf_types.INT16,
    np.dtype("int32"): cudf_types.INT32,
    np.dtype("int64"): cudf_types.INT64,
    np.dtype("float32"): cudf_types.FLOAT32,
    np.dtype("float64"): cudf_types.FLOAT64,
    np.dtype("datetime64[s]"): cudf_types.TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): cudf_types.TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): cudf_types.TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): cudf_types.TIMESTAMP_NANOSECONDS,
    np.dtype("object"): cudf_types.STRING,
    np.dtype("bool"): cudf_types.BOOL8,
}

cudf_to_np_types = {
    cudf_types.INT8: np.dtype("int8"),
    cudf_types.INT16: np.dtype("int16"),
    cudf_types.INT32: np.dtype("int32"),
    cudf_types.INT64: np.dtype("int64"),
    cudf_types.FLOAT32: np.dtype("float32"),
    cudf_types.FLOAT64: np.dtype("float64"),
    cudf_types.TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    cudf_types.TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    cudf_types.TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    cudf_types.TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    cudf_types.STRING: np.dtype("object"),
    cudf_types.BOOL8: np.dtype("bool"),
}
