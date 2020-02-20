import numpy as np


np_to_cudf_types = {
    np.dtype("int8"): INT8,
    np.dtype("int16"): INT16,
    np.dtype("int32"): INT32,
    np.dtype("int64"): INT64,
    np.dtype("float32"): FLOAT32,
    np.dtype("float64"): FLOAT64,
    np.dtype("datetime64[s]"): TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): TIMESTAMP_NANOSECONDS,
    np.dtype("object"): STRING,
    np.dtype("bool"): BOOL8,
}

cudf_to_np_types = {
    INT8: np.dtype("int8"),
    INT16: np.dtype("int16"),
    INT32: np.dtype("int32"),
    INT64: np.dtype("int64"),
    FLOAT32: np.dtype("float32"),
    FLOAT64: np.dtype("float64"),
    TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    STRING: np.dtype("object"),
    BOOL8: np.dtype("bool"),
}
