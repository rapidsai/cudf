# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from cudf._lib.pylibcudf.libcudf.types cimport data_type, type_id

from cudf._lib.pylibcudf.libcudf.types import type_id as TypeId  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import nan_policy as NanPolicy  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import null_policy as NullPolicy  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import interpolation as Interpolation  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import mask_state as MaskState  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import nan_equality as NanEquality  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import null_equality as NullEquality  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import null_order as NullOrder  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import order as Order  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import sorted as Sorted  # no-cython-lint, isort:skip

import numpy as np


cdef class DataType:
    """Indicator for the logical data type of an element in a column.

    This is the Cython representation of :cpp:class:`cudf::data_type`.

    Parameters
    ----------
    id : type_id
        The type's identifier
    scale : int
        The scale associated with the data. Only used for decimal data types.
    """
    def __cinit__(self, type_id id, int32_t scale=0):
        if (
            id == type_id.DECIMAL32
            or id == type_id.DECIMAL64
            or id == type_id.DECIMAL128
        ):
            self.c_obj = data_type(id, scale)
        else:
            self.c_obj = data_type(id)

    # TODO: Consider making both id and scale cached properties.
    cpdef type_id id(self):
        """Get the id associated with this data type."""
        return self.c_obj.id()

    cpdef int32_t scale(self):
        """Get the scale associated with this data type."""
        return self.c_obj.scale()

    def __eq__(self, other):
        return type(self) is type(other) and (
            self.c_obj == (<DataType>other).c_obj
        )

    @staticmethod
    cdef DataType from_libcudf(data_type dt):
        """Create a DataType from a libcudf data_type.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # Spoof an empty data type then swap in the real one.
        cdef DataType ret = DataType.__new__(DataType, type_id.EMPTY)
        ret.c_obj = dt
        return ret

SUPPORTED_NUMPY_TO_LIBCUDF_TYPES = {
    np.dtype("int8"): TypeId.INT8,
    np.dtype("int16"): TypeId.INT16,
    np.dtype("int32"): TypeId.INT32,
    np.dtype("int64"): TypeId.INT64,
    np.dtype("uint8"): TypeId.UINT8,
    np.dtype("uint16"): TypeId.UINT16,
    np.dtype("uint32"): TypeId.UINT32,
    np.dtype("uint64"): TypeId.UINT64,
    np.dtype("float32"): TypeId.FLOAT32,
    np.dtype("float64"): TypeId.FLOAT64,
    np.dtype("datetime64[s]"): TypeId.TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): TypeId.TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): TypeId.TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): TypeId.TIMESTAMP_NANOSECONDS,
    np.dtype("object"): TypeId.STRING,
    np.dtype("bool"): TypeId.BOOL8,
    np.dtype("timedelta64[s]"): TypeId.DURATION_SECONDS,
    np.dtype("timedelta64[ms]"): TypeId.DURATION_MILLISECONDS,
    np.dtype("timedelta64[us]"): TypeId.DURATION_MICROSECONDS,
    np.dtype("timedelta64[ns]"): TypeId.DURATION_NANOSECONDS,
}

LIBCUDF_TO_SUPPORTED_NUMPY_TYPES = {
    # There's no equivalent to EMPTY in cudf.  We translate EMPTY
    # columns from libcudf to ``int8`` columns of all nulls in Python.
    # ``int8`` is chosen because it uses the least amount of memory.
    TypeId.EMPTY: np.dtype("int8"),
    TypeId.INT8: np.dtype("int8"),
    TypeId.INT16: np.dtype("int16"),
    TypeId.INT32: np.dtype("int32"),
    TypeId.INT64: np.dtype("int64"),
    TypeId.UINT8: np.dtype("uint8"),
    TypeId.UINT16: np.dtype("uint16"),
    TypeId.UINT32: np.dtype("uint32"),
    TypeId.UINT64: np.dtype("uint64"),
    TypeId.FLOAT32: np.dtype("float32"),
    TypeId.FLOAT64: np.dtype("float64"),
    TypeId.BOOL8: np.dtype("bool"),
    TypeId.TIMESTAMP_SECONDS: np.dtype("datetime64[s]"),
    TypeId.TIMESTAMP_MILLISECONDS: np.dtype("datetime64[ms]"),
    TypeId.TIMESTAMP_MICROSECONDS: np.dtype("datetime64[us]"),
    TypeId.TIMESTAMP_NANOSECONDS: np.dtype("datetime64[ns]"),
    TypeId.DURATION_SECONDS: np.dtype("timedelta64[s]"),
    TypeId.DURATION_MILLISECONDS: np.dtype("timedelta64[ms]"),
    TypeId.DURATION_MICROSECONDS: np.dtype("timedelta64[us]"),
    TypeId.DURATION_NANOSECONDS: np.dtype("timedelta64[ns]"),
    TypeId.STRING: np.dtype("object"),
    TypeId.STRUCT: np.dtype("object"),
}
