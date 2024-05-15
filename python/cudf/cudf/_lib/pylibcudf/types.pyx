# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from cudf._lib.pylibcudf.libcudf.types cimport data_type, type_id

from cudf._lib.pylibcudf.libcudf.types import type_id as TypeId  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import nan_policy as NanPolicy  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import null_policy as NullPolicy  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import interpolation as Interpolation  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import nan_equality as NanEquality  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import null_equality as NullEquality  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import null_order as NullOrder  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import order as Order  # no-cython-lint, isort:skip
from cudf._lib.pylibcudf.libcudf.types import sorted as Sorted  # no-cython-lint, isort:skip


cdef class DataType:
    """Indicator for the logical data type of an element in a column.

    This is the Cython representation of :cpp:class:`cudf::data_type`.

    Parameters
    ----------
    id : TypeId
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
        if not isinstance(other, DataType):
            return False
        return self.id() == other.id() and self.scale() == other.scale()

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
