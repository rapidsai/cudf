# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from pylibcudf.libcudf.types cimport (
    data_type,
    size_of as cpp_size_of,
    size_type,
    type_id,
)
from pylibcudf.libcudf.utilities.type_dispatcher cimport type_to_id

from pylibcudf.libcudf.types import type_id as TypeId  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import nan_policy as NanPolicy  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import null_policy as NullPolicy  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import interpolation as Interpolation  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import mask_state as MaskState  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import nan_equality as NanEquality  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import null_equality as NullEquality  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import null_order as NullOrder  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import order as Order  # no-cython-lint, isort:skip
from pylibcudf.libcudf.types import sorted as Sorted  # no-cython-lint, isort:skip

__all__ = [
    "DataType",
    "Interpolation",
    "MaskState",
    "NanEquality",
    "NanPolicy",
    "NullEquality",
    "NullOrder",
    "NullPolicy",
    "Order",
    "SIZE_TYPE",
    "SIZE_TYPE_ID",
    "Sorted",
    "TypeId",
    "size_of"
]

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

    def __hash__(self):
        return hash((self.c_obj.id(), self.c_obj.scale()))

    def __reduce__(self):
        return (type(self), (self.c_obj.id(), self.c_obj.scale()))

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

cpdef size_type size_of(DataType t):
    """Returns the size in bytes of elements of the specified data_type.

    Only fixed-width types are supported.

    For details, see :cpp:func:`size_of`.

    Parameters
    ----------
    t : DataType
        The DataType to get the size of.

    Returns
    -------
    int
        Size in bytes of an element of the specified type.
    """
    with nogil:
        return cpp_size_of(t.c_obj)

SIZE_TYPE = DataType(type_to_id[size_type]())
SIZE_TYPE_ID = SIZE_TYPE.id()
