# Copyright (c) 2023, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from cudf._lib.cpp.types cimport type_id


cdef type_id py_type_to_c_type(TypeId py_type_id) nogil:
    return <type_id> (<underlying_type_t_type_id> py_type_id)


cdef class DataType:
    """Indicator for the logical data type of an element in a column.

    This is the Cython representation of libcudf's data_type.

    Parameters
    ----------
    id : TypeId
        The type's identifier
    scale : int
        The scale associated with the data. Only used for decimal data types.
    """
    def __cinit__(self, TypeId id, int32_t scale=0):
        self.c_obj = data_type(py_type_to_c_type(id), scale)

    # TODO: Consider making both id and scale cached properties.
    cpdef TypeId id(self):
        """Get the id associated with this data type."""
        return TypeId(self.c_obj.id())

    cpdef int32_t scale(self):
        """Get the scale associated with this data type."""
        return self.c_obj.scale()

    @staticmethod
    cdef DataType from_libcudf(data_type dt):
        """Create a DataType from a libcudf data_type.

        This method is for pylibcudf's functions to use to ingest outputs of
        calling libcudf algorithms, and should generally not be needed by users
        (even direct pylibcudf Cython users).
        """
        # Spoof an empty data type then swap in the real one.
        cdef DataType ret = DataType.__new__(DataType, TypeId.EMPTY)
        ret.c_obj = dt
        return ret
