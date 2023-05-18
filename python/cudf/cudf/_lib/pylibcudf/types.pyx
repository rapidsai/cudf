# Copyright (c) 2023, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from cudf._lib.cpp.types cimport type_id


# Cython doesn't support scoped enumerations. It assumes that enums correspond
# to their underlying value types and will thus attempt operations that are
# invalid. This code will ensure that these values are explicitly cast to the
# underlying type before casting to the final type.
cdef type_id py_type_to_c_type(TypeId py_type_id) nogil:
    return <type_id> (<underlying_type_t_type_id> py_type_id)


cdef class DataType:
    def __cinit__(self, TypeId id, int32_t scale=0):
        self.c_obj = data_type(py_type_to_c_type(id), scale)

    # TODO: Consider making both id and scale cached properties.
    cpdef TypeId id(self):
        return TypeId(self.c_obj.id())

    cpdef int32_t scale(self):
        return self.c_obj.scale()

    @staticmethod
    cdef DataType from_data_type(data_type dt):
        cdef DataType ret = DataType.__new__(DataType)
        ret.c_obj = dt
        return ret
