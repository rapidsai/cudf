# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.utility cimport move

# Have to alias this so that the Pythonic name gets capitalized
from cudf._lib.cpp.aggregation cimport Kind as kind_t


# TODO: If/when https://github.com/cython/cython/issues/1271 is resolved, we
# should migrate the various factories to cpdef classmethods instead of Python
# def methods.
cdef class Aggregation:
    """A wrapper for aggregations.

    **Neither this class nor any of its subclasses should ever be instantiated
    using a standard constructor, only using one of its many factories.** The
    factory approach matches the libcudf approach, which is necessary because
    some aggregations require additional arguments beyond the kind.
    """
    def __init__(self):
        raise ValueError(
            "Aggregation types should not be constructed directly. "
            "Use one of the factory functions."
        )

    def __dealloc__(self):
        # Deletion is always handled by child classes.
        if type(self) is not Aggregation and self.c_ptr is not NULL:
            del self.c_ptr

    cpdef kind_t kind(self):
        """Get the kind of the aggregation."""
        return dereference(self.c_ptr).kind


cdef class SumAggregation(Aggregation):
    @classmethod
    def sum(cls):
        # We construct using cls but cdef as the base class because Cython
        # objects are effectively pointers so this will polymorphically store
        # the derived object. This allows us to construct the desired derived
        # object without needing to override this method in the base class.
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[rolling_aggregation] ptr = move(
            cpp_aggregation.make_sum_aggregation[rolling_aggregation]())
        out.c_ptr = ptr.release()
        return out


cdef class RollingAggregation(SumAggregation):
    pass


cdef class GroupbyAggregation(Aggregation):
    pass


cdef class GroupbyScanAggregation(Aggregation):
    pass


cdef class ReduceAggregation(Aggregation):
    pass


cdef class ScanAggregation(Aggregation):
    pass
