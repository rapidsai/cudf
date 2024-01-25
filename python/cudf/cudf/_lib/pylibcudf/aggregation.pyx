# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference

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

    cpdef kind_t kind(self):
        """Get the kind of the aggregation."""
        return dereference(self.c_parent_obj).kind


cdef class RollingAggregation(Aggregation):
    pass


cdef class GroupbyAggregation(Aggregation):
    pass


cdef class GroupbyScanAggregation(Aggregation):
    pass


cdef class ReduceAggregation(Aggregation):
    pass


cdef class ScanAggregation(Aggregation):
    pass
