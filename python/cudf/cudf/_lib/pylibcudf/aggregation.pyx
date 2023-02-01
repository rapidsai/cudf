# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf._lib.cpp.aggregation cimport (
    groupby_aggregation,
    make_sum_aggregation,
)


# In libcudf we benefit from compile-time checks around which types of
# aggregations are valid via the multiple inheritance patterns used by
# aggregation types. In Python code we cannot get the same benefits, however,
# so we also don't need to mirror the same inheritance structures and classes.
cdef class Aggregation:
    def __init__(self):
        raise ValueError(
            "Aggregation types should not be constructed directly. "
            "Use one of the factory functions."
        )


cdef class GroupbyAggregation(Aggregation):
    # TODO: We need to decide what we actually want this API to look like. The
    # inability to have cpdef classmethods here is pretty restrictive because
    # it really is the most natural API for factories e.g.
    # GroupbyAggregation.sum(...).  For now cpdef free functions may be the
    # best that we can do but it really isn't as readable to have something
    # like groupbyaggregation_sum(...) (or any other variation `make_*` etc)
    @staticmethod
    def sum(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(make_sum_aggregation[groupby_aggregation]())
        return obj

    cdef groupby_aggregation * get(self) nogil:
        """Get the underlying agg object."""
        return self.c_obj.get()
