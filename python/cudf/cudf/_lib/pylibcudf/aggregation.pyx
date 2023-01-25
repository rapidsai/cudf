# Copyright (c) 2023, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.cast cimport dynamic_cast
# from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp.aggregation cimport (
    groupby_aggregation,
    make_sum_aggregation,
)
from cudf._lib.cpp.groupby cimport aggregation_request, aggregation_result
from cudf._lib.cpp.table.table cimport table

from .column cimport Column
from .column_view cimport ColumnView
from .table cimport Table
#
# from rmm._lib.device_buffer cimport DeviceBuffer
#
# from cudf._lib.cpp.column.column cimport column, column_contents
# from cudf._lib.cpp.types cimport size_type
#
from .table_view cimport TableView


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


ctypedef groupby_aggregation * gba_ptr


# TODO: This belongs in a separate groupby module eventually.
cdef class AggregationRequest:
    def __init__(self, ColumnView values, list aggregations):
        self.c_obj.values = dereference(values.get())

        cdef GroupbyAggregation agg
        for agg in aggregations:
            # TODO: There must be a more elegant way to do this...
            # The old Cython code paths don't have to deal with this particular
            # cast because they never clone. These classes aren't user facing
            # so we just construct internally and move directly. We could
            # consider switching away from storing unique pointers but there
            # are issues with that here because the factories for aggs return
            # unique pointers. At present I'm not even verifying that the
            # dynamic cast is valid, but I'll fix that later if we really have
            # to stick with this approach.
            self.c_obj.aggregations.push_back(
                move(
                    unique_ptr[groupby_aggregation](
                        dynamic_cast[gba_ptr](agg.get().clone().release())
                    )
                )
            )

    # TODO: This API shouldn't be necessary yet, but is currently required
    # because aggregation requests accept a vector of unique pointers rather
    # than a vector of references (should be changed).
    cdef AggregationRequest copy(self):
        cdef AggregationRequest obj = AggregationRequest.__new__()
        obj.c_obj.values = self.c_obj.values
        cdef int i
        for i in range(self.c_obj.aggregations.size()):
            obj.c_obj.aggregations.push_back(
                move(
                    unique_ptr[groupby_aggregation](
                        dynamic_cast[gba_ptr](
                            self.c_obj.aggregations[i].get().clone().release()
                        )
                    )
                )
            )
        return obj


cdef class GroupBy:
    def __init__(self, TableView keys):
        self.c_obj.reset(new groupby(dereference(keys.get())))

    cpdef aggregate(self, list requests):
        # This copy is necessary because the aggregation request contains a
        # vector of unique_ptrs rather than references. We need to change that.
        cdef list new_requests = requests.copy()

        cdef AggregationRequest request
        cdef vector[aggregation_request] c_requests
        for request in new_requests:
            # TODO: Accessing c_obj directly isn't great.
            c_requests.push_back(move(request.c_obj))

        cdef pair[unique_ptr[table], vector[aggregation_result]] c_res = move(
            self.get().aggregate(c_requests)
        )
        cdef Table group_keys = Table.from_table(move(c_res.first))

        # TODO: For now, I'm assuming that all aggregations produce a single
        # column. I'm not sure what the exceptions are, but I know there must
        # be some. I expect that to be obvious when I start replacing the
        # existing libcudf code.
        cdef int i
        cdef list results = []
        for i in range(c_res.second.size()):
            results.append(
                Column.from_column(move(c_res.second[i].results[0]))
            )
        return group_keys, results

    cdef groupby * get(self) nogil:
        """Get the underlying groupby object."""
        return self.c_obj.get()
