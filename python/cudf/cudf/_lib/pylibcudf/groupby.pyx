# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp.groupby cimport (
    aggregation_request,
    aggregation_result,
    groupby,
)
from cudf._lib.cpp.table.table cimport table

from .aggregation cimport Aggregation
from .column cimport Column
from .table cimport Table


cdef class AggregationRequest:
    """A request for a groupby aggregation.

    Parameters
    ----------
    values : Column
        The column to aggregate.
    aggregations : list
        The list of aggregations to perform.
    """
    def __init__(self, Column values, list aggregations):
        self.values = values
        self.aggregations = aggregations

    cdef aggregation_request to_libcudf(self) except *:
        """Convert to a libcudf aggregation_request object.

        This method is for internal use only. It creates a new libcudf
        :cpp:class:`cudf::groupby::aggregation_request` object each time it is
        called.
        """
        cdef aggregation_request c_obj
        c_obj.values = self.values.view()

        cdef Aggregation agg
        for agg in self.aggregations:
            c_obj.aggregations.push_back(move(agg.clone_underlying_as_groupby()))
        return move(c_obj)


cdef class GroupBy:
    """Group values by keys and compute various aggregate quantities.

    Parameters
    ----------
    keys : Table
        The columns to group by.
    """
    def __init__(self, Table keys):
        self.c_obj.reset(new groupby(keys.view()))

    cpdef tuple aggregate(self, list requests):
        """Compute aggregations on columns.

        Parameters
        ----------
        requests : list
            The list of aggregation requests, each representing a set of
            aggregations to perform on a given column of values.

        Returns
        -------
        Tuple[Table, List[Table, ...]]
            A tuple whose first element is the unique keys and whose second
            element is a table of aggregation results. One table is returned
            for each aggregation request, with the columns corresponding to the
            sequence of aggregations in the request.
        """
        cdef AggregationRequest request
        cdef vector[aggregation_request] c_requests
        for request in requests:
            c_requests.push_back(move(request.to_libcudf()))

        cdef pair[unique_ptr[table], vector[aggregation_result]] c_res = move(
            dereference(self.c_obj).aggregate(c_requests)
        )
        cdef Table group_keys = Table.from_libcudf(move(c_res.first))

        cdef int i, j
        cdef list results = []
        cdef list inner_results
        for i in range(c_res.second.size()):
            inner_results = []
            for j in range(c_res.second[i].results.size()):
                inner_results.append(
                    Column.from_libcudf(move(c_res.second[i].results[j]))
                )
            results.append(Table(inner_results))
        return group_keys, results
