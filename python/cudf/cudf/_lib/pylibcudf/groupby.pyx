# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.functional cimport reference_wrapper
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.groupby cimport (
    aggregation_request,
    aggregation_result,
    groupby,
    groups,
    scan_request,
)
from cudf._lib.pylibcudf.libcudf.replace cimport replace_policy
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .aggregation cimport Aggregation
from .column cimport Column
from .table cimport Table
from .types cimport null_policy, sorted
from .utils cimport _as_vector


cdef class GroupByRequest:
    """A request for a groupby aggregation or scan.

    This class is functionally polymorphic and can represent either an
    aggregation or a scan depending on the algorithm it is used with. For
    details on the libcudf types it converts to, see
    :cpp:class:`cudf::groupby::aggregation_request` and
    :cpp:class:`cudf::groupby::scan_request`.

    Parameters
    ----------
    values : Column
        The column to aggregate.
    aggregations : List[Aggregation]
        The list of aggregations to perform.
    """
    def __init__(self, Column values, list aggregations):
        self._values = values
        self._aggregations = aggregations

    cdef aggregation_request _to_libcudf_agg_request(self) except *:
        """Convert to a libcudf aggregation_request object.

        This method is for internal use only. It creates a new libcudf
        :cpp:class:`cudf::groupby::aggregation_request` object each time it is
        called.
        """
        cdef aggregation_request c_obj
        c_obj.values = self._values.view()

        cdef Aggregation agg
        for agg in self._aggregations:
            c_obj.aggregations.push_back(move(agg.clone_underlying_as_groupby()))
        return move(c_obj)

    cdef scan_request _to_libcudf_scan_request(self) except *:
        """Convert to a libcudf scan_request object.

        This method is for internal use only. It creates a new libcudf
        :cpp:class:`cudf::groupby::scan_request` object each time it is
        called.
        """
        cdef scan_request c_obj
        c_obj.values = self._values.view()

        cdef Aggregation agg
        for agg in self._aggregations:
            c_obj.aggregations.push_back(move(agg.clone_underlying_as_groupby_scan()))
        return move(c_obj)


cdef class GroupBy:
    """Group values by keys and compute various aggregate quantities.

    For details, see :cpp:class:`cudf::groupby::groupby`.

    Parameters
    ----------
    keys : Table
        The columns to group by.
    null_handling : null_policy, optional
        Whether or not to include null rows in ``keys``. Default is null_policy.EXCLUDE.
    keys_are_sorted : sorted, optional
        Whether the keys are already sorted. Default is sorted.NO.
    """
    def __init__(
        self,
        Table keys,
        null_policy null_handling=null_policy.EXCLUDE,
        sorted keys_are_sorted=sorted.NO
    ):
        self.c_obj.reset(new groupby(keys.view(), null_handling, keys_are_sorted))
        # keep a reference to the keys table so it doesn't get
        # deallocated from under us:
        self._keys = keys

    @staticmethod
    cdef tuple _parse_outputs(
        pair[unique_ptr[table], vector[aggregation_result]] c_res
    ):
        # Convert libcudf aggregation/scan outputs into pylibcudf objects.
        # This function is for internal use only.
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

    cpdef tuple aggregate(self, list requests):
        """Compute aggregations on columns.

        For details, see :cpp:func:`cudf::groupby::groupby::aggregate`.

        Parameters
        ----------
        requests : List[GroupByRequest]
            The list of `~.cudf._lib.pylibcudf.groupby.GroupByRequest` , each
            representing a set of aggregations to perform on a given column of values.

        Returns
        -------
        Tuple[Table, List[Table, ...]]
            A tuple whose first element is the unique keys and whose second
            element is a table of aggregation results. One table is returned
            for each aggregation request, with the columns corresponding to the
            sequence of aggregations in the request.
        """
        cdef GroupByRequest request
        cdef vector[aggregation_request] c_requests
        for request in requests:
            c_requests.push_back(move(request._to_libcudf_agg_request()))

        cdef pair[unique_ptr[table], vector[aggregation_result]] c_res
        # TODO: Need to capture C++ exceptions indicating that an invalid type was used.
        # We rely on libcudf to tell us this rather than checking the types beforehand
        # ourselves.
        with nogil:
            c_res = move(dereference(self.c_obj).aggregate(c_requests))
        return GroupBy._parse_outputs(move(c_res))

    cpdef tuple scan(self, list requests):
        """Compute scans on columns.

        For details, see :cpp:func:`cudf::groupby::groupby::scan`.

        Parameters
        ----------
        requests : List[GroupByRequest]
            The list of `~.cudf._lib.pylibcudf.groupby.GroupByRequest` , each
            representing a set of aggregations to perform on a given column of values.

        Returns
        -------
        Tuple[Table, List[Table, ...]]
            A tuple whose first element is the unique keys and whose second
            element is a table of aggregation results. One table is returned
            for each aggregation request, with the columns corresponding to the
            sequence of aggregations in the request.
        """
        cdef GroupByRequest request
        cdef vector[scan_request] c_requests
        for request in requests:
            c_requests.push_back(move(request._to_libcudf_scan_request()))

        cdef pair[unique_ptr[table], vector[aggregation_result]] c_res
        with nogil:
            c_res = move(dereference(self.c_obj).scan(c_requests))
        return GroupBy._parse_outputs(move(c_res))

    cpdef tuple shift(self, Table values, list offset, list fill_values):
        """Compute shifts on columns.

        For details, see :cpp:func:`cudf::groupby::groupby::shift`.

        Parameters
        ----------
        values : Table
            The columns to shift.
        offset : List[int]
            The offsets to shift by.
        fill_values : List[Scalar]
            The values to use to fill in missing values.

        Returns
        -------
        Tuple[Table, Table]
            A tuple whose first element is the group's keys and whose second
            element is a table of shifted values.
        """
        cdef vector[reference_wrapper[const scalar]] c_fill_values = \
            _as_vector(fill_values)

        cdef vector[size_type] c_offset = offset
        cdef pair[unique_ptr[table], unique_ptr[table]] c_res
        with nogil:
            c_res = move(
                dereference(self.c_obj).shift(values.view(), c_offset, c_fill_values)
            )

        return (
            Table.from_libcudf(move(c_res.first)),
            Table.from_libcudf(move(c_res.second)),
        )

    cpdef tuple replace_nulls(self, Table value, list replace_policies):
        """Replace nulls in columns.

        For details, see :cpp:func:`cudf::groupby::groupby::replace_nulls`.

        Parameters
        ----------
        values : Table
            The columns to replace nulls in.
        replace_policies : List[replace_policy]
            The policies to use to replace nulls.

        Returns
        -------
        Tuple[Table, Table]
            A tuple whose first element is the group's keys and whose second
            element is a table of values with nulls replaced.
        """
        cdef pair[unique_ptr[table], unique_ptr[table]] c_res
        cdef vector[replace_policy] c_replace_policies = replace_policies
        with nogil:
            c_res = move(
                dereference(self.c_obj).replace_nulls(value.view(), c_replace_policies)
            )

        return (
            Table.from_libcudf(move(c_res.first)),
            Table.from_libcudf(move(c_res.second)),
        )

    cpdef tuple get_groups(self, Table values=None):
        """Get the grouped keys and values labels for each row.

        For details, see :cpp:func:`cudf::groupby::groupby::get_groups`.

        Parameters
        ----------
        values : Table, optional
            The columns to get group labels for. If not specified,
            `None` is returned for the group values.

        Returns
        -------
        Tuple[List[int], Table, Table]]
            A tuple of tables containing three items:
                - A list of integer offsets into the group keys/values
                - A table of group keys
                - A table of group values or None
        """

        cdef groups c_groups
        if values:
            c_groups = dereference(self.c_obj).get_groups(values.view())
            return (
                c_groups.offsets,
                Table.from_libcudf(move(c_groups.keys)),
                Table.from_libcudf(move(c_groups.values)),
            )
        else:
            # c_groups.values is nullptr
            c_groups = dereference(self.c_obj).get_groups()
            return (
                c_groups.offsets,
                Table.from_libcudf(move(c_groups.keys)),
                None,
            )
