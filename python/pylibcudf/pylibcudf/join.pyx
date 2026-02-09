# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cython.operator import dereference

from libc.stddef cimport size_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport optional
from libcpp.utility cimport move
from pylibcudf.libcudf cimport join as cpp_join
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport null_equality

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .expressions cimport Expression
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource

from pylibcudf.libcudf.join import set_as_build_table as SetAsBuildTable  # no-cython-lint

__all__ = [
    "conditional_full_join",
    "conditional_inner_join",
    "conditional_left_anti_join",
    "conditional_left_join",
    "conditional_left_semi_join",
    "cross_join",
    "FilteredJoin",
    "full_join",
    "inner_join",
    "left_anti_join",
    "left_join",
    "left_semi_join",
    "mixed_full_join",
    "mixed_inner_join",
    "mixed_left_anti_join",
    "mixed_left_join",
    "mixed_left_semi_join",
    "SetAsBuildTable",
]

cdef Column _column_from_gather_map(
    cpp_join.gather_map_type gather_map, Stream stream, DeviceMemoryResource mr
):
    # helper to convert a gather map to a Column
    return Column.from_libcudf(
        move(
            make_unique[column](
                move(dereference(gather_map.get())),
                device_buffer(),
                0
            )
        ),
        stream,
        mr
    )


cpdef tuple inner_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform an inner join between two tables.

    For details, see :cpp:func:`inner_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.inner_join(
            left_keys.view(), right_keys.view(), nulls_equal, stream.view(), mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef tuple left_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a left join between two tables.

    For details, see :cpp:func:`left_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.left_join(
            left_keys.view(), right_keys.view(), nulls_equal, stream.view(), mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef tuple full_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a full join between two tables.

    For details, see :cpp:func:`full_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.full_join(
            left_keys.view(), right_keys.view(), nulls_equal, stream.view(), mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef Column left_semi_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a left semi join between two tables.

    For details, see :cpp:class:`cudf::filtered_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    cdef unique_ptr[cpp_join.filtered_join] join_obj

    with nogil:
        join_obj.reset(
            new cpp_join.filtered_join(
                right_keys.view(),
                nulls_equal,
                cpp_join.set_as_build_table.RIGHT,
                stream.view()
            )
        )
        c_result = join_obj.get()[0].semi_join(
            left_keys.view(),
            stream.view(),
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), stream, mr)


cpdef Column left_anti_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a left anti join between two tables.

    For details, see :cpp:class:`cudf::filtered_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    cdef unique_ptr[cpp_join.filtered_join] join_obj

    with nogil:
        join_obj.reset(
            new cpp_join.filtered_join(
                right_keys.view(),
                nulls_equal,
                cpp_join.set_as_build_table.RIGHT,
                stream.view()
            )
        )
        c_result = join_obj.get()[0].anti_join(
            left_keys.view(),
            stream.view(),
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), stream, mr)


cpdef Table cross_join(
    Table left, Table right, Stream stream=None, DeviceMemoryResource mr=None
):
    """Perform a cross join on two tables.

    For details see :cpp:func:`cross_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right: Table
        The right table to join.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    Table
        The result of cross joining the two inputs.
    """
    cdef unique_ptr[table] result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        result = cpp_join.cross_join(
            left.view(), right.view(), stream.view(), mr.get_mr()
        )
    return Table.from_libcudf(move(result), stream, mr)


cpdef tuple conditional_inner_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional inner join between two tables.

    For details, see :cpp:func:`conditional_inner_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef optional[size_t] output_size

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.conditional_inner_join(
            left.view(),
            right.view(),
            dereference(binary_predicate.c_obj.get()),
            output_size,
            stream.view(),
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef tuple conditional_left_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional left join between two tables.

    For details, see :cpp:func:`conditional_left_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef optional[size_t] output_size

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.conditional_left_join(
            left.view(),
            right.view(),
            dereference(binary_predicate.c_obj.get()),
            output_size,
            stream.view(),
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef tuple conditional_full_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional full join between two tables.

    For details, see :cpp:func:`conditional_full_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.conditional_full_join(
            left.view(),
            right.view(),
            dereference(binary_predicate.c_obj.get()),
            stream.view(),
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef Column conditional_left_semi_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional left semi join between two tables.

    For details, see :cpp:func:`conditional_left_semi_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result
    cdef optional[size_t] output_size

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.conditional_left_semi_join(
            left.view(),
            right.view(),
            dereference(binary_predicate.c_obj.get()),
            output_size,
            stream.view(),
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), stream, mr)


cpdef Column conditional_left_anti_join(
    Table left,
    Table right,
    Expression binary_predicate,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional left anti join between two tables.

    For details, see :cpp:func:`conditional_left_anti_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result
    cdef optional[size_t] output_size

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.conditional_left_anti_join(
            left.view(),
            right.view(),
            dereference(binary_predicate.c_obj.get()),
            output_size,
            stream.view(),
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), stream, mr)


cpdef tuple mixed_inner_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed inner join between two tables.

    For details, see :cpp:func:`mixed_inner_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef cpp_join.output_size_data_type empty_optional

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.mixed_inner_join(
            left_keys.view(),
            right_keys.view(),
            left_conditional.view(),
            right_conditional.view(),
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            empty_optional,
            stream.view(),
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef tuple mixed_left_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed left join between two tables.

    For details, see :cpp:func:`mixed_left_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef cpp_join.output_size_data_type empty_optional

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.mixed_left_join(
            left_keys.view(),
            right_keys.view(),
            left_conditional.view(),
            right_conditional.view(),
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            empty_optional,
            stream.view(),
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef tuple mixed_full_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed full join between two tables.

    For details, see :cpp:func:`mixed_full_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef cpp_join.output_size_data_type empty_optional

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.mixed_full_join(
            left_keys.view(),
            right_keys.view(),
            left_conditional.view(),
            right_conditional.view(),
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            empty_optional,
            stream.view(),
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), stream, mr),
        _column_from_gather_map(move(c_result.second), stream, mr),
    )


cpdef Column mixed_left_semi_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed left semi join between two tables.

    For details, see :cpp:func:`mixed_left_semi_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.mixed_left_semi_join(
            left_keys.view(),
            right_keys.view(),
            left_conditional.view(),
            right_conditional.view(),
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            stream.view(),
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), stream, mr)


cpdef Column mixed_left_anti_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed left anti join between two tables.

    For details, see :cpp:func:`mixed_left_anti_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result

    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_join.mixed_left_anti_join(
            left_keys.view(),
            right_keys.view(),
            left_conditional.view(),
            right_conditional.view(),
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            stream.view(),
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), stream, mr)


cdef class FilteredJoin:
    """
    Filtered hash join that builds hash table on creation and probes
    results in subsequent join member functions.

    For details, see :cpp:class:`cudf::filtered_join`.
    """

    def __cinit__(
        self,
        Table build,
        null_equality compare_nulls,
        cpp_join.set_as_build_table reuse_tbl,
        double load_factor=0.5,
        Stream stream=None,
    ):
        """
        Construct a filtered hash join object for subsequent probe calls.

        Parameters
        ----------
        build : Table
            The build table, from which the hash map is built.
        compare_nulls : NullEquality
            Controls whether null join-key values should match or not.
        reuse_tbl : SetAsBuildTable
            Specifies which table to use as the build table. If LEFT, the build
            table is considered as the left table and is reused with multiple right
            (probe) tables. If RIGHT, the build table is considered as the
            right/filter table and will be applied to multiple left (probe) tables.
        load_factor : float, optional
            The desired ratio of filled slots to total slots in the hash table,
            must be in range (0,1]. Defaults to 0.5.
        stream : Stream, optional
            CUDA stream used for device memory operations and kernel launches.
        """
        stream = _get_stream(stream)

        with nogil:
            self.c_obj.reset(
                new cpp_join.filtered_join(
                    build.view(),
                    compare_nulls,
                    reuse_tbl,
                    load_factor,
                    stream.view()
                )
            )

    def semi_join(
        self,
        Table probe,
        Stream stream=None,
        DeviceMemoryResource mr=None,
    ):
        """
        Returns a column of row indices corresponding to a semi-join
        between the build table and probe table.

        For details, see :cpp:func:`cudf::filtered_join::semi_join`.

        Parameters
        ----------
        probe : Table
            The probe table.
        stream : Stream, optional
            CUDA stream used for device memory operations and kernel launches.
        mr : DeviceMemoryResource, optional
            Device memory resource used to allocate the returned column's device memory.

        Returns
        -------
        Column
            A column containing the row indices from the left table after the join.
        """
        cdef cpp_join.gather_map_type c_result

        stream = _get_stream(stream)
        mr = _get_memory_resource(mr)

        with nogil:
            c_result = self.c_obj.get()[0].semi_join(
                probe.view(),
                stream.view(),
                mr.get_mr()
            )
        return _column_from_gather_map(move(c_result), stream, mr)

    def anti_join(
        self,
        Table probe,
        Stream stream=None,
        DeviceMemoryResource mr=None,
    ):
        """
        Returns a column of row indices corresponding to an anti-join
        between the build table and probe table.

        For details, see :cpp:func:`cudf::filtered_join::anti_join`.

        Parameters
        ----------
        probe : Table
            The probe table.
        stream : Stream, optional
            CUDA stream used for device memory operations and kernel launches.
        mr : DeviceMemoryResource, optional
            Device memory resource used to allocate the returned column's device memory.

        Returns
        -------
        Column
            A column containing the row indices from the left table after the join.
        """
        cdef cpp_join.gather_map_type c_result

        stream = _get_stream(stream)
        mr = _get_memory_resource(mr)

        with nogil:
            c_result = self.c_obj.get()[0].anti_join(
                probe.view(),
                stream.view(),
                mr.get_mr()
            )
        return _column_from_gather_map(move(c_result), stream, mr)
