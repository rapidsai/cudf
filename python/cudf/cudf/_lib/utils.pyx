# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import numpy as np
import pyarrow as pa

import cudf

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column, column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.types cimport size_type

try:
    import ujson as json
except ImportError:
    import json

from cudf.api.types import (
    is_categorical_dtype,
    is_decimal_dtype,
    is_list_dtype,
    is_struct_dtype,
)
from cudf.utils.dtypes import np_dtypes_to_pandas_dtypes, np_to_pa_dtype

PARQUET_META_TYPE_MAP = {
    str(cudf_dtype): str(pandas_dtype)
    for cudf_dtype, pandas_dtype in np_dtypes_to_pandas_dtypes.items()
}

cdef table_view table_view_from_columns(columns) except*:
    """Create a cudf::table_view from an iterable of Columns."""
    cdef vector[column_view] column_views

    cdef Column col
    for col in columns:
        column_views.push_back(col.view())

    return table_view(column_views)


cdef table_view table_view_from_table(tbl, ignore_index=False) except*:
    """Create a cudf::table_view from a Table.

    Parameters
    ----------
    ignore_index : bool, default False
        If True, don't include the index in the columns.
    """
    return table_view_from_columns(
        tbl._index._data.columns + tbl._data.columns
        if not ignore_index and tbl._index is not None
        else tbl._data.columns
    )
cdef vector[column_view] make_column_views(object columns):
    cdef vector[column_view] views
    views.reserve(len(columns))
    for col in columns:
        views.push_back((<Column> col).view())
    return views


cdef vector[string] get_column_names(object table, object index):
    cdef vector[string] column_names
    if index is not False:
        if isinstance(table._index, cudf.core.multiindex.MultiIndex):
            for idx_name in table._index.names:
                column_names.push_back(str.encode(idx_name))
        else:
            if table._index.name is not None:
                column_names.push_back(str.encode(table._index.name))

    for col_name in table._column_names:
        column_names.push_back(str.encode(col_name))

    return column_names


cpdef generate_pandas_metadata(table, index):
    col_names = []
    types = []
    index_levels = []
    index_descriptors = []

    # Columns
    for name, col in table._data.items():
        col_names.append(name)
        if is_categorical_dtype(col):
            raise ValueError(
                "'category' column dtypes are currently not "
                + "supported by the gpu accelerated parquet writer"
            )
        elif (
            is_list_dtype(col)
            or is_struct_dtype(col)
            or is_decimal_dtype(col)
        ):
            types.append(col.dtype.to_arrow())
        else:
            # A boolean element takes 8 bits in cudf and 1 bit in
            # pyarrow. To make sure the cudf format is interperable
            # in arrow, we use `int8` type when converting from a
            # cudf boolean array.
            if col.dtype.type == np.bool_:
                types.append(pa.int8())
            else:
                types.append(np_to_pa_dtype(col.dtype))

    # Indexes
    if index is not False:
        for level, name in enumerate(table._index.names):
            if isinstance(table._index, cudf.core.multiindex.MultiIndex):
                idx = table.index.get_level_values(level)
            else:
                idx = table.index

            if isinstance(idx, cudf.core.index.RangeIndex):
                if index is None:
                    descr = {
                        "kind": "range",
                        "name": table.index.name,
                        "start": table.index.start,
                        "stop": table.index.stop,
                        "step": table.index.step,
                    }
                else:
                    # When `index=True`, RangeIndex needs to be materialized.
                    materialized_idx = cudf.Index(idx._values, name=idx.name)
                    descr = \
                        _index_level_name(
                            index_name=materialized_idx.name,
                            level=level,
                            column_names=col_names
                        )
                    index_levels.append(materialized_idx)
            else:
                descr = \
                    _index_level_name(
                        index_name=idx.name,
                        level=level,
                        column_names=col_names
                    )
                if is_categorical_dtype(idx):
                    raise ValueError(
                        "'category' column dtypes are currently not "
                        + "supported by the gpu accelerated parquet writer"
                    )
                elif is_list_dtype(idx):
                    types.append(col.dtype.to_arrow())
                else:
                    # A boolean element takes 8 bits in cudf and 1 bit in
                    # pyarrow. To make sure the cudf format is interperable
                    # in arrow, we use `int8` type when converting from a
                    # cudf boolean array.
                    if idx.dtype.type == np.bool_:
                        types.append(pa.int8())
                    else:
                        types.append(np_to_pa_dtype(idx.dtype))

                index_levels.append(idx)
            col_names.append(name)
            index_descriptors.append(descr)

    metadata = pa.pandas_compat.construct_metadata(
        columns_to_convert=[
            col
            for col in table._columns
        ],
        df=table,
        column_names=col_names,
        index_levels=index_levels,
        index_descriptors=index_descriptors,
        preserve_index=index,
        types=types,
    )

    md_dict = json.loads(metadata[b"pandas"])

    # correct metadata for list and struct and nullable numeric types
    for col_meta in md_dict["columns"]:
        if (
            col_meta["name"] in table._column_names
            and table._data[col_meta["name"]].nullable
            and col_meta["numpy_type"] in PARQUET_META_TYPE_MAP
            and col_meta["pandas_type"] != "decimal"
        ):
            col_meta["numpy_type"] = PARQUET_META_TYPE_MAP[
                col_meta["numpy_type"]
            ]
        if col_meta["numpy_type"] in ("list", "struct"):
            col_meta["numpy_type"] = "object"

    return json.dumps(md_dict)


def _index_level_name(index_name, level, column_names):
    """
    Return the name of an index level or a default name
    if `index_name` is None or is already a column name.

    Parameters
    ----------
    index_name : name of an Index object
    level : level of the Index object

    Returns
    -------
    name : str
    """
    if index_name is not None and index_name not in column_names:
        return index_name
    else:
        return f"__index_level_{level}__"


cdef columns_from_unique_ptr(
    unique_ptr[table] c_tbl
):
    """Convert a libcudf table into list of columns.

    Parameters
    ----------
    c_tbl : unique_ptr[cudf::table]
        The libcudf table whose columns will be extracted

    Returns
    -------
    list[Column]
        A list of columns.
    """
    cdef vector[unique_ptr[column]] c_columns = move(c_tbl.get().release())
    cdef vector[unique_ptr[column]].iterator it = c_columns.begin()

    cdef size_t i

    columns = [Column.from_unique_ptr(move(dereference(it+i)))
               for i in range(c_columns.size())]

    return columns


cdef data_from_unique_ptr(
    unique_ptr[table] c_tbl, column_names, index_names=None
):
    """Convert a libcudf table into a dict with an index.

    This method is intended to provide the bridge between the columns returned
    from calls to libcudf APIs and the cuDF Python Frame objects, which require
    named columns and a separate index.

    Since cuDF Python has an independent representation of a table as a
    collection of columns, this function simply returns a dict of columns
    suitable for conversion into data to be passed to cuDF constructors.
    This method returns the columns of the table in the order they are
    stored in libcudf, but calling code is responsible for partitioning and
    labeling them as needed.

    Parameters
    ----------
    c_tbl : unique_ptr[cudf::table]
        The libcudf table whose columns will be extracted
    column_names : iterable
        The keys associated with the columns in the output data.
    index_names : iterable, optional
        If provided, an iterable of strings that will be used to label the
        corresponding first set of columns into a (Multi)Index. If this
        argument is omitted, all columns are assumed to be part of the output
        table and no index is constructed.


    Returns
    -------
    tuple(Dict[str, Column], Optional[Index])
        A dict of the columns in the output table.
    """

    columns = columns_from_unique_ptr(move(c_tbl))

    # First construct the index, if any
    index = (
        # TODO: For performance, the _from_data methods of Frame types assume
        # that the passed index object is already an Index because cudf.Index
        # and cudf.as_index are expensive. As a result, this function is
        # currently somewhat inconsistent in returning a dict of columns for
        # the data while actually constructing the Index object here (instead
        # of just returning a dict for that as well). As we clean up the
        # Frame factories we may want to look for a less dissonant approach
        # that does not impose performance penalties. The same applies to
        # data_from_table_view below.
        cudf.core.index._index_from_data(
            {
                name: columns[i]
                for i, name in enumerate(index_names)
            }
        )
        if index_names is not None
        else None
    )
    n_index_columns = len(index_names) if index_names is not None else 0
    data = {
        name: columns[i + n_index_columns]
        for i, name in enumerate(column_names)
    }
    return data, index

cdef columns_from_table_view(
    table_view tv,
    object owners,
):
    """
    Given a ``cudf::table_view``, constructs a list of columns from it,
    along with referencing an owner Python object that owns the memory
    lifetime. owner must be either None or a list of column. If owner
    is a list of columns, the owner of the `i`th ``cudf::column_view``
    in the table view is ``owners[i]``. For more about memory ownership,
    see ``Column.from_column_view``.
    """

    return [
        Column.from_column_view(
            tv.column(i), owners[i] if isinstance(owners, list) else None
        ) for i in range(tv.num_columns())
    ]

cdef data_from_table_view(
    table_view tv,
    object owner,
    object column_names,
    object index_names=None
):
    """
    Given a ``cudf::table_view``, constructs a Frame from it,
    along with referencing an ``owner`` Python object that owns the memory
    lifetime. If ``owner`` is a Frame we reach inside of it and
    reach inside of each ``cudf.Column`` to make the owner of each newly
    created ``Buffer`` underneath the ``cudf.Column`` objects of the
    created Frame the respective ``Buffer`` from the relevant
    ``cudf.Column`` of the ``owner`` Frame
    """
    cdef size_type column_idx = 0
    table_owner = isinstance(owner, cudf.core.frame.Frame)

    # First construct the index, if any
    index = None
    if index_names is not None:
        index_columns = []
        for _ in index_names:
            column_owner = owner
            if table_owner:
                column_owner = owner._index._columns[column_idx]
            index_columns.append(
                Column.from_column_view(
                    tv.column(column_idx),
                    column_owner
                )
            )
            column_idx += 1
        index = cudf.core.index._index_from_data(
            dict(zip(index_names, index_columns)))

    # Construct the data dict
    cdef size_type source_column_idx = 0
    data_columns = []
    for _ in column_names:
        column_owner = owner
        if table_owner:
            column_owner = owner._columns[source_column_idx]
        data_columns.append(
            Column.from_column_view(tv.column(column_idx), column_owner)
        )
        column_idx += 1
        source_column_idx += 1

    return dict(zip(column_names, data_columns)), index
