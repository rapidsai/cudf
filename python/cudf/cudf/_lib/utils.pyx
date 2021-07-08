# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import cudf

import pyarrow as pa

from cython.operator cimport dereference

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table
from cudf._lib.cpp.column.column cimport column, column_view
from cudf._lib.cpp.table.table cimport table_view
from cudf._lib.cpp.types cimport size_type

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

try:
    import ujson as json
except ImportError:
    import json

from cudf.utils.dtypes import (
    np_to_pa_dtype,
    is_categorical_dtype,
    is_list_dtype,
    is_struct_dtype,
    is_decimal_dtype,
)


cdef vector[column_view] make_column_views(object columns):
    cdef vector[column_view] views
    views.reserve(len(columns))
    for col in columns:
        views.push_back((<Column> col).view())
    return views


cdef vector[table_view] make_table_views(object tables):
    cdef vector[table_view] views
    views.reserve(len(tables))
    for tbl in tables:
        views.push_back((<Table> tbl).view())
    return views


cdef vector[table_view] make_table_data_views(object tables):
    cdef vector[table_view] views
    views.reserve(len(tables))
    for tbl in tables:
        views.push_back((<Table> tbl).data_view())
    return views


cdef vector[string] get_column_names(Table table, object index):
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


cpdef generate_pandas_metadata(Table table, index):
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

    # correct metadata for list and struct types
    for col_meta in md_dict["columns"]:
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


# TODO: Look into simplifying calling APIs that don't use the index from this.
# TODO: There's a bit of an inconsistency in use cases where calling functions
# are calling this function once to get the index and once to get the data. The
# index is converted to an index object, while the data is not. Perhaps this
# should be made more consistent.
cdef data_from_unique_ptr(
    unique_ptr[table] c_tbl, column_names, index_names=None
):
    """Convert a libcudf table into a dict with an index.

    This method is intended to provide the bridge between the columns returned
    from calls to libcudf APIs and the cuDF Python Table objects, which require
    named columns and a separate index.

    Since cuDF Python has an independent representation of a table as a
    collection of columns, this function simply returns a list of columns
    suitable for conversion into data to be passed to cuDF constructors.
    This method returns the columns of the table in the order they are
    stored in libcudf, but calling code is responsible for partitioning and
    labeling them as needed.

    Parameters
    ----------
    c_tbl : unique_ptr[cudf::table]
    index_names : iterable
    column_names : iterable

    Returns
    -------
    List[Column]
        A list of the columns in the output table.
    """
    cdef vector[unique_ptr[column]] c_columns = move(c_tbl.get().release())
    cdef vector[unique_ptr[column]].iterator it = c_columns.begin()

    # First construct the index, if any
    cdef int i

    columns = [Column.from_unique_ptr(move(dereference(it+i)))
               for i in range(c_columns.size())]

    index = (
        cudf.Index._from_data(
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


cdef data_from_table_view(
    table_view tv,
    object owner,
    object column_names,
    object index_names=None
):
    """
    Given a ``cudf::table_view``, constructs a ``cudf.Table`` from it,
    along with referencing an ``owner`` Python object that owns the memory
    lifetime. If ``owner`` is a ``cudf.Table``, we reach inside of it and
    reach inside of each ``cudf.Column`` to make the owner of each newly
    created ``Buffer`` underneath the ``cudf.Column`` objects of the
    created ``cudf.Table`` the respective ``Buffer`` from the relevant
    ``cudf.Column`` of the ``owner`` ``cudf.Table``.
    """
    cdef size_type column_idx = 0
    table_owner = isinstance(owner, Table)

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
        index = cudf.Index._from_data(dict(zip(index_names, index_columns)))

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
