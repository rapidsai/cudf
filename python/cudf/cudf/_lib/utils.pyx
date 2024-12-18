# Copyright (c) 2020-2024, NVIDIA CORPORATION.
import cudf

from cudf._lib.column cimport Column


cpdef columns_from_pylibcudf_table(tbl):
    """Convert a pylibcudf table into list of columns.

    Parameters
    ----------
    tbl : pylibcudf.Table
        The pylibcudf table whose columns will be extracted

    Returns
    -------
    list[Column]
        A list of columns.
    """
    return [Column.from_pylibcudf(plc) for plc in tbl.columns()]


cpdef _data_from_columns(columns, column_names, index_names=None):
    """Convert a list of columns into a dict with an index.

    This method is intended to provide the bridge between the columns returned
    from calls to libcudf or pylibcudf APIs and the cuDF Python Frame objects, which
    require named columns and a separate index.

    Since cuDF Python has an independent representation of a table as a
    collection of columns, this function simply returns a dict of columns
    suitable for conversion into data to be passed to cuDF constructors.
    This method returns the columns of the table in the order they are
    stored in libcudf, but calling code is responsible for partitioning and
    labeling them as needed.

    Parameters
    ----------
    columns : list[Column]
        The columns to be extracted
    column_names : iterable
        The keys associated with the columns in the output data.
    index_names : iterable, optional
        If provided, an iterable of strings that will be used to label the
        corresponding first set of columns into a (Multi)Index. If this
        argument is omitted, all columns are assumed to be part of the output
        table and no index is constructed.
    """
    # First construct the index, if any
    index = (
        # TODO: For performance, the _from_data methods of Frame types assume
        # that the passed index object is already an Index because cudf.Index
        # and cudf.as_index are expensive. As a result, this function is
        # currently somewhat inconsistent in returning a dict of columns for
        # the data while actually constructing the Index object here (instead
        # of just returning a dict for that as well). As we clean up the
        # Frame factories we may want to look for a less dissonant approach
        # that does not impose performance penalties.
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


cpdef data_from_pylibcudf_table(tbl, column_names, index_names=None):
    return _data_from_columns(
        columns_from_pylibcudf_table(tbl),
        column_names,
        index_names
    )

cpdef data_from_pylibcudf_io(tbl_with_meta, column_names=None, index_names=None):
    """
    Unpacks the TableWithMetadata from libcudf I/O
    into a dict of columns and an Index (cuDF format)
    """
    if column_names is None:
        column_names = tbl_with_meta.column_names(include_children=False)
    return _data_from_columns(
        columns=[Column.from_pylibcudf(plc) for plc in tbl_with_meta.columns],
        column_names=column_names,
        index_names=index_names
    )
