# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import pylibcudf

from cudf._lib.utils cimport columns_from_pylibcudf_table

from cudf.core.buffer import acquire_spill_lock
from cudf.core.dtypes import ListDtype, StructDtype


def from_dlpack(object dlpack_capsule):
    """
    Converts a DLPack Tensor PyCapsule into a list of columns.

    DLPack Tensor PyCapsule is expected to have the name "dltensor".
    """
    return columns_from_pylibcudf_table(
        pylibcudf.interop.from_dlpack(dlpack_capsule)
    )


def to_dlpack(list source_columns):
    """
    Converts a list of columns into a DLPack Tensor PyCapsule.

    DLPack Tensor PyCapsule will have the name "dltensor".
    """
    return pylibcudf.interop.to_dlpack(
        pylibcudf.Table(
            [col.to_pylibcudf(mode="read") for col in source_columns]
        )
    )


def gather_metadata(object cols_dtypes):
    """
    Generates a ColumnMetadata vector for each column.

    Parameters
    ----------
    cols_dtypes : iterable
        An iterable of ``(column_name, dtype)`` pairs.
    """
    cpp_metadata = []
    if cols_dtypes is not None:
        for idx, (col_name, col_dtype) in enumerate(cols_dtypes):
            cpp_metadata.append(pylibcudf.interop.ColumnMetadata(col_name))
            if isinstance(col_dtype, (ListDtype, StructDtype)):
                _set_col_children_metadata(col_dtype, cpp_metadata[idx])
    else:
        raise TypeError(
            "An iterable of (column_name, dtype) pairs is required to "
            "construct column_metadata"
        )
    return cpp_metadata


def _set_col_children_metadata(dtype, col_meta):
    if isinstance(dtype, StructDtype):
        for name, value in dtype.fields.items():
            element_metadata = pylibcudf.interop.ColumnMetadata(name)
            _set_col_children_metadata(value, element_metadata)
            col_meta.children_meta.append(element_metadata)
    elif isinstance(dtype, ListDtype):
        # Offsets - child 0
        col_meta.children_meta.append(pylibcudf.interop.ColumnMetadata())

        # Element column - child 1
        element_metadata = pylibcudf.interop.ColumnMetadata()
        _set_col_children_metadata(dtype.element_type, element_metadata)
        col_meta.children_meta.append(element_metadata)
    else:
        col_meta.children_meta.append(pylibcudf.interop.ColumnMetadata())


@acquire_spill_lock()
def to_arrow(list source_columns, object column_dtypes):
    """Convert a list of columns from
    cudf Frame to a PyArrow Table.

    Parameters
    ----------
    source_columns : a list of columns to convert
    column_dtypes : Iterable of ``(column_name, column_dtype)`` pairs

    Returns
    -------
    pyarrow table
    """
    cpp_metadata = gather_metadata(column_dtypes)
    return pylibcudf.interop.to_arrow(
        pylibcudf.Table([c.to_pylibcudf(mode="read") for c in source_columns]),
        cpp_metadata,
    )


@acquire_spill_lock()
def from_arrow(object input_table):
    """Convert from PyArrow Table to a list of columns.

    Parameters
    ----------
    input_table : PyArrow table

    Returns
    -------
    A list of columns to construct Frame object
    """
    return columns_from_pylibcudf_table(
        pylibcudf.interop.from_arrow(input_table)
    )
