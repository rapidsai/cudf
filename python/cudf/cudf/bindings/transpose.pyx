# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from libc.stdlib cimport free
from libcpp.vector cimport vector

from cudf.dataframe.columnops import column_empty
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.series import Series
from cudf.dataframe.buffer import Buffer


def transpose(df):
    """Transpose index and columns.

    See Also
    --------
    cudf.dataframe.DataFrame.transpose
    """

    if len(df.columns) == 0:
        return df

    dtype = df.dtypes.iloc[0]
    if pd.api.types.is_categorical_dtype(dtype):
        raise NotImplementedError('Categorical columns are not yet '
                                  'supported for function')
    elif np.dtype(dtype).kind in 'OU':
        raise NotImplementedError('String columns are not yet '
                                  'supported for function')

    if any(t != dtype for t in df.dtypes):
        raise ValueError('all columns must have the same dtype')
    has_null = any(c.null_count for c in df._cols.values())

    out_df = DataFrame()

    ncols = len(df.columns)
    cdef vector[gdf_column*] cols
    for col in df._cols:
        cols.push_back(column_view_from_column(df[col]._column))

    new_nrow = ncols
    new_ncol = len(df)

    new_col_series = [
        Series(column_empty(new_nrow, dtype=dtype, masked=has_null))
        for i in range(0, new_ncol)
    ]

    cdef vector[gdf_column*] new_cols
    for i in range(0, new_ncol):
        new_cols.push_back(column_view_from_column(new_col_series[i]._column))

    with nogil:
        result = gdf_transpose(
            ncols,
            cols.data(),
            new_cols.data()
        )

    for i in range(ncols):
        free(cols[i])
    for i in range(new_ncol):
        free(new_cols[i])

    check_gdf_error(result)

    for series in new_col_series:
        series._column._update_null_count()

    for i in range(0, new_ncol):
        out_df[str(i)] = new_col_series[i]
    return out_df

