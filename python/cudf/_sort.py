# Copyright (c) 2018, NVIDIA CORPORATION.


from cudf.dataframe.buffer import Buffer
from cudf.dataframe.column import Column
from cudf.utils import cudautils
import cudf.bindings.sort as cpp_sort
from cudf.dataframe import columnops


def get_sorted_inds(by, ascending=True, na_position="last"):
    """
        Sort by the values.

        Parameters
        ----------
        by : str or list of str
            Name or list of names to sort by.
        ascending : bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {‘first’ or ‘last’}, default ‘last’
            Argument ‘first’ puts NaNs at the beginning, ‘last’ puts NaNs at
            the end.
        Returns
        -------
        col_inds : cuDF Column of indices sorted based on input

        Difference from pandas:
          * Support axis='index' only.
          * Not supporting: inplace, kind
    """
    if isinstance(by, (Column)):
        by = [by]

    inds = Buffer(cudautils.arange(len(by[0])))
    col_inds = columnops.as_column(inds).astype('int32')

    if ascending is True:
        if na_position == "last":
            na_position = 0
        elif na_position == "first":
            na_position = 1
    elif ascending is False:
        if na_position == "last":
            na_position = 1
        elif na_position == "first":
            na_position = 0

    cpp_sort.apply_order_by(by, col_inds, ascending, na_position)

    return col_inds
