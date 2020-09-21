# Copyright (c) 2018, NVIDIA CORPORATION.


import logging

import numpy as np

import cudf
import cudf._libxx as libcudfxx

logging.basicConfig(format="%(levelname)s:%(message)s")


def get_sorted_inds(by, ascending=True, na_position="last"):
    """
        Sort by the values.

        Parameters
        ----------
        by : Column or list of Column
            Column or list of Column objects to sort by.
        ascending : bool or list of bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {‘first’ or ‘last’}, default ‘last’
            Argument ‘first’ puts NaNs at the beginning, ‘last’ puts NaNs at
            the end.
        Returns
        -------
        out_column_inds : cuDF Column of indices sorted based on input

        Difference from pandas:
          * Support axis='index' only.
          * Not supporting: inplace, kind
          * Ascending can be a list of bools to control per column
    """
    from cudf.core.column import ColumnBase

    number_of_columns = 1
    if isinstance(by, (ColumnBase)):
        by = by.as_frame()
    elif isinstance(by, (cudf.core.frame.Frame)):
        number_of_columns = by._num_columns

    # This needs to be updated to handle list of bools for ascending
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
    else:
        logging.warning(
            "When using a sequence of booleans for `ascending`, `na_position`"
            " flag is not yet supported and defaults to treating nulls as "
            "greater than all numbers"
        )
        na_position = 0

    # If given a scalar need to construct a sequence of length # of columns
    if np.isscalar(ascending):
        ascending = [ascending] * number_of_columns

    return libcudfxx.sort.order_by(by, ascending, na_position)
