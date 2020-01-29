# Copyright (c) 2018, NVIDIA CORPORATION.


import collections
import logging

import numpy as np

import rmm

import cudf._lib as libcudf
from cudf.core.column import ColumnBase, column
from cudf.utils import cudautils

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
        col_inds : cuDF Column of indices sorted based on input

        Difference from pandas:
          * Support axis='index' only.
          * Not supporting: inplace, kind
          * Ascending can be a list of bools to control per column
    """
    if isinstance(by, (ColumnBase)):
        by = [by]

    col_inds = column.as_column(cudautils.arange(len(by[0]), dtype="int32"))

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
            "When using a sequence of booleans for `ascending`, `na_position` "
            "flag is not yet supported and defaults to treating nulls as "
            "greater than all numbers"
        )
        na_position = 0

    # If given a scalar need to construct a sequence of length # of columns
    if np.isscalar(ascending):
        ascending = [ascending] * len(by)
    # If given a list-like need to convert to a numpy array and copy to device
    if isinstance(ascending, collections.abc.Sequence):
        # Need to flip the boolean here since libcudf has 0 as ascending
        ascending = [not val for val in ascending]
        ascending = rmm.to_device(np.array(ascending, dtype="int8"))
    else:
        raise ValueError("Must use a boolean or list of booleans")

    libcudf.sort.order_by(by, col_inds, ascending, na_position)

    return col_inds
