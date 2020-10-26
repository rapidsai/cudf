# Copyright (c) 2020, NVIDIA CORPORATION.
import random

import pandas as pd

import cudf
from cudf.tests.utils import assert_eq
from cudf.utils.dtypes import pyarrow_dtypes_to_pandas_dtypes

ALL_POSSIBLE_VALUES = "ALL_POSSIBLE_VALUES"


def _generate_rand_meta(obj, dtypes_list):
    obj._current_params = {}
    num_rows = obj._rand(obj._max_rows)
    num_cols = obj._rand(obj._max_columns)

    dtypes_meta = []

    for _ in range(num_cols):
        dtype = random.choice(dtypes_list)
        null_frequency = random.uniform(0, 1)
        cardinality = obj._rand(obj._max_rows)
        meta = dict()
        if dtype == "str":
            # We want to operate near the limits of string column
            # Hence creating a string column of size almost
            # equal to 2 Billion bytes(sizeof(int))
            if obj._max_string_length is None:
                meta["max_string_length"] = random.randrange(
                    0, int(2000000000 / num_rows)
                )
            else:
                meta["max_string_length"] = obj._max_string_length

        meta["dtype"] = dtype
        meta["null_frequency"] = null_frequency
        meta["cardinality"] = cardinality
        dtypes_meta.append(meta)
    return dtypes_meta, num_rows, num_cols


def run_test(funcs, args):
    if len(args) != 2:
        ValueError("Usage is python file_name.py function_name")

    function_name_to_run = args[1]
    try:
        funcs[function_name_to_run]()
    except KeyError:
        print(
            f"Provided function name({function_name_to_run}) does not exist."
        )


def pyarrow_to_pandas(table):
    """
    Converts a pyarrow table to a pandas dataframe
    with Nullable dtypes.

    Parameters
    ----------
    table: Pyarrow Table
        Pyarrow table to be converted to pandas

    Returns
    -------
    DataFrame
        A Pandas dataframe with nullable dtypes.
    """
    df = pd.DataFrame()

    for column in table.columns:
        if column.type in pyarrow_dtypes_to_pandas_dtypes:
            df[column._name] = pd.Series(
                column, dtype=pyarrow_dtypes_to_pandas_dtypes[column.type]
            )
        else:
            df[column._name] = column.to_pandas()

    return df


def cudf_to_pandas(df):
    pdf = df.to_pandas()
    for col in pdf.columns:
        if df[col].dtype in cudf.utils.dtypes.cudf_dtypes_to_pandas_dtypes:
            pdf[col] = pdf[col].astype(
                cudf.utils.dtypes.cudf_dtypes_to_pandas_dtypes[df[col].dtype]
            )
        elif cudf.utils.dtypes.is_categorical_dtype(df[col].dtype):
            pdf[col] = pdf[col].astype("category")
    return pdf


def compare_content(a, b):
    if a == b:
        return
    else:
        raise ValueError(
            f"Contents of two files are different:\n left: {a} \n right: {b}"
        )


def compare_dataframe(left, right, nullable=True):
    if nullable and isinstance(left, cudf.DataFrame):
        left = cudf_to_pandas(left)
    if nullable and isinstance(right, cudf.DataFrame):
        right = cudf_to_pandas(right)

    if len(left.index) == 0 and len(right.index) == 0:
        check_index_type = False
    else:
        check_index_type = True

    return assert_eq(left, right, check_index_type=check_index_type)
