# Copyright (c) 2020, NVIDIA CORPORATION.
import random

import fastavro
import numpy as np
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq
from cudf.utils.dtypes import (
    pandas_dtypes_to_cudf_dtypes,
    pyarrow_dtypes_to_pandas_dtypes,
)

ALL_POSSIBLE_VALUES = "ALL_POSSIBLE_VALUES"

_PANDAS_TO_AVRO_SCHEMA_MAP = {
    np.dtype("int8"): "int",
    pd.Int8Dtype(): ["int", "null"],
    pd.Int16Dtype(): ["int", "null"],
    pd.Int32Dtype(): ["int", "null"],
    pd.Int64Dtype(): ["long", "null"],
    pd.BooleanDtype(): ["boolean", "null"],
    pd.StringDtype(): ["string", "null"],
    np.dtype("bool_"): "boolean",
    np.dtype("int16"): "int",
    np.dtype("int32"): "int",
    np.dtype("int64"): "long",
    np.dtype("O"): "string",
    np.dtype("str"): "string",
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
    np.dtype("<M8[ns]"): {"type": "long", "logicalType": "timestamp-millis"},
    np.dtype("<M8[ms]"): {"type": "long", "logicalType": "timestamp-millis"},
    np.dtype("<M8[us]"): {"type": "long", "logicalType": "timestamp-micros"},
}


def _generate_rand_meta(obj, dtypes_list, null_frequency_override=None):
    obj._current_params = {}
    num_rows = obj._rand(obj._max_rows)
    num_cols = obj._rand(obj._max_columns)

    dtypes_meta = []

    for _ in range(num_cols):
        dtype = random.choice(dtypes_list)
        null_frequency = (
            random.uniform(0, 1)
            if null_frequency_override is None
            else null_frequency_override
        )
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


def get_avro_dtype_info(dtype):
    if dtype in _PANDAS_TO_AVRO_SCHEMA_MAP:
        return _PANDAS_TO_AVRO_SCHEMA_MAP[dtype]
    else:
        raise TypeError(
            f"Unsupported dtype({dtype}) according to avro spec:"
            f" https://avro.apache.org/docs/current/spec.html"
        )


def get_avro_schema(df):

    fields = [
        {"name": col_name, "type": get_avro_dtype_info(col_dtype)}
        for col_name, col_dtype in df.dtypes.items()
    ]
    schema = {"type": "record", "name": "Root", "fields": fields}
    return schema


def convert_nulls_to_none(records, df):
    columns_with_nulls = {col for col in df.columns if df[col].isnull().any()}
    scalar_columns_convert = [
        col
        for col in df.columns
        if df[col].dtype in pandas_dtypes_to_cudf_dtypes
        or pd.api.types.is_datetime64_dtype(df[col].dtype)
        or pd.api.types.is_timedelta64_dtype(df[col].dtype)
    ]

    for record in records:
        for col, value in record.items():
            if col in scalar_columns_convert:
                if col in columns_with_nulls and value in (pd.NA, pd.NaT):
                    record[col] = None
                else:
                    if isinstance(value, str):
                        record[col] = value
                    elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                        record[col] = int(value.value)
                    else:
                        record[col] = value.item()

    return records


def pandas_to_avro(df, file_name=None, file_io_obj=None):
    schema = get_avro_schema(df)
    avro_schema = fastavro.parse_schema(schema)

    records = df.to_dict("records")
    records = convert_nulls_to_none(records, df)

    if file_name is not None:
        with open(file_name, "wb") as out:
            fastavro.writer(out, avro_schema, records)
    elif file_io_obj is not None:
        fastavro.writer(file_io_obj, avro_schema, records)


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
