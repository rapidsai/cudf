# Copyright (c) 2020, NVIDIA CORPORATION.

import random

import fastavro
import numpy as np
import pandas as pd
import pyarrow as pa

pyarrow_dtypes_to_pandas_dtypes = {
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
    pa.string(): pd.StringDtype(),
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


def compare_content(a, b):
    if a == b:
        return
    else:
        raise ValueError(
            f"Contents of two files are different:\n left: {a} \n right: {b}"
        )


PANDAS_TO_AVRO_TYPES = {
    np.dtype("int8"): "int",
    pd.Int8Dtype(): "int",
    pd.Int16Dtype(): "int",
    pd.Int32Dtype(): "int",
    pd.Int64Dtype(): "long",
    pd.BooleanDtype(): "boolean",
    np.dtype("bool_"): "boolean",
    np.dtype("int16"): "int",
    np.dtype("int32"): "int",
    np.dtype("int64"): "long",
    np.dtype("O"): "string",
    np.dtype("float32"): "float",
    np.dtype("float64"): "double",
    np.dtype("<M8[ns]"): {"type": "long", "logicalType": "timestamp-millis"},
    np.dtype("<M8[ms]"): {"type": "long", "logicalType": "timestamp-millis"},
    np.dtype("<M8[us]"): {"type": "long", "logicalType": "timestamp-micros"},
}


def get_dtype_info(dtype):
    if dtype in PANDAS_TO_AVRO_TYPES:
        return PANDAS_TO_AVRO_TYPES[dtype]
    else:
        print(dtype)
        raise TypeError(
            "Unsupported dtype according to avro spec:"
            " https://avro.apache.org/docs/current/spec.html"
        )


def get_schema(df):

    fields = [
        {"name": col_name, "type": get_dtype_info(col_dtype)}
        for col_name, col_dtype in df.dtypes.items()
    ]
    schema = {"type": "record", "name": "Root", "fields": fields}
    return schema


def pandas_to_avro(df, file_name=None, file_io_obj=None):
    schema = get_schema(df)
    avro_schema = fastavro.parse_schema(schema)

    records = df.to_dict("records")
    if file_name is not None:
        with open(file_name, "wb") as out:
            fastavro.writer(out, avro_schema, records)
    elif file_io_obj is not None:
        fastavro.writer(file_io_obj, avro_schema, records)
