# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import random
from collections import OrderedDict

import fastavro
import numpy as np
import pandas as pd
import pyarrow as pa
import pyorc

import cudf
from cudf.testing._utils import assert_eq
from cudf.utils.dtypes import (
    pandas_dtypes_to_np_dtypes,
    pyarrow_dtypes_to_pandas_dtypes,
)

ALL_POSSIBLE_VALUES = "ALL_POSSIBLE_VALUES"

_PANDAS_TO_AVRO_SCHEMA_MAP = {
    cudf.dtype("int8"): "int",
    pd.Int8Dtype(): ["int", "null"],
    pd.Int16Dtype(): ["int", "null"],
    pd.Int32Dtype(): ["int", "null"],
    pd.Int64Dtype(): ["long", "null"],
    pd.Float32Dtype(): ["float", "null"],
    pd.Float64Dtype(): ["double", "null"],
    pd.BooleanDtype(): ["boolean", "null"],
    pd.StringDtype(): ["string", "null"],
    cudf.dtype("bool_"): "boolean",
    cudf.dtype("int16"): "int",
    cudf.dtype("int32"): "int",
    cudf.dtype("int64"): "long",
    cudf.dtype("O"): "string",
    cudf.dtype("str"): "string",
    cudf.dtype("float32"): "float",
    cudf.dtype("float64"): "double",
    cudf.dtype("<M8[ns]"): {"type": "long", "logicalType": "timestamp-millis"},
    cudf.dtype("<M8[ms]"): {"type": "long", "logicalType": "timestamp-millis"},
    cudf.dtype("<M8[us]"): {"type": "long", "logicalType": "timestamp-micros"},
}

PANDAS_TO_ORC_TYPES = {
    cudf.dtype("int8"): pyorc.TinyInt(),
    pd.Int8Dtype(): pyorc.TinyInt(),
    pd.Int16Dtype(): pyorc.SmallInt(),
    pd.Int32Dtype(): pyorc.Int(),
    pd.Int64Dtype(): pyorc.BigInt(),
    pd.Float32Dtype(): pyorc.Float(),
    pd.Float64Dtype(): pyorc.Double(),
    pd.BooleanDtype(): pyorc.Boolean(),
    cudf.dtype("bool_"): pyorc.Boolean(),
    cudf.dtype("int16"): pyorc.SmallInt(),
    cudf.dtype("int32"): pyorc.Int(),
    cudf.dtype("int64"): pyorc.BigInt(),
    cudf.dtype("O"): pyorc.String(),
    pd.StringDtype(): pyorc.String(),
    cudf.dtype("float32"): pyorc.Float(),
    cudf.dtype("float64"): pyorc.Double(),
    cudf.dtype("<M8[ns]"): pyorc.Timestamp(),
    cudf.dtype("<M8[ms]"): pyorc.Timestamp(),
    cudf.dtype("<M8[us]"): pyorc.Timestamp(),
}

ORC_TO_PANDAS_TYPES = {
    pyorc.TinyInt().name: pd.Int8Dtype(),
    pyorc.Int().name: pd.Int32Dtype(),
    pyorc.Boolean().name: pd.BooleanDtype(),
    pyorc.SmallInt().name: pd.Int16Dtype(),
    pyorc.BigInt().name: pd.Int64Dtype(),
    pyorc.String().name: pd.StringDtype(),
    pyorc.Float().name: pd.Float32Dtype(),
    pyorc.Double().name: pd.Float64Dtype(),
    pyorc.Timestamp().name: cudf.dtype("<M8[ns]"),
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
        # `cardinality` has to be at least 1.
        cardinality = max(1, obj._rand(obj._max_rows))
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
        elif dtype == "list":
            if obj._max_lists_length is None:
                meta["lists_max_length"] = np.random.randint(0, 2000000000)
            else:
                meta["lists_max_length"] = obj._max_lists_length

            if obj._max_lists_nesting_depth is None:
                meta["nesting_max_depth"] = np.random.randint(
                    1, np.iinfo("int64").max
                )
            else:
                meta["nesting_max_depth"] = obj._max_lists_nesting_depth

            meta["value_type"] = random.choice(
                list(cudf.utils.dtypes.ALL_TYPES - {"category"})
            )
        elif dtype == "struct":
            if obj._max_lists_nesting_depth is None:
                meta["nesting_max_depth"] = np.random.randint(2, 10)
            else:
                meta["nesting_max_depth"] = obj._max_lists_nesting_depth

            if obj._max_struct_null_frequency is None:
                meta["max_null_frequency"] = random.uniform(0, 1)
            else:
                meta["max_null_frequency"] = obj._max_struct_null_frequency

            if obj._max_struct_types_at_each_level is None:
                meta["max_types_at_each_level"] = np.random.randint(
                    low=1, high=10
                )
            else:
                meta[
                    "max_types_at_each_level"
                ] = obj._max_struct_types_at_each_level

        elif dtype == "decimal64":
            meta["max_precision"] = cudf.Decimal64Dtype.MAX_PRECISION
        elif dtype == "decimal32":
            meta["max_precision"] = cudf.Decimal32Dtype.MAX_PRECISION

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
        elif isinstance(column.type, pa.StructType):
            df[column._name] = column.to_pandas(integer_object_nulls=True)
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


def get_avro_dtype_info(dtype):
    if dtype in _PANDAS_TO_AVRO_SCHEMA_MAP:
        return _PANDAS_TO_AVRO_SCHEMA_MAP[dtype]
    else:
        raise TypeError(
            f"Unsupported dtype({dtype}) according to avro spec:"
            f" https://avro.apache.org/docs/current/spec.html"
        )


def get_orc_dtype_info(dtype):
    if dtype in PANDAS_TO_ORC_TYPES:
        return PANDAS_TO_ORC_TYPES[dtype]
    else:
        raise TypeError(
            f"Unsupported dtype({dtype}) according to orc spec:"
            f" https://orc.apache.org/specification/"
        )


def get_arrow_dtype_info_for_pyorc(dtype):
    if isinstance(dtype, pa.StructType):
        return get_orc_schema(df=None, arrow_table_schema=dtype)
    else:
        pd_dtype = cudf.dtype(dtype.to_pandas_dtype())
        return get_orc_dtype_info(pd_dtype)


def get_avro_schema(df):
    fields = [
        {"name": col_name, "type": get_avro_dtype_info(col_dtype)}
        for col_name, col_dtype in df.dtypes.items()
    ]
    schema = {"type": "record", "name": "Root", "fields": fields}
    return schema


def get_orc_schema(df, arrow_table_schema=None):
    if arrow_table_schema is None:
        ordered_dict = OrderedDict(
            (col_name, get_orc_dtype_info(col_dtype))
            for col_name, col_dtype in df.dtypes.items()
        )
    else:
        ordered_dict = OrderedDict(
            (field.name, get_arrow_dtype_info_for_pyorc(field.type))
            for field in arrow_table_schema
        )

    schema = pyorc.Struct(**ordered_dict)
    return schema


def convert_nulls_to_none(records, df):
    columns_with_nulls = {col for col in df.columns if df[col].isnull().any()}
    scalar_columns_convert = [
        col
        for col in df.columns
        if df[col].dtype in pandas_dtypes_to_np_dtypes
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


def _preprocess_to_orc_tuple(df, arrow_table_schema):
    def _null_to_None(value):
        if value is pd.NA or value is pd.NaT:
            return None
        else:
            return value

    def sanitize(value, struct_type):
        if value is None:
            return None

        values_list = []
        for name, sub_type in struct_type.fields.items():
            if isinstance(sub_type, cudf.StructDtype):
                values_list.append(sanitize(value[name], sub_type))
            else:
                values_list.append(value[name])
        return tuple(values_list)

    has_nulls_or_nullable_dtype = any(
        (col := df[colname]).dtype in pandas_dtypes_to_np_dtypes
        or col.isnull().any()
        for colname in df.columns
    )
    pdf = df.copy(deep=True)
    for field in arrow_table_schema:
        if isinstance(field.type, pa.StructType):
            pdf[field.name] = pdf[field.name].apply(
                sanitize, args=(cudf.StructDtype.from_arrow(field.type),)
            )
        else:
            pdf[field.name] = pdf[field.name]

    tuple_list = [
        tuple(map(_null_to_None, tup)) if has_nulls_or_nullable_dtype else tup
        for tup in pdf.itertuples(index=False, name=None)
    ]

    return tuple_list, pdf, df


def pandas_to_orc(
    df,
    file_name=None,
    file_io_obj=None,
    stripe_size=67108864,
    arrow_table_schema=None,
):
    schema = get_orc_schema(df, arrow_table_schema=arrow_table_schema)

    tuple_list, pdf, df = _preprocess_to_orc_tuple(
        df, arrow_table_schema=arrow_table_schema
    )

    if file_name is not None:
        with open(file_name, "wb") as data:
            with pyorc.Writer(data, schema, stripe_size=stripe_size) as writer:
                writer.writerows(tuple_list)
    elif file_io_obj is not None:
        with pyorc.Writer(
            file_io_obj, schema, stripe_size=stripe_size
        ) as writer:
            writer.writerows(tuple_list)


def orc_to_pandas(file_name=None, file_io_obj=None, stripes=None):
    if file_name is not None:
        f = open(file_name, "rb")
    elif file_io_obj is not None:
        f = file_io_obj

    reader = pyorc.Reader(f)

    dtypes = {
        col: ORC_TO_PANDAS_TYPES[pyorc_type.name]
        for col, pyorc_type in reader.schema.fields.items()
    }

    if stripes is None:
        df = pd.DataFrame.from_records(
            reader, columns=reader.schema.fields.keys()
        )
    else:
        records = [
            record for i in stripes for record in list(reader.read_stripe(i))
        ]
        df = pd.DataFrame.from_records(
            records, columns=reader.schema.fields.keys()
        )

    # Need to type-cast to extracted `dtypes` from pyorc schema because
    # a fully empty/ full <NA> can result in incorrect dtype by pandas.
    df = df.astype(dtypes)

    return df


def compare_dataframe(left, right, nullable=True):
    if nullable and isinstance(left, cudf.DataFrame):
        left = left.to_pandas(nullable=True)
    if nullable and isinstance(right, cudf.DataFrame):
        right = right.to_pandas(nullable=True)

    if len(left.index) == 0 and len(right.index) == 0:
        check_index_type = False
    else:
        check_index_type = True

    return assert_eq(left, right, check_index_type=check_index_type)
