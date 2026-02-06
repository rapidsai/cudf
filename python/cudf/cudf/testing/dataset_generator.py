# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This module is for generating "synthetic" datasets. It was originally
# designed for testing filtered reading. Generally, it should be useful
# if you want to generate data where certain phenomena (e.g., cardinality)
# are exaggerated.

import string
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

import cudf
from cudf.utils.dtypes import cudf_dtype_to_pa_type


class ColumnParameters:
    """Parameters for generating column of data

    Attributes
    ----------
    cardinality : int or None
        Size of a random set of values that generated data is sampled from.
        The values in the random set are derived from the given generator.
        If cardinality is None, the Iterable returned by the given generator
        is invoked for each value to be generated.
    null_frequency : 0.1
        Probability of a generated value being null
    generator : Callable
        Function for generating random data.
    is_sorted : bool
        Sort this column. Columns are sorted in same order as ColumnParameters
        instances stored in column_params of Parameters. If there are one or
        more columns marked as sorted, the generated PyArrow Table will be
        converted to a Pandas DataFrame to do the sorting. This may implicitly
        convert numbers to floats in the presence of nulls.
    dtype : optional
        a numpy dtype to control the format of the data
    """

    def __init__(
        self,
        cardinality=100,
        null_frequency=0.1,
        generator=None,
        is_sorted=True,
        dtype=None,
    ):
        self.cardinality = cardinality
        self.null_frequency = null_frequency
        if generator is None:
            rng = np.random.default_rng(seed=0)
            self.generator = lambda: [
                _generate_string(
                    string.ascii_letters, rng, rng.integers(4, 8).item()
                )
                for _ in range(100)
            ]
        else:
            self.generator = generator
        self.is_sorted = is_sorted
        self.dtype = dtype


class Parameters:
    """Parameters for random dataset generation

    Attributes
    ----------
    num_rows : int
        Number of rows to generate
    column_parameters : List[ColumnParams]
        ColumnParams for each column
    seed : int or None, default None
        Seed for random data generation
    """

    def __init__(
        self,
        num_rows=2048,
        column_parameters=None,
        seed=None,
    ):
        self.num_rows = num_rows
        if column_parameters is None:
            column_parameters = []
        self.column_parameters = column_parameters
        self.seed = seed


def _write(tbl, path, format):
    if format["name"] == "parquet":
        if isinstance(tbl, pa.Table):
            pq.write_table(tbl, path, row_group_size=format["row_group_size"])
        elif isinstance(tbl, pd.DataFrame):
            tbl.to_parquet(path, row_group_size=format["row_group_size"])


def _generate_column(column_params, num_rows, rng):
    # If cardinality is specified, we create a set to sample from.
    # Otherwise, we simply use the given generator to generate each value.

    if column_params.cardinality is not None:
        # Construct set of values to sample from where
        # set size = cardinality

        if (
            isinstance(column_params.dtype, str)
            and column_params.dtype == "category"
        ):
            vals = pa.array(
                column_params.generator,
                size=column_params.cardinality,
                safe=False,
            )
            return pa.DictionaryArray.from_arrays(
                dictionary=vals,
                indices=rng.integers(low=0, high=len(vals), size=num_rows),
                mask=rng.choice(
                    [True, False],
                    size=num_rows,
                    p=[
                        column_params.null_frequency,
                        1 - column_params.null_frequency,
                    ],
                )
                if column_params.null_frequency > 0.0
                else None,
            )

        if hasattr(column_params.dtype, "to_arrow"):
            arrow_type = column_params.dtype.to_arrow()
        elif column_params.dtype is not None:
            arrow_type = cudf_dtype_to_pa_type(cudf.dtype(column_params.dtype))
        else:
            arrow_type = None

        if isinstance(column_params.dtype, cudf.StructDtype):
            vals = pa.StructArray.from_arrays(
                column_params.generator,
                names=column_params.dtype.fields.keys(),
                mask=pa.array(
                    rng.choice(
                        [True, False],
                        size=num_rows,
                        p=[
                            column_params.null_frequency,
                            1 - column_params.null_frequency,
                        ],
                    )
                )
                if column_params.null_frequency > 0.0
                else None,
            )
            return vals
        elif not isinstance(arrow_type, pa.lib.Decimal128Type):
            vals = pa.array(
                column_params.generator,
                size=column_params.cardinality,
                safe=False,
                type=arrow_type,
            )
        vals = pa.array(
            rng.choice(column_params.generator, size=num_rows)
            if isinstance(arrow_type, pa.lib.Decimal128Type)
            else rng.choice(vals, size=num_rows),
            mask=rng.choice(
                [True, False],
                size=num_rows,
                p=[
                    column_params.null_frequency,
                    1 - column_params.null_frequency,
                ],
            )
            if column_params.null_frequency > 0.0
            else None,
            size=num_rows,
            safe=False,
            type=None
            if isinstance(arrow_type, pa.lib.Decimal128Type)
            else arrow_type,
        )
        if isinstance(arrow_type, pa.lib.Decimal128Type):
            vals = vals.cast(arrow_type, safe=False)
        return vals
    else:
        # Generate data for current column
        return pa.array(
            column_params.generator,
            mask=rng.choice(
                [True, False],
                size=num_rows,
                p=[
                    column_params.null_frequency,
                    1 - column_params.null_frequency,
                ],
            )
            if column_params.null_frequency > 0.0
            else None,
            size=num_rows,
            safe=False,
        )


def generate(
    path,
    parameters,
    format=None,
    use_threads=True,
):
    """
    Generate dataset using given parameters and write to given format

    Parameters
    ----------
    path : str or file-like object
        Path to write to
    parameters : Parameters
        Parameters specifying how to randomly generate data
    format : Dict
        Format to write
    """
    if format is None:
        format = {"name": "parquet", "row_group_size": 64}
    df = get_dataframe(parameters, use_threads)

    # Write
    _write(df, path, format)


def get_dataframe(parameters, use_threads):
    # Initialize seeds
    if parameters.seed is not None:
        rng = np.random.default_rng(seed=parameters.seed)
    else:
        rng = np.random.default_rng(seed=0)

    # For each column, invoke the data generator
    for column_params in parameters.column_parameters:
        column_params.generator = column_params.generator()

    # Get schema for each column
    table_fields = []
    for i, column_params in enumerate(parameters.column_parameters):
        if (
            isinstance(column_params.dtype, str)
            and column_params.dtype == "category"
        ):
            arrow_type = pa.dictionary(
                index_type=pa.int64(),
                value_type=cudf_dtype_to_pa_type(
                    cudf.dtype(type(next(iter(column_params.generator))))
                ),
            )
        elif hasattr(column_params.dtype, "to_arrow"):
            arrow_type = column_params.dtype.to_arrow()
        else:
            arrow_type = cudf_dtype_to_pa_type(
                cudf.dtype(type(next(iter(column_params.generator))))
                if column_params.dtype is None
                else column_params.dtype
            )
        table_fields.append(
            pa.field(
                name=str(i),
                type=arrow_type,
                nullable=column_params.null_frequency > 0,
            )
        )

    schema = pa.schema(table_fields)

    # Initialize column data and which columns should be sorted
    column_data = [None] * len(parameters.column_parameters)
    columns_to_sort = [
        str(i)
        for i, column_params in enumerate(parameters.column_parameters)
        if column_params.is_sorted
    ]
    # Generate data
    if not use_threads:
        for i, column_params in enumerate(parameters.column_parameters):
            column_data[i] = _generate_column(
                column_params,
                parameters.num_rows,
                rng,
            )
    else:
        pool = Pool(pa.cpu_count())
        column_data = pool.starmap(
            _generate_column,
            [
                (column_params, parameters.num_rows, rng)
                for i, column_params in enumerate(parameters.column_parameters)
            ],
        )
        pool.close()
        pool.join()
    # Convert to Pandas DataFrame and sort columns appropriately
    tbl = pa.Table.from_arrays(
        column_data,
        schema=schema,
    )
    if columns_to_sort:
        tbl = tbl.to_pandas()
        tbl = tbl.sort_values(columns_to_sort)
        tbl = pa.Table.from_pandas(tbl, schema)
    return tbl


def _generate_string(str_seq: str, rng, length: int = 10) -> str:
    return "".join(rng.choice(list(str_seq), size=length))
