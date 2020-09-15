# Copyright (c) 2020, NVIDIA CORPORATION.

# This module is for generating "synthetic" datasets. It was originally
# designed for testing filtered reading. Generally, it should be useful
# if you want to generate data where certain phenomena (e.g., cardinality)
# are exaggerated.

import random
import string
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
from mimesis import Generic
from pyarrow import parquet as pq


class ColumnParameters:
    """Parameters for generating column of data

    Attributes
    ---
    cardinality : int or None
        Size of a random set of values that generated data is sampled from.
        The values in the random set are derived from the given generator.
        If cardinality is None, the Iterable returned by the given generator
        is invoked for each value to be generated.
    null_frequency : 0.1
        Probability of a generated value being null
    generator : Callable
        Function for generating random data. It is passed a Mimesis Generic
        provider and returns an Iterable that generates data.
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
        generator=lambda g: [g.address.country for _ in range(100)],
        is_sorted=True,
        dtype=None,
    ):
        self.cardinality = cardinality
        self.null_frequency = null_frequency
        self.generator = generator
        self.is_sorted = is_sorted
        self.dtype = dtype


class Parameters:
    """Parameters for random dataset generation

    Attributes
    ---
    num_rows : int
        Number of rows to generate
    column_parameters : List[ColumnParams]
        ColumnParams for each column
    seed : int or None, default None
        Seed for random data generation
    """

    def __init__(
        self, num_rows=2048, column_parameters=[], seed=None,
    ):
        self.num_rows = num_rows
        self.column_parameters = column_parameters
        self.seed = seed


def _write(tbl, path, format):
    if format["name"] == "parquet":
        if isinstance(tbl, pa.Table):
            pq.write_table(tbl, path, row_group_size=format["row_group_size"])
        elif isinstance(tbl, pd.DataFrame):
            tbl.to_parquet(path, row_group_size=format["row_group_size"])


def _generate_column(column_params, num_rows):
    # If cardinality is specified, we create a set to sample from.
    # Otherwise, we simply use the given generator to generate each value.
    if column_params.cardinality is not None:
        # Construct set of values to sample from where
        # set size = cardinality
        vals = pa.array(
            column_params.generator,
            size=column_params.cardinality,
            safe=False,
        )

        if (
            isinstance(column_params.dtype, str)
            and column_params.dtype == "category"
        ):
            return pa.DictionaryArray.from_arrays(
                dictionary=vals,
                indices=np.random.choice(np.arange(len(vals)), size=num_rows),
                mask=np.random.choice(
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

        # Generate data for current column
        choices = np.random.randint(0, len(vals) - 1, size=num_rows)
        return pa.array(
            [vals[choices[i]].as_py() for i in range(num_rows)],
            mask=np.random.choice(
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

    else:
        # Generate data for current column
        return pa.array(
            column_params.generator,
            mask=np.random.choice(
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
    format={"name": "parquet", "row_group_size": 64},
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

    df = get_dataframe(parameters, use_threads)

    # Write
    _write(df, path, format)


def get_dataframe(parameters, use_threads):
    # Initialize seeds
    if parameters.seed is not None:
        np.random.seed(parameters.seed)
    column_seeds = np.arange(len(parameters.column_parameters))
    np.random.shuffle(column_seeds)
    # For each column, use a generic Mimesis producer to create an Iterable
    # for generating data
    for i, column_params in enumerate(parameters.column_parameters):
        column_params.generator = column_params.generator(
            Generic("en", seed=column_seeds[i])
        )
    # Get schema for each column
    schema = pa.schema(
        [
            pa.field(
                name=str(i),
                type=pa.dictionary(
                    index_type=pa.int64(),
                    value_type=pa.from_numpy_dtype(
                        type(next(iter(column_params.generator)))
                    ),
                )
                if isinstance(column_params.dtype, str)
                and column_params.dtype == "category"
                else pa.from_numpy_dtype(
                    type(next(iter(column_params.generator)))
                    if column_params.dtype is None
                    else column_params.dtype
                ),
                nullable=column_params.null_frequency > 0,
            )
            for i, column_params in enumerate(parameters.column_parameters)
        ]
    )

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
                column_params, parameters.num_rows
            )
    else:
        pool = Pool(pa.cpu_count())
        column_data = pool.starmap(
            _generate_column,
            [
                (column_params, parameters.num_rows)
                for i, column_params in enumerate(parameters.column_parameters)
            ],
        )
        pool.close()
        pool.join()
    # Convert to Pandas DataFrame and sort columns appropriately
    tbl = pa.Table.from_arrays(column_data, schema=schema,)
    if columns_to_sort:
        tbl = tbl.to_pandas()
        tbl = tbl.sort_values(columns_to_sort)
        tbl = pa.Table.from_pandas(tbl, schema)
    return tbl


def rand_dataframe(dtypes_meta, rows, seed=random.randint(0, 2 ** 32 - 1)):
    random.seed(seed)

    column_params = []
    for meta in dtypes_meta:
        dtype, null_frequency, cardinality = meta

        if dtype == "category":
            column_params.append(
                ColumnParameters(
                    cardinality=cardinality,
                    null_frequency=null_frequency,
                    generator=lambda g: [
                        g.random.randstr(unique=True, length=2000)
                        for _ in range(cardinality)
                    ],
                    is_sorted=False,
                    dtype="category",
                )
            )
        else:
            dtype = np.dtype(dtype)
            if dtype.kind in ("i", "u"):
                iinfo = np.iinfo(dtype)
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=lambda g: g.random.randints(
                            rows, iinfo.min, iinfo.max
                        ),
                        is_sorted=False,
                    )
                )
            elif dtype.kind == "f":
                finfo = np.finfo(dtype)
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=lambda g: np.random.uniform(
                            low=finfo.min / 2, high=finfo.max / 2, size=rows,
                        )
                        * 2,
                        is_sorted=False,
                    )
                )
            elif dtype.kind in ("U", "O"):
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=lambda g: [
                            g.random.schoice(string.printable, 2000)
                            for _ in range(rows)
                        ],
                        is_sorted=False,
                    )
                )
            elif dtype.kind == "M":
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=lambda g: g.random.randints(
                            rows, 0, 2147483647 - 1
                        ),
                        is_sorted=False,
                        dtype=np.dtype(dtype),
                    )
                )
            elif dtype.kind == "m":
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=lambda g: g.random.randints(
                            rows, -2147483648, 2147483647 - 1
                        ),
                        is_sorted=False,
                        dtype=np.dtype(dtype),
                    )
                )
            elif dtype.kind == "b":
                column_params.append(
                    ColumnParameters(
                        cardinality=cardinality,
                        null_frequency=null_frequency,
                        generator=lambda g: [
                            g.development.boolean() for _ in range(rows)
                        ],
                        is_sorted=False,
                        dtype=np.dtype(dtype),
                    )
                )
            else:
                raise TypeError(f"Unsupported dtype: {dtype}")

    df = get_dataframe(
        Parameters(num_rows=rows, column_parameters=column_params, seed=seed,),
        use_threads=True,
    )

    return df
