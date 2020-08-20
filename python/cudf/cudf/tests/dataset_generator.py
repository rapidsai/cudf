# Copyright (c) 2020, NVIDIA CORPORATION.

# This module is for generating "synthetic" datasets. It was originally
# designed for testing filtered reading. Generally, it should be useful
# if you want to generate data where certain phenomena (e.g., cardinality)
# are exaggurated.

import pandas as pd
import numpy as np

import pyarrow as pa
from mimesis import Generic


class ColumnParameters:
    """Parameters for generating column of data

    Attributes
    ---
    cardinality : int
        Number of rows to generate
    null_frequency : 0.1
        Number of columns to generate
    generator : Callable
        Function for generating random data. It is passed a Mimesis Generic
        provider and returns a callable that generates data.
    is_sorted : bool
        Sort this column. Columns are sorted in same order as ColumnParameters
        instances stored in column_params of Parameters.
    """

    def __init__(
        self,
        cardinality=100,
        null_frequency=0.1,
        generator=lambda g: g.address.country,
        is_sorted=True,
    ):
        self.cardinality = cardinality
        self.null_frequency = null_frequency
        self.generator = generator
        self.is_sorted = is_sorted


class Parameters:
    """Parameters for random dataset generation

    Attributes
    ---
    num_rows : int
        Number of rows to generate
    column_params : List[ColumnParams]
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


SIMPLE = Parameters(
    num_rows=2048,
    column_parameters=[
        ColumnParameters(
            cardinality=100,
            null_frequency=0.05,
            generator=lambda g: g.address.country,
            is_sorted=False,
        ),
        ColumnParameters(
            cardinality=40,
            null_frequency=0.2,
            generator=lambda g: g.person.age,
            is_sorted=True,
        ),
        ColumnParameters(
            cardinality=30,
            null_frequency=0.1,
            generator=lambda g: g.text.color,
            is_sorted=False,
        ),
        ColumnParameters(
            cardinality=10,
            null_frequency=0.1,
            generator=lambda g: g.hardware.manufacturer,
            is_sorted=False,
        ),
    ],
    seed=0,
)


def generate(
    path, parameters, format={"name": "parquet", "row_group_size": 64}
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

    # Initialize seeds
    g = Generic("en", seed=parameters.seed)
    if parameters.seed is not None:
        np.random.seed(parameters.seed)

    # Generate data for each column in Arrow Table
    schema = pa.schema(
        [
            pa.field(
                name=str(i),
                type=pa.from_numpy_dtype(type(column_params.generator(g)())),
                nullable=column_params.null_frequency > 0,
            )
            for i, column_params in enumerate(parameters.column_parameters)
        ]
    )
    column_data = []
    columns_to_sort = [
        str(i)
        for i, column_params in enumerate(parameters.column_parameters)
        if column_params.is_sorted
    ]
    for i, column_params in enumerate(parameters.column_parameters):
        vals = pa.array(
            (
                column_params.generator(g)()
                for _ in range(column_params.cardinality)
            ),
            size=column_params.cardinality,
        )

        # Generate data for current column
        column_data.append(
            pa.array(
                (
                    None
                    if np.random.rand() < column_params.null_frequency
                    else np.random.choice(vals)
                    for _ in range(parameters.num_rows)
                ),
                size=parameters.num_rows,
            )
        )

    # Convert to Pandas DataFrame and sort columns appropriately
    df = pa.Table.from_arrays(column_data, schema=schema,).to_pandas()
    df = df.sort_values(columns_to_sort)

    # Write
    if format["name"] == "parquet":
        df.to_parquet(path, row_group_size=format["row_group_size"])
